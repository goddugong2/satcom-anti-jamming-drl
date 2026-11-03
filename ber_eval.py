"""물리계층(BER, J/S) 평가 모듈.

validate : 시뮬레이터 검증
   (a) AWGN에서 BPSK BER 시뮬레이션이 이론식 Q(sqrt(2 Eb/N0))와 맞는지
   (b) 광대역 백색 재머가 있을 때 SINR 기반 이론과 맞는지
   (c) 재머 종류별(단일톤, 스윕, 광대역, 부분대역) BER vs J/S 곡선
link     : run_e2e.py compare 결과(JSON)의 방법별 성공률을 링크 BER로 환산
   슬롯의 h(= 1 - 성공률)만큼 재밍을 받는다고 보고
       BER_평균 = (1 - h) * BER_정상 + h * BER_피격(재머 종류, SNR, J/S)

사용:
    python ber_eval.py validate
    python ber_eval.py link --compare compare_multi4_c32_h4_d0_bin.json

가정 (README에 그대로 적을 것):
  - 수신기: 반송파 위상을 아는 동기 검파(상관기, 적분-덤프). BPSK, 심볼당 SYMBOL_LEN 샘플
  - SNR, J/S(= JSR)는 신호 전력 기준, 잡음과 재머 전력은 샘플링 대역 전체에 대한 값
  - Eb/N0 = (SYMBOL_LEN / 2) * SINR,  SINR = 신호 전력 / (잡음 전력 + 재머 전력)
  - 피격 슬롯에서는 재머 파형이 그 채널에 들어온다. 재머의 주파수와 대역은 채널 안에서 무작위
  - 정상 슬롯의 BER은 이론식을 쓴다 (실제 Eb/N0가 높아 몬테카를로로는 0이 나오기 때문)
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from signal_gen import N, SYMBOL_LEN, JAMMER_FUNCS, scale_to_power, get_rng, T_AXIS

NSYM = N // SYMBOL_LEN
JAM_TYPES = ["single_tone", "sweep", "noise", "partial_band"]
JAM_LABEL = {"single_tone": "Single-tone", "sweep": "Swept chirp",
             "noise": "Broadband noise", "partial_band": "Partial-band noise"}
JAM_COLOR = {"single_tone": "tab:red", "sweep": "tab:green", "noise": "tab:blue",
             "partial_band": "tab:orange"}
EB_OFFSET_DB = 10 * np.log10(SYMBOL_LEN / 2)       # Eb/N0(dB) = SINR(dB) + 이 값


# ---------------- 이론 ----------------
def q_func(x):
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def ber_theory(snr_db, jsr_db=None):
    """동기 검파 BPSK의 이론 BER. 재머가 있으면 백색 잡음처럼 SINR에 합산(광대역 재머의 이론)."""
    inv = 1.0 / 10 ** (np.asarray(snr_db, dtype=float) / 10)
    if jsr_db is not None:
        inv = inv + 10 ** (np.asarray(jsr_db, dtype=float) / 10)
    sinr = 1.0 / inv
    ebn0 = (SYMBOL_LEN / 2) * sinr
    return q_func(np.sqrt(2.0 * ebn0))


# ---------------- 몬테카를로 ----------------
def bpsk_frame(rng, integer_cycles=False):
    """BPSK 한 프레임. 평균 전력 1로 정규화. 반환: (tx, bits, carrier)

    integer_cycles=True면 한 심볼에 반송파가 정수 주기 들어가도록 고정(이론식과의 정확한 대조용).
    """
    fc = float(rng.choice([300, 400, 500, 600, 700])) if integer_cycles else rng.uniform(250, 750)
    phase0 = rng.uniform(0, 2 * np.pi)
    bits = rng.choice([-1.0, 1.0], size=NSYM)
    carrier = np.cos(2 * np.pi * fc * T_AXIS + phase0)
    tx = np.repeat(bits, SYMBOL_LEN) * carrier
    tx = tx / np.sqrt(np.mean(tx ** 2))
    return tx, bits, carrier


def ber_mc(snr_db, jam_type=None, jsr_db=0.0, frames=3000, seed=0, integer_cycles=False, white_jam=False):
    """몬테카를로 BER. 반환: (BER, 오류 수, 전체 비트 수)."""
    rng = get_rng(seed)
    snr, jsr = 10 ** (snr_db / 10), 10 ** (jsr_db / 10)
    errs, total = 0, 0
    for _ in range(frames):
        tx, bits, carrier = bpsk_frame(rng, integer_cycles)
        rx = tx + rng.normal(0, np.sqrt(1.0 / snr), N)
        if white_jam:
            rx = rx + rng.normal(0, np.sqrt(jsr), N)
        elif jam_type is not None:
            rx = rx + scale_to_power(JAMMER_FUNCS[jam_type](rng), jsr)
        d = (rx * carrier).reshape(NSYM, SYMBOL_LEN).sum(1)       # 적분-덤프
        errs += int(((d >= 0) != (bits > 0)).sum())
        total += NSYM
    return errs / total, errs, total


def zscore(errs, total, p):
    """이론 BER p 아래에서 관측 오류 수가 얼마나 벗어났는지 (표준편차 단위). 기대 오류 10개 미만이면 None."""
    exp = total * p
    if exp < 10:
        return None
    return float((errs - exp) / np.sqrt(total * p * (1 - p)))


def floor_plot(ber, total):
    """로그 그래프용: 오류 0개면 0.5/전체 비트 수 위치에 찍는다."""
    return max(ber, 0.5 / total)


def fmt(v):
    return "<1e-6" if v < 1e-6 else f"{v:.2e}"


# ---------------- validate ----------------
def cmd_validate(args):
    t0 = time.time()
    print(f"[설정] 프레임당 심볼 {NSYM}개, 심볼당 샘플 {SYMBOL_LEN}, 점당 프레임 {args.frames} "
          f"(= {args.frames * NSYM:,} 비트)\n")

    print("(a) AWGN에서 BPSK BER: 시뮬레이션 vs 이론 Q(sqrt(2 Eb/N0))")
    snr_a = np.arange(-12.0, -1.0, 1.0)
    rows_a = []
    for i, s in enumerate(snr_a):
        ber, e, tot = ber_mc(s, frames=args.frames, seed=args.seed + i, integer_cycles=True)
        th = float(ber_theory(s))
        z = zscore(e, tot, th)
        rows_a.append({"ebn0_db": float(s + EB_OFFSET_DB), "sim": ber, "theory": th, "errs": e, "bits": tot, "z": z})
        zs = "  -  " if z is None else f"{z:+5.1f}"
        print(f"   Eb/N0 {s + EB_OFFSET_DB:5.1f} dB | 시뮬 {ber:.3e} | 이론 {th:.3e} | z {zs}")
    zs_a = [abs(r["z"]) for r in rows_a if r["z"] is not None]
    print(f"   -> 이론과의 최대 편차 |z| = {max(zs_a):.1f} (3 미만이면 통계 오차 범위)\n")

    print(f"(b) 광대역 백색 재머 (SNR {args.snr:.0f} dB 고정): 시뮬레이션 vs SINR 기반 이론")
    jsr_list = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
    rows_b = []
    for i, j in enumerate(jsr_list):
        ber, e, tot = ber_mc(args.snr, jsr_db=j, frames=args.frames, seed=args.seed + 100 + i,
                             integer_cycles=True, white_jam=True)
        th = float(ber_theory(args.snr, j))
        z = zscore(e, tot, th)
        rows_b.append({"jsr_db": j, "sim": ber, "theory": th, "errs": e, "bits": tot, "z": z})
        zs = "  -  " if z is None else f"{z:+5.1f}"
        print(f"   J/S {j:4.0f} dB | 시뮬 {ber:.3e} | 이론 {th:.3e} | z {zs}")
    zs_b = [abs(r["z"]) for r in rows_b if r["z"] is not None]
    if zs_b:
        print(f"   -> 최대 |z| = {max(zs_b):.1f}\n")

    print(f"(c) 재머 종류별 BER vs J/S (SNR {args.snr:.0f} dB, 반송파/재머 주파수는 무작위)")
    rows_c = {}
    for k, t in enumerate(JAM_TYPES):
        rows_c[t] = []
        for i, j in enumerate(jsr_list):
            ber, e, tot = ber_mc(args.snr, t, j, frames=args.frames, seed=args.seed + 200 + 10 * k + i)
            rows_c[t].append({"jsr_db": j, "ber": ber, "errs": e, "bits": tot})
        print(f"   {JAM_LABEL[t]:19s}", " ".join(f"{r['ber']:.1e}" for r in rows_c[t]))
    print(f"   {'(J/S dB)':19s}", " ".join(f"{j:7.0f}" for j in jsr_list))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    xs = np.linspace(-2, 8, 200)
    a1.semilogy(xs, q_func(np.sqrt(2 * 10 ** (xs / 10))), "k-", label="Theory  Q(sqrt(2 Eb/N0))")
    a1.semilogy([r["ebn0_db"] for r in rows_a], [floor_plot(r["sim"], r["bits"]) for r in rows_a],
                "o", color="tab:blue", label="Simulation")
    a1.set_xlabel("Eb/N0 (dB)"); a1.set_ylabel("BER"); a1.set_ylim(1e-5, 1)
    a1.set_title("BPSK in AWGN: simulation vs theory"); a1.grid(ls=":", which="both"); a1.legend()

    xj = np.linspace(0, 18, 200)
    a2.semilogy(xj, ber_theory(args.snr, xj), "k-", label="Theory (white jammer, SINR)")
    a2.semilogy([r["jsr_db"] for r in rows_b], [floor_plot(r["sim"], r["bits"]) for r in rows_b],
                "o", color="gray", label="Sim: white jammer")
    for t in JAM_TYPES:
        a2.semilogy(jsr_list, [floor_plot(r["ber"], r["bits"]) for r in rows_c[t]], "-s", ms=4,
                    color=JAM_COLOR[t], label=JAM_LABEL[t])
    a2.set_xlabel("J/S (dB)"); a2.set_ylabel("BER"); a2.set_ylim(1e-5, 1)
    a2.set_title(f"BER vs J/S by jammer type (SNR {args.snr:.0f} dB)"); a2.grid(ls=":", which="both")
    a2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("ber_validation.png", dpi=250)
    with open("ber_validation.json", "w") as f:
        json.dump({"snr_db": args.snr, "awgn": rows_a, "white_jammer": rows_b, "by_type": rows_c}, f, indent=2)
    print(f"\nsaved ber_validation.png/.json  ({time.time() - t0:.0f}s)")


# ---------------- link ----------------
def cmd_link(args):
    t0 = time.time()
    with open(args.compare) as f:
        d = json.load(f)
    if "summary" not in d or "args" not in d:
        raise SystemExit("compare_*.json 파일(compare 실험 결과)을 지정하세요.")
    snr, jsr = float(d["args"]["snr"]), float(d["args"]["jsr"])
    summary = d["summary"]
    ber_clean = float(ber_theory(snr))
    print(f"[입력] {os.path.basename(args.compare)} | 재머={d['args']['jammer']} 채널={d['args']['channels']} | "
          f"SNR {snr:g} dB, J/S {jsr:g} dB")
    print(f"[링크] 정상 슬롯 BER(이론) = {ber_clean:.2e}  (Eb/N0 = {snr + EB_OFFSET_DB:.1f} dB)\n")

    ber_hit, info = {}, {}
    for k, t in enumerate(args.jam_types):
        b, e, tot = ber_mc(snr, t, jsr, frames=args.frames, seed=args.seed + k)
        ber_hit[t] = b
        info[t] = {"ber": b, "errs": e, "bits": tot}
        print(f"  피격 슬롯 BER [{JAM_LABEL[t]:19s}] {b:.3e}  (오류 {e} / {tot:,} 비트)")

    methods = [m for m in ("fixed", "random", "fhss", "avoid", "afh", "qtable", "qtable2", "qtable4",
                           "qmask", "dqn_oracle", "dqn_cnn") if m in summary]
    table = {}
    print("\n방법별 평균 BER (= 정상 슬롯 BER과 피격 슬롯 BER의 가중 평균)")
    print(f"  {'method':11s} {'성공률':>7s} {'피격률':>7s} | " + " ".join(f"{t[:12]:>12s}" for t in args.jam_types))
    for m in methods:
        succ = summary[m]["mean"]
        h = 1.0 - succ / 100.0
        table[m] = {"success": succ, "hit_rate": h,
                    "ber": {t: (1 - h) * ber_clean + h * ber_hit[t] for t in args.jam_types}}
        print(f"  {m:11s} {succ:6.2f}% {h * 100:6.2f}% | "
              + " ".join(f"{fmt(table[m]['ber'][t]):>12s}" for t in args.jam_types))

    if "random" in table and "dqn_cnn" in table:
        print("\nDQN(CNN 관측)의 평균 BER 감소 배율 (랜덤 호핑의 BER / DQN의 BER)")
        for t in args.jam_types:
            r, q = table["random"]["ber"][t], table["dqn_cnn"]["ber"][t]
            print(f"  {JAM_LABEL[t]:19s} x{r / q:.1f}" if q > 0 else f"  {JAM_LABEL[t]:19s} -")

    fig, ax = plt.subplots(figsize=(10.5, 5))
    w = 0.8 / len(args.jam_types)
    x = np.arange(len(methods))
    for i, t in enumerate(args.jam_types):
        vals = [max(table[m]["ber"][t], 1e-7) for m in methods]
        ax.bar(x + (i - (len(args.jam_types) - 1) / 2) * w, vals, w, color=JAM_COLOR[t], label=JAM_LABEL[t])
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20)
    ax.set_ylabel("Average BER")
    ax.set_title(f"Link-level BER by hopping method | {d['args']['jammer']}, C={d['args']['channels']}, "
                 f"SNR {snr:g} dB, J/S {jsr:g} dB", fontsize=10)
    ax.grid(axis="y", ls=":", which="both"); ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    stem = os.path.splitext(os.path.basename(args.compare))[0]
    plt.savefig(f"ber_link_{stem}.png", dpi=250)
    with open(f"ber_link_{stem}.json", "w") as f:
        json.dump({"source": args.compare, "snr": snr, "jsr": jsr, "ber_clean_theory": ber_clean,
                   "hit_slot_ber": info, "methods": table}, f, indent=2)
    print(f"\nsaved ber_link_{stem}.png/.json  ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate")
    v.add_argument("--frames", type=int, default=3000, help="점당 프레임 수 (프레임당 50비트)")
    v.add_argument("--snr", type=float, default=10.0)
    v.add_argument("--seed", type=int, default=0)
    v.set_defaults(fn=cmd_validate)

    l = sub.add_parser("link")
    l.add_argument("--compare", required=True, help="run_e2e.py compare가 만든 compare_*.json")
    l.add_argument("--jam-types", nargs="+", choices=JAM_TYPES, default=JAM_TYPES)
    l.add_argument("--frames", type=int, default=4000)
    l.add_argument("--seed", type=int, default=0)
    l.set_defaults(fn=cmd_link)

    args = ap.parse_args()
    fn = args.fn
    del args.fn
    fn(args)


if __name__ == "__main__":
    main()
