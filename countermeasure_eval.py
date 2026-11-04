"""1단계: 재머 종류에 맞는 대응을 고르는 항재밍 알고리즘의 링크 수준 평가.

매 슬롯(샘플 1000개 = BPSK 50심볼)마다 실제 파형을 보내고 받아 비트 오류를 세고,
대응별 처리량(슬롯당 정보 비트, 성공한 프레임만 인정)을 계산한다.

대응 (행동)
  plain-*  : 일반 BPSK 그대로 (부호 none / BCH t=2, 5, 7)
  excise-* : 수신 신호의 STFT에서 미리 정한 기준보다 센 칸을 지우고 복원한 뒤 복조 (시간-주파수 간섭 제거)
  dsss2/4  : 같은 대역폭에서 비트 하나를 2 / 4개 칩으로 확산 (PN 부호), 정보량 1/2, 1/4
  hop-*    : 다른 채널로 도약. 도약 실패 확률은 재머 종류별 가정값, 정착 시간 비용 5%
  부호      : 단축 BCH (50,38,t=2), (50,23,t=5), (50,11,t=7). 한계거리 복호 = 오류 수가 t 이하이면 성공

전략 = '무엇을 관측하는가'. 관측값마다 처리량이 가장 큰 행동을 설계용 절반 데이터에서 고르고(표), 나머지
절반에서 평가한다. 재밍 슬롯과 정상 슬롯의 비중은 --clean-weight (기본 0.5).
  none        : 대응 없음 (plain-none 고정)
  always_hop  : 재밍을 정확히 감지하면 항상 도약(부호 없음) = 지금까지의 방식
  detect_fixed: 재밍 여부만 정확히 앎 -> 가장 좋은 고정 대응 하나
  energy      : 에너지 검출기(오탐 1%로 기준 설정)의 재밍 판정 -> 고정 대응
  energy_pwr  : 에너지 검출기 + 추정 J/S 구간 (세기는 알지만 종류는 모름)
  type        : 재머 종류를 정확히 앎 (CNN의 상한)
  type_pwr    : 종류 + 추정 J/S 구간
  cnn         : CNN이 예측한 클래스 (--cnn-path 를 주면)
  cnn_pwr     : CNN 클래스 + 추정 J/S 구간
  oracle      : 종류와 실제 J/S를 모두 앎 (상한)

사용:
    python countermeasure_eval.py --out-tag awgn                       # CNN 없이 (torch 불필요)
    python countermeasure_eval.py --cnn-path cnn_best.pt --out-tag awgn_cnn

가정 (README에 그대로 적을 것)
  - 시뮬레이션 스케일: FS 2 kHz, 반송파 500 Hz 고정, 심볼(칩) 20샘플. SNR, J/S는 신호 전력 기준(샘플링 대역 전체)
  - 수신기: 반송파와 PN 부호를 아는 동기 검파. 감지는 일반 BPSK 수신 구간에서 한다고 가정
  - 도약 실패 확률(재밍 채널로 다시 갈 확률): 단일톤 0.05, 스윕 0.25, 광대역 잡음 1.0(전 대역 재밍), 부분대역 0.10
  - CNN(cnn_best.pt)은 4클래스(none, single_tone, sweep, noise)로 학습되어 부분대역 잡음은 처음 보는 재머다
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps

import signal_gen as sg

FS, N, SL = sg.FS, sg.N, sg.SYMBOL_LEN
NSYM = N // SL                                        # 50
FC = 500.0
T = sg.T_AXIS
JAM_TYPES = ["single_tone", "sweep", "noise", "partial_band"]
JAM_LABEL = {"single_tone": "Single-tone", "sweep": "Swept chirp", "noise": "Broadband noise",
             "partial_band": "Partial-band noise"}
CODES = {"none": (50, 0), "t2": (38, 2), "t5": (23, 5), "t7": (11, 7)}   # 정보 비트, 정정 능력
HOP_COST = 0.05
DEFAULT_HOP_FAIL = {"single_tone": 0.05, "sweep": 0.25, "noise": 1.0, "partial_band": 0.10}
ACTIONS = ([f"plain-{c}" for c in CODES] + [f"excise-{c}" for c in CODES] + ["dsss2", "dsss4"]
           + [f"hop-{c}" for c in CODES])
STRATS = ["none", "always_hop", "detect_fixed", "energy", "energy_pwr", "type", "type_pwr",
          "cnn", "cnn_pwr", "oracle"]
SLABEL = {"none": "No countermeasure", "always_hop": "Always hop (current approach)",
          "detect_fixed": "Perfect detection + best fixed action", "energy": "Energy detector + fixed action",
          "energy_pwr": "Energy detector + J/S bins (no type)", "type": "True jammer type",
          "type_pwr": "True type + J/S bins", "cnn": "CNN class", "cnn_pwr": "CNN class + J/S bins",
          "oracle": "Oracle (type and true J/S)"}
SCOLOR = {"none": "lightgray", "always_hop": "tab:blue", "detect_fixed": "tab:cyan", "energy": "tab:gray",
          "energy_pwr": "black", "type": "tab:orange", "type_pwr": "tab:red", "cnn": "tab:purple",
          "cnn_pwr": "tab:pink", "oracle": "tab:green"}
PWR_EDGES = [-7.5, -2.5, 2.5, 7.5, 12.5]                # 추정 J/S(dB) 구간 경계


# ---------------- 파형 ----------------
def carrier(rng):
    return np.cos(2 * np.pi * FC * T + rng.uniform(0, 2 * np.pi))


def tx_plain(bits, car):
    x = np.repeat(bits, SL) * car
    return x / np.sqrt(np.mean(x ** 2))


def tx_dsss(bits, pn, k, car):
    chips = np.zeros(NSYM)
    chips[:len(bits) * k] = np.repeat(bits, k) * pn
    x = np.repeat(chips, SL) * car
    return x / np.sqrt(np.mean(x ** 2))


def rx_plain(rx, car):
    return (rx * car).reshape(NSYM, SL).sum(1)


def rx_dsss(rx, pn, k, nb, car):
    chip = (rx * car).reshape(NSYM, SL).sum(1)[:nb * k] * pn
    return chip.reshape(nb, k).sum(1)


def excise(rx, tau):
    _, _, z = sps.stft(rx, fs=FS, nperseg=64, noverlap=48)
    z[np.abs(z) > tau] = 0.0
    _, y = sps.istft(z, fs=FS, nperseg=64, noverlap=48)
    return y[:N]


def calibrate(rng, snr_db, n=300):
    """정상 슬롯으로 간섭 제거 기준(|STFT| 99.9 백분위)과 에너지 통계 분포를 정한다."""
    mags, pw = [], []
    for _ in range(n):
        car = carrier(rng)
        rx = tx_plain(rng.choice([-1.0, 1.0], NSYM), car) + rng.normal(0, np.sqrt(10 ** (-snr_db / 10)), N)
        _, _, z = sps.stft(rx, fs=FS, nperseg=64, noverlap=48)
        mags.append(np.abs(z).ravel())
        pw.append(np.mean(rx ** 2))
    return float(np.quantile(np.concatenate(mags), 0.999)), np.array(pw)


# ---------------- 한 슬롯 ----------------
def one_frame(rng, jtype, jsr_db, snr_db, tau, hop_fail):
    car = carrier(rng)
    sig_n = np.sqrt(10 ** (-snr_db / 10))
    bits = rng.choice([-1.0, 1.0], NSYM)
    b2, b4 = rng.choice([-1.0, 1.0], NSYM // 2), rng.choice([-1.0, 1.0], NSYM // 4)
    pn2, pn4 = rng.choice([-1.0, 1.0], 25 * 2), rng.choice([-1.0, 1.0], 12 * 4)
    noise = rng.normal(0, sig_n, N)
    jam = np.zeros(N) if jtype == "none" else sg.scale_to_power(sg.JAMMER_FUNCS[jtype](rng), 10 ** (jsr_db / 10))
    interf = noise + jam                                   # 같은 잡음, 같은 재머를 모든 파형에 (쌍 비교)
    r_plain = tx_plain(bits, car) + interf
    e_plain = int(((rx_plain(r_plain, car) >= 0) != (bits > 0)).sum())
    e_exc = int(((rx_plain(excise(r_plain, tau), car) >= 0) != (bits > 0)).sum())
    e_d2 = int(((rx_dsss(tx_dsss(b2, pn2, 2, car) + interf, pn2, 2, 25, car) >= 0) != (b2 > 0)).sum())
    e_d4 = int(((rx_dsss(tx_dsss(b4, pn4, 4, car) + interf, pn4, 4, 12, car) >= 0) != (b4 > 0)).sum())
    clean = tx_plain(bits, car) + rng.normal(0, sig_n, N)
    e_clean = int(((rx_plain(clean, car) >= 0) != (bits > 0)).sum())
    hop_bad = jtype != "none" and rng.random() < hop_fail.get(jtype, 0.0)
    e_hop = e_plain if hop_bad else e_clean
    return {"e_plain": e_plain, "e_exc": e_exc, "e_d2": e_d2, "e_d4": e_d4, "e_hop": e_hop,
            "power": float(np.mean(r_plain ** 2)), "img": sg.spec_to_image(r_plain)[0]}


def throughput(f):
    """행동별 정규화 처리량 (슬롯당 정보 비트 / 50)."""
    out = {}
    for c, (k, t) in CODES.items():
        out[f"plain-{c}"] = k * (f["e_plain"] <= t)
        out[f"excise-{c}"] = k * (f["e_exc"] <= t)
        out[f"hop-{c}"] = (1 - HOP_COST) * k * (f["e_hop"] <= t)
    out["dsss2"] = 25 * (f["e_d2"] == 0)
    out["dsss4"] = 12 * (f["e_d4"] == 0)
    return {a: v / 50.0 for a, v in out.items()}


# ---------------- 전략 ----------------
def pwr_bin(jhat_db):
    return int(np.searchsorted(PWR_EDGES, jhat_db))


def observe(strat, f):
    det = f["det"]
    pb = pwr_bin(f["jhat_db"]) if det else -1
    return {"none": 0, "always_hop": f["jtype"] != "none", "detect_fixed": f["jtype"] != "none",
            "energy": det, "energy_pwr": pb, "type": f["jtype"], "type_pwr": (f["jtype"], pb),
            "cnn": f.get("cnn"), "cnn_pwr": (f.get("cnn"), pb),
            "oracle": (f["jtype"], f["jsr"])}[strat]


def design(strat, frames, w_clean):
    """관측값마다 가중 평균 처리량이 가장 큰 행동을 고른 표."""
    if strat == "none":
        return {0: "plain-none"}
    if strat == "always_hop":
        return {True: "hop-none", False: "plain-none"}
    n_clean = sum(f["jtype"] == "none" for f in frames)
    n_jam = len(frames) - n_clean
    acc = {}
    for f in frames:
        w = w_clean / n_clean if f["jtype"] == "none" else (1 - w_clean) / n_jam
        o = observe(strat, f)
        d = acc.setdefault(o, {a: 0.0 for a in ACTIONS})
        for a, v in f["tp"].items():
            d[a] += w * v
    return {o: max(d, key=d.get) for o, d in acc.items()}


def apply(table, strat, f):
    return table.get(observe(strat, f), "plain-none")


def cross(xs, ys, level=0.5):
    """처리량이 level 아래로 처음 떨어지는 J/S(dB, 선형 보간). 끝까지 안 떨어지면 None."""
    for i in range(1, len(xs)):
        if ys[i - 1] >= level > ys[i]:
            return float(xs[i - 1] + (ys[i - 1] - level) / (ys[i - 1] - ys[i]) * (xs[i] - xs[i - 1]))
    return None if ys[-1] >= level else float(xs[0])


# ---------------- 실행 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--jsr-levels", type=float, nargs="+", default=[float(x) for x in np.arange(-10, 20.1, 2.5)])
    ap.add_argument("--frames", type=int, default=600, help="(재머 종류, J/S)마다 슬롯 수. 절반 설계, 절반 평가")
    ap.add_argument("--clean-frames", type=int, default=4000)
    ap.add_argument("--clean-weight", type=float, default=0.5, help="정상 슬롯의 비중")
    ap.add_argument("--energy-fpr", type=float, default=0.01)
    ap.add_argument("--hop-fail", type=float, nargs=4, default=None,
                    metavar=("TONE", "SWEEP", "NOISE", "PARTIAL"))
    ap.add_argument("--cnn-path", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    hop_fail = dict(DEFAULT_HOP_FAIL) if args.hop_fail is None else dict(zip(JAM_TYPES, args.hop_fail))
    tau, pw_clean = calibrate(rng, args.snr)
    p_ref = float(np.mean(pw_clean))
    e_thr = float(np.quantile(pw_clean, 1 - args.energy_fpr))
    print(f"[설정] SNR {args.snr} dB, J/S {args.jsr_levels[0]:g}~{args.jsr_levels[-1]:g} dB, 조건당 {args.frames}슬롯, "
          f"정상 {args.clean_frames}슬롯 | 간섭 제거 기준 {tau:.3f}, 에너지 기준 {e_thr:.3f} (기준 전력 {p_ref:.3f})")
    print(f"[가정] 도약 실패 확률 {hop_fail}, 도약 비용 {HOP_COST:.0%}")

    frames = []
    conds = [("none", None, args.clean_frames)] + [(j, s, args.frames) for j in JAM_TYPES for s in args.jsr_levels]
    for jtype, jsr, n in conds:
        for i in range(n):
            f = one_frame(rng, jtype, jsr if jsr is not None else 0.0, args.snr, tau, hop_fail)
            f.update(jtype=jtype, jsr=jsr, split="design" if i % 2 == 0 else "test", det=f["power"] > e_thr,
                     jhat_db=float(10 * np.log10(max(f["power"] - p_ref, 1e-6))))
            f["tp"] = throughput(f)
            frames.append(f)
        print(f"  {jtype:13s} J/S {'-' if jsr is None else f'{jsr:5.1f}'} 완료 ({time.time() - t0:.0f}s)")

    strats = [s for s in STRATS if not s.startswith("cnn")]
    if args.cnn_path:
        from train_cnn import get_device, load_cnn, predict_proba
        dev = get_device()
        model, classes = load_cnn(args.cnn_path, dev)
        x = np.stack([f["img"] for f in frames])[:, None].astype(np.float32)
        pred = predict_proba(model, x, dev).argmax(1)
        for f, p in zip(frames, pred):
            f["cnn"] = str(classes[p])
        strats = list(STRATS)
        print(f"[CNN] {args.cnn_path} 클래스 {list(classes)}")
    for f in frames:
        del f["img"]

    des = [f for f in frames if f["split"] == "design"]
    test = [f for f in frames if f["split"] == "test"]
    tables = {s: design(s, des, args.clean_weight) for s in strats}

    res = {s: {j: [] for j in JAM_TYPES} for s in strats}
    clean_tp = {}
    for s in strats:
        for j in JAM_TYPES:
            for jsr in args.jsr_levels:
                sel = [f for f in test if f["jtype"] == j and f["jsr"] == jsr]
                res[s][j].append(float(np.mean([f["tp"][apply(tables[s], s, f)] for f in sel])))
        cl = [f for f in test if f["jtype"] == "none"]
        clean_tp[s] = float(np.mean([f["tp"][apply(tables[s], s, f)] for f in cl]))
    avg = {s: np.mean([res[s][j] for j in JAM_TYPES], axis=0).tolist() for s in strats}
    xs = list(args.jsr_levels)
    j50 = {s: cross(xs, avg[s]) for s in strats}
    overall = {s: (1 - args.clean_weight) * float(np.mean(avg[s])) + args.clean_weight * clean_tp[s] for s in strats}

    # 감지 성능
    det_rate = {j: [float(np.mean([f["det"] for f in test if f["jtype"] == j and f["jsr"] == s])) for s in xs]
                for j in JAM_TYPES}
    det_fpr = float(np.mean([f["det"] for f in test if f["jtype"] == "none"]))

    print(f"\n에너지 검출 오탐률(평가) {det_fpr * 100:.2f}%")
    print(f"\n평균 처리량 (재밍 슬롯, 4종 평균) vs J/S")
    print(f"  {'전략':30s} " + " ".join(f"{x:6.1f}" for x in xs) + " | 50% J/S | 정상 슬롯 | 종합")
    for s in strats:
        js = "   >max" if j50[s] is None else f"{j50[s]:7.1f}"
        print(f"  {s:30s} " + " ".join(f"{v:6.3f}" for v in avg[s]) + f" | {js} | {clean_tp[s]:8.3f} | {overall[s]:.3f}")
    print("\n전략별 행동 표 (관측값 -> 행동)")
    for s in strats:
        if s in ("oracle",):
            continue
        items = sorted(tables[s].items(), key=lambda kv: str(kv[0]))
        print(f"  {s:13s} " + ", ".join(f"{k}->{v}" for k, v in items))
    print("\n정답을 아는 경우(oracle)의 최적 행동 (재머 종류 x J/S)")
    for j in JAM_TYPES:
        print(f"  {j:13s} " + " ".join(f"{tables['oracle'].get((j, s), '-'):>10s}" for s in xs))

    # 그림
    show = [s for s in ("none", "always_hop", "energy", "energy_pwr", "type", "type_pwr", "cnn", "cnn_pwr", "oracle")
            if s in strats]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6), sharey=True)
    panels = [("Average of 4 jammer types", avg)] + [(JAM_LABEL[j], {s: res[s][j] for s in strats}) for j in JAM_TYPES]
    for ax, (title, data) in zip(axes, panels):
        for s in show:
            ax.plot(xs, data[s], marker="o", ms=3, color=SCOLOR[s], lw=2.2 if s in ("cnn_pwr", "type_pwr", "always_hop") else 1.3,
                    ls="--" if s == "oracle" else "-", label=SLABEL[s])
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.set_title(title, fontsize=10); ax.set_xlabel("J/S (dB)"); ax.grid(ls=":")
    axes[0].set_ylabel("Normalized throughput (info bits / 50 per slot)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Countermeasure selection, link-level | SNR {args.snr:g} dB, jammed slots only "
                 f"(clean-slot throughput in JSON)", fontsize=11)
    plt.tight_layout()
    tag = f"countermeasure_{args.out_tag}" if args.out_tag else "countermeasure"
    plt.savefig(f"{tag}.png", dpi=200)
    plt.close(fig)
    with open(f"{tag}.json", "w") as fo:
        json.dump({"args": vars(args), "hop_fail": hop_fail, "hop_cost": HOP_COST, "codes": CODES,
                   "excise_tau": tau, "energy_threshold": e_thr, "energy_fpr_test": det_fpr,
                   "energy_det_rate": det_rate, "jsr_levels": xs, "per_type": res, "avg": avg,
                   "clean_throughput": clean_tp, "overall": overall, "jsr_at_50pct": j50,
                   "tables": {s: {str(k): v for k, v in t.items()} for s, t in tables.items()}}, fo, indent=2)
    print(f"\nsaved {tag}.png/.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
