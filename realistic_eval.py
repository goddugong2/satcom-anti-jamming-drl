"""현실 채널(도플러, 링크 버짓 SNR, 페이딩, 위상 잡음)에서 CNN 재밍 탐지 성능을 앙각별로 평가.

사용:
    python realistic_eval.py detect --cnn-path cnn_best.pt
    python realistic_eval.py detect --cnn-path cnn_best.pt --h-km 550 --fc-ghz 2.2 --rs-ksps 100 \
        --snr-zenith 15 --jsr 8 --out-tag leo550

    # 2단계: 채널 증강 데이터로 재학습한 CNN과 비교
    python realistic_eval.py dataset --out dataset_ch.npz
    python train_cnn.py --data dataset_ch.npz --out cnn_ch.pt
    python realistic_eval.py detect --cnn-path cnn_best.pt --compare-cnn cnn_ch.pt --out-tag leo550_aug

    # 3단계: 위성 한 번 통과 동안의 통신 성공률 (채널 -> CNN 탐지 -> DQN 채널 선택)
    python realistic_eval.py pass --cnn-path cnn_best.pt --out-tag leo550_base
    python realistic_eval.py pass --cnn-path cnn_ch.pt --out-tag leo550_aug

비교 조건(같은 앙각, 같은 링크 SNR에서 한 가지씩 켬)
    awgn      : 기존 시뮬레이션과 같은 AWGN + 재머 (링크 SNR만 앙각에 따라 변함)
    doppler   : + 도플러 편이
    fading    : + Rician 페이딩
    full      : 도플러 + 페이딩 + 위상 잡음

출력: realistic_detect_<tag>.png / .json   (가정값은 json의 params 에 모두 기록)
주의: 모든 채널 수치는 가정값이며 특정 실제 시스템 사양이 아니다. channel.py 상단의 한계를 참고.
"""
import argparse
import json
from dataclasses import asdict, replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import channel as ch
import signal_gen as sg
from train_cnn import get_device, load_cnn, predict_proba

CONDITIONS = {
    "awgn": dict(doppler=False, fading=False, pn=False),
    "doppler": dict(doppler=True, fading=False, pn=False),
    "fading": dict(doppler=False, fading=True, pn=False),
    "full": dict(doppler=True, fading=True, pn=True),
}
COLORS = {"awgn": "tab:gray", "doppler": "tab:orange", "fading": "tab:green", "full": "tab:red"}


def params_for(base, cond):
    c = CONDITIONS[cond]
    return replace(base,
                   doppler_residual=base.doppler_residual if c["doppler"] else 0.0,
                   fading=c["fading"],
                   pn_std=base.pn_std if c["pn"] else 0.0)


def make_set(classes, n_per_class, jsr_db, p, el, rng):
    xs, ys = [], []
    for label, name in enumerate(classes):
        for _ in range(n_per_class):
            img, _ = ch.generate_sample_ch(name, jsr_db, p, el, seed=rng)
            xs.append(img)
            ys.append(label)
    return np.stack(xs)[:, None, :, :].astype(np.float32), np.array(ys)


def metrics(model, x, y, device):
    pred = predict_proba(model, x, device).argmax(1)
    jam_t, jam_p = y != 0, pred != 0
    tpr = float(jam_p[jam_t].mean())
    fpr = float(jam_p[~jam_t].mean())
    return {"acc": float((pred == y).mean()), "tpr": tpr, "fpr": fpr,
            "det_acc": 0.5 * (tpr + 1 - fpr)}


def build_bank_ch(cnn_path, p, el_deg, jsr_db, n_per_class=500, seed=0, device=None):
    """run_e2e.py 의 SensingBank 와 같은 형식의 은행을 채널 조건에서 만든다 (2단계: DQN 평가용)."""
    from env import SensingBank
    device = device or get_device()
    model, classes = load_cnn(cnn_path, device)
    rng = np.random.default_rng(seed)
    x, y = make_set(classes, n_per_class, jsr_db, p, el_deg, rng)
    p_jam = 1.0 - predict_proba(model, x, device)[:, 0]
    jam = y != 0
    return SensingBank(p_jam[~jam], p_jam[jam])


def pass_geometry(h_km, el_edges):
    """관측자 바로 위를 지나는 원궤도 통과에서 앙각 구간별 체류 시간 비율과 전체 가시 시간(초).

    지구 중심각 theta는 시간에 비례해 변하고(각속도 v/r), 앙각 E에서 theta(E) = arccos(Re cosE / r) - E.
    따라서 [E_lo, E_hi] 구간에 머무는 시간은 theta(E_lo) - theta(E_hi)에 비례한다(상승, 하강 대칭이므로 2배).
    """
    r = ch.RE_KM + h_km

    def theta(e_deg):
        e = np.radians(e_deg)
        return np.arccos(ch.RE_KM * np.cos(e) / r) - e

    w = np.array([theta(lo) - theta(hi) for lo, hi in zip(el_edges[:-1], el_edges[1:])])
    omega = np.sqrt(ch.MU_KM3S2 / r) / r                   # rad/s
    visible_s = 2 * theta(el_edges[0]) / omega
    return w / w.sum(), float(visible_s)


def _bank(model, classes, device, p, el, jsr, n_per_class, rng):
    from env import SensingBank
    x, y = make_set(classes, n_per_class, jsr, p, el, rng)
    p_jam = 1.0 - predict_proba(model, x, device)[:, 0]
    jam = y != 0
    return SensingBank(p_jam[~jam], p_jam[jam])


def cmd_pass(args):
    """채널(앙각별) -> CNN 탐지 은행 -> 채널 도약 정책의 통신 성공률, 그리고 통과 전체의 시간 가중 평균.

    학습: --train-bank pass 이면 통과 시간 비율대로 섞은 앙각별 은행으로 한 번 학습(실제 운용처럼 한 정책이
    통과 전체를 담당), awgn 이면 기존 실험과 같은 AWGN 은행(SNR --awgn-snr)으로 학습해 배치 불일치를 본다.
    평가: 학습된 같은 정책을 앙각별 은행에서 각각 평가.
    """
    import argparse as _ap
    import time

    import run_e2e as R
    from env import SensingBank

    t0 = time.time()
    device = get_device()
    model, classes = load_cnn(args.cnn_path, device)
    base = ch.ChannelParams(h_km=args.h_km, fc_ghz=args.fc_ghz, rs_ksps=args.rs_ksps,
                            snr_zenith_db=args.snr_zenith, doppler_residual=args.doppler_residual,
                            jammer_doppler=args.jammer_doppler)
    els = list(args.elevations)
    edges = [els[0]] + [(a + b) / 2 for a, b in zip(els[:-1], els[1:])] + [els[-1]]
    w, visible_s = pass_geometry(args.h_km, edges)

    rng = np.random.default_rng(args.seed)
    banks = [_bank(model, classes, device, base, el, args.jsr, args.bank_n, rng) for el in els]
    if args.train_bank == "pass":
        tr_rng = np.random.default_rng(args.seed + 1)
        n_tot = sum(len(b.p_clean) for b in banks)
        pc = np.concatenate([tr_rng.choice(b.p_clean, max(1, int(round(wi * n_tot)))) for b, wi in zip(banks, w)])
        pj = np.concatenate([tr_rng.choice(b.p_jammed, max(1, int(round(wi * n_tot)))) for b, wi in zip(banks, w)])
        train_bank = SensingBank(pc, pj)
    else:
        train_bank = SensingBank.build(args.cnn_path, args.awgn_snr, args.jsr, seed=0)

    rargs = _ap.Namespace(channels=args.channels, jammer=args.jammer, n_jammers=args.n_jammers,
                          history=args.history, episode_len=args.episode_len, switch_cost=0.0,
                          obs_delay=args.obs_delay, obs_quant=args.obs_quant, episodes=args.episodes,
                          table_episodes=args.table_episodes, afh_window=4, eval_episodes=args.eval_episodes,
                          static_dwell=50.0, p_know=0.9)
    ceiling = R.ceiling_of(rargs)

    print(f"[통과] 고도 {args.h_km:.0f} km, 앙각 {els[0]:g}도 이상 가시 시간 약 {visible_s:.0f}초 | "
          f"CNN {args.cnn_path} | 학습 은행 {args.train_bank} (TPR {train_bank.tpr * 100:.1f}%, FPR {train_bank.fpr * 100:.1f}%)")
    print(f"[환경] 재머 {R.jname(rargs)}, 채널 {args.channels}, 관측 {args.obs_quant}, 지연 {args.obs_delay}, "
          f"DQN {args.episodes} / 표 {args.table_episodes or args.episodes} 에피소드, 시드 {args.seeds}, 이론 상한 {ceiling:.2f}%")

    res = {m: np.zeros((args.seeds, len(els))) for m in args.methods}
    for seed in range(args.seeds):
        for m in args.methods:
            R.set_seed(seed)
            cnn_obs = m in R.CNN_METHODS
            env = R.make_env(rargs, "cnn" if cnn_obs else "oracle", train_bank if cnn_obs else None, seed)
            policy = R.build_policy(m, env, rargs, seed)
            if m in R.LEARNING_METHODS:
                for _ in range(R.episodes_for(m, rargs)):
                    R.run_episode(env, policy, greedy=False, learn=True)
            for i, b in enumerate(banks):
                ev = R.make_env(rargs, "cnn" if cnn_obs else "oracle", b if cnn_obs else None,
                                seed + 10_000 + 100 * i, pattern_seed=seed)
                if hasattr(policy, "env"):
                    policy.env = ev
                res[m][seed, i] = np.mean([R.run_episode(ev, policy, greedy=True, learn=False)
                                           for _ in range(args.eval_episodes)])
            print(f"  seed {seed} | {m:11s} 통과 평균 {float(res[m][seed] @ w):.2f}% | 앙각별 "
                  + " ".join(f"{v:.1f}" for v in res[m][seed]))

    summary = {}
    for m in args.methods:
        per_el = res[m].mean(0)
        pass_avg = res[m] @ w
        summary[m] = {"per_el_mean": per_el.tolist(), "per_el_std": res[m].std(0).tolist(),
                      "pass_mean": float(pass_avg.mean()), "pass_std": float(pass_avg.std()),
                      "raw": res[m].tolist()}
    print(f"\n{'앙각':>5s} {'시간비':>6s} {'FPR':>6s} | " + " | ".join(f"{m:>10s}" for m in args.methods))
    for i, el in enumerate(els):
        print(f"{el:5.0f} {w[i] * 100:5.1f}% {banks[i].fpr * 100:5.1f}% | "
              + " | ".join(f"{summary[m]['per_el_mean'][i]:10.2f}" for m in args.methods))
    print(f"{'통과 평균':>19s} | " + " | ".join(f"{summary[m]['pass_mean']:6.2f}±{summary[m]['pass_std']:.2f}"
                                           for m in args.methods) + f"   (이론 상한 {ceiling:.2f})")

    import run_e2e as R2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    for m in args.methods:
        a1.errorbar(els, summary[m]["per_el_mean"], yerr=summary[m]["per_el_std"], color=R2.COLORS.get(m, "k"),
                    marker="o", lw=2 if m.startswith("dqn") else 1.3, capsize=3, label=R2.LABELS.get(m, m))
    a1.axhline(ceiling, color="black", ls=(0, (1, 3)), lw=1.2, label=f"theory ceiling ({ceiling:.1f}%)")
    a1.set_xlabel("Elevation (deg)"); a1.set_ylabel("Communication success rate (%)")
    a1.set_title(f"Satellite pass, LEO {args.h_km:.0f} km | {R2.jname(rargs)}, C={args.channels}, "
                 f"J/S {args.jsr:g} dB, train bank: {args.train_bank}", fontsize=9)
    a1.grid(ls=":"); a1.legend(fontsize=7, loc="lower right")
    tw = a1.twinx()
    tw.bar(els, [b.fpr * 100 for b in banks], width=4, color="gray", alpha=0.25, label="CNN FPR")
    tw.set_ylabel("CNN false-positive rate (%, gray bars)"); tw.set_ylim(0, max(40, 2 * max(b.fpr * 100 for b in banks)))
    ms = list(args.methods)
    a2.barh(ms, [summary[m]["pass_mean"] for m in ms], xerr=[summary[m]["pass_std"] for m in ms],
            color=[R2.COLORS.get(m, "gray") for m in ms], capsize=3)
    for i, m in enumerate(ms):
        a2.text(summary[m]["pass_mean"] + 0.3, i, f"{summary[m]['pass_mean']:.1f}", va="center", fontsize=9)
    lo = min(summary[m]["pass_mean"] for m in ms)
    a2.set_xlim(max(0, lo - 8), 100.5)
    a2.set_xlabel("Pass-averaged success rate (%, time-weighted)")
    a2.set_title(f"Whole pass (elev >= {els[0]:g} deg, ~{visible_s:.0f} s)")
    a2.grid(axis="x", ls=":")
    plt.tight_layout()
    tag = f"realistic_pass_{args.out_tag}" if args.out_tag else "realistic_pass"
    plt.savefig(f"{tag}.png", dpi=200)
    plt.close(fig)
    with open(f"{tag}.json", "w") as f:
        json.dump({"params": asdict(base), "jsr_db": args.jsr, "cnn_path": args.cnn_path,
                   "train_bank": args.train_bank, "env": vars(rargs), "ceiling": ceiling,
                   "elevations": els, "time_weights": w.tolist(), "visible_s": visible_s,
                   "bank": [{"el": el, **ch.describe(base, el), "tpr": b.tpr, "fpr": b.fpr} for el, b in zip(els, banks)],
                   "train_bank_rates": {"tpr": train_bank.tpr, "fpr": train_bank.fpr},
                   "summary": summary}, f, indent=2)
    print(f"saved {tag}.png, {tag}.json  ({time.time() - t0:.0f}s)")


def cmd_dataset(args):
    """채널 증강 학습 데이터: 절반은 기존과 같은 AWGN 샘플, 절반은 현실 채널 샘플.

    현실 채널 샘플은 앙각 U(10, 90), 천정 SNR U(snr_zenith_lo, hi), J/S U(0, 15),
    상승/하강 무작위, 도플러 + 페이딩 + 위상 잡음을 모두 켠다.
    평가(detect)와 다른 시드를 써서 학습/평가 샘플이 겹치지 않게 한다.
    """
    from make_dataset import stratified_split
    rng = np.random.default_rng(args.seed)
    classes = sg.CLASSES_4
    xs, ys = [], []
    for label, name in enumerate(classes):
        for i in range(args.n_per_class):
            jsr = rng.uniform(0, 15)
            if i % 2 == 0:
                img, _ = sg.generate_sample(name, rng.uniform(0, 20), jsr, seed=rng)
            else:
                p = ch.ChannelParams(h_km=args.h_km, fc_ghz=args.fc_ghz, rs_ksps=args.rs_ksps,
                                     snr_zenith_db=rng.uniform(*args.snr_zenith_range),
                                     rising=bool(rng.random() < 0.5))
                img, _ = ch.generate_sample_ch(name, jsr, p, rng.uniform(10, 90), seed=rng)
            xs.append(img)
            ys.append(label)
        print(f"  {name}: {args.n_per_class}장")
    x = np.stack(xs)[:, None, :, :].astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    tr, va, te = stratified_split(y, rng)
    np.savez_compressed(args.out, x_train=x[tr], y_train=y[tr], x_val=x[va], y_val=y[va],
                        x_test=x[te], y_test=y[te], classes=np.array(classes))
    print(f"saved {args.out}: train {len(tr)} / val {len(va)} / test {len(te)}")


def cmd_detect(args):
    device = get_device()
    model, classes = load_cnn(args.cnn_path, device)
    model2 = None
    if args.compare_cnn:
        model2, classes2 = load_cnn(args.compare_cnn, device)
        assert list(classes2) == list(classes), "두 CNN의 클래스가 같아야 합니다"
    base = ch.ChannelParams(h_km=args.h_km, fc_ghz=args.fc_ghz, rs_ksps=args.rs_ksps,
                            snr_zenith_db=args.snr_zenith, doppler_residual=args.doppler_residual,
                            jammer_doppler=args.jammer_doppler)
    rng = np.random.default_rng(args.seed)
    rows = []
    print(f"CNN 클래스 {classes} | 고도 {args.h_km:.0f} km, 반송파 {args.fc_ghz} GHz, "
          f"심볼률 {args.rs_ksps} ksps, 천정 SNR {args.snr_zenith} dB, J/S {args.jsr} dB")
    print(f"{'el':>4} {'SNR':>5} {'fd kHz':>7} | " + " | ".join(f"{c:>8}" for c in CONDITIONS) + "   (균형 탐지 정확도 %)")
    for el in args.elevations:
        info = ch.describe(base, el)
        row = {**info, "cond": {}}
        if model2 is not None:
            row["cond2"] = {}
        for cond in CONDITIONS:
            x, y = make_set(classes, args.n_per_class, args.jsr, params_for(base, cond), el, rng)
            row["cond"][cond] = metrics(model, x, y, device)
            if model2 is not None:                       # 같은 샘플로 두 모델 비교
                row["cond2"][cond] = metrics(model2, x, y, device)
        rows.append(row)
        line = (f"{el:4.0f} {info['snr_db']:5.1f} {info['doppler_real_khz']:7.1f} | " +
                " | ".join(f"{row['cond'][c]['det_acc'] * 100:8.1f}" for c in CONDITIONS))
        if model2 is not None:
            line += (f"   || 증강 CNN: awgn {row['cond2']['awgn']['det_acc'] * 100:5.1f}, "
                     f"full {row['cond2']['full']['det_acc'] * 100:5.1f} "
                     f"(FPR {row['cond2']['full']['fpr'] * 100:4.1f}%)")
        print(line)

    els = [r["el_deg"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    for cond in CONDITIONS:
        ax.plot(els, [r["cond"][cond]["det_acc"] * 100 for r in rows], "-o", color=COLORS[cond],
                lw=2.2 if cond == "full" else 1.4, label=cond)
    if model2 is not None:
        ax.plot(els, [r["cond2"]["full"]["det_acc"] * 100 for r in rows], "--D", color="tab:purple",
                lw=2.2, label="full (channel-aug CNN)")
        ax.plot(els, [r["cond2"]["awgn"]["det_acc"] * 100 for r in rows], ":D", color="tab:purple",
                lw=1.2, label="awgn (channel-aug CNN)")
    ax.set_xlabel("Elevation (deg)"); ax.set_ylabel("Jam detection, balanced acc (%)")
    ax.set_title("Detection vs elevation" if model2 is not None
                 else "Detection vs elevation (CNN trained on AWGN-only data)")
    ax.grid(ls=":"); ax.legend(fontsize=8)
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(els)
    sec.set_xticklabels([f"{r['snr_db']:.0f}" for r in rows], fontsize=7)
    sec.set_xlabel("Link SNR (dB)", fontsize=8)

    ax = axes[1]
    ax.plot(els, [r["cond"]["full"]["tpr"] * 100 for r in rows], "-o", color="tab:red", label="TPR (full)")
    ax.plot(els, [r["cond"]["full"]["fpr"] * 100 for r in rows], "--s", color="tab:blue", label="FPR (full)")
    ax.plot(els, [r["cond"]["awgn"]["tpr"] * 100 for r in rows], ":o", color="gray", label="TPR (awgn)")
    ax.plot(els, [r["cond"]["awgn"]["fpr"] * 100 for r in rows], ":s", color="lightgray", label="FPR (awgn)")
    if model2 is not None:
        ax.plot(els, [r["cond2"]["full"]["tpr"] * 100 for r in rows], "--D", color="tab:purple",
                label="TPR (full, aug CNN)")
        ax.plot(els, [r["cond2"]["full"]["fpr"] * 100 for r in rows], "--v", color="tab:pink",
                label="FPR (full, aug CNN)")
    ax.set_xlabel("Elevation (deg)"); ax.set_ylabel("Rate (%)")
    ax.set_title("TPR / FPR, full channel vs AWGN-only")
    ax.grid(ls=":"); ax.legend(fontsize=8)
    plt.tight_layout()

    tag = f"realistic_detect_{args.out_tag}" if args.out_tag else "realistic_detect"
    plt.savefig(f"{tag}.png", dpi=200)
    plt.close(fig)
    with open(f"{tag}.json", "w") as f:
        json.dump({"params": asdict(base), "jsr_db": args.jsr, "n_per_class": args.n_per_class,
                   "classes": list(classes), "seed": args.seed,
                   "cnn_path": args.cnn_path, "compare_cnn": args.compare_cnn or None,
                   "rows": rows}, f, indent=2)
    print(f"saved {tag}.png, {tag}.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--cnn-path", default="cnn_best.pt")
    d.add_argument("--elevations", type=float, nargs="+", default=[10, 20, 30, 45, 60, 75, 90])
    d.add_argument("--h-km", type=float, default=550.0)
    d.add_argument("--fc-ghz", type=float, default=2.2)
    d.add_argument("--rs-ksps", type=float, default=100.0)
    d.add_argument("--snr-zenith", type=float, default=15.0)
    d.add_argument("--jsr", type=float, default=8.0)
    d.add_argument("--doppler-residual", type=float, default=1.0,
                   help="1=보상 없음, 0.1=90%% 보상")
    d.add_argument("--jammer-doppler", action="store_true")
    d.add_argument("--n-per-class", type=int, default=300)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out-tag", default="")
    d.add_argument("--compare-cnn", default="",
                   help="두 번째 CNN(예: 채널 증강 재학습 cnn_ch.pt)을 같은 샘플로 비교")

    s = sub.add_parser("dataset", help="채널 증강 학습 데이터 생성")
    s.add_argument("--n-per-class", type=int, default=3000)
    s.add_argument("--h-km", type=float, default=550.0)
    s.add_argument("--fc-ghz", type=float, default=2.2)
    s.add_argument("--rs-ksps", type=float, default=100.0)
    s.add_argument("--snr-zenith-range", type=float, nargs=2, default=[10.0, 20.0])
    s.add_argument("--seed", type=int, default=123)
    s.add_argument("--out", default="dataset_ch.npz")
    q = sub.add_parser("pass", help="위성 한 번 통과 동안 채널 -> CNN -> 채널 도약 정책의 통신 성공률")
    q.add_argument("--cnn-path", default="cnn_best.pt")
    q.add_argument("--elevations", type=float, nargs="+", default=[10, 20, 30, 45, 60, 75, 90])
    q.add_argument("--h-km", type=float, default=550.0)
    q.add_argument("--fc-ghz", type=float, default=2.2)
    q.add_argument("--rs-ksps", type=float, default=100.0)
    q.add_argument("--snr-zenith", type=float, default=15.0)
    q.add_argument("--jsr", type=float, default=8.0)
    q.add_argument("--doppler-residual", type=float, default=1.0)
    q.add_argument("--jammer-doppler", action="store_true")
    q.add_argument("--bank-n", type=int, default=500, help="앙각별 은행의 클래스당 샘플 수")
    q.add_argument("--train-bank", choices=("pass", "awgn"), default="pass")
    q.add_argument("--awgn-snr", type=float, default=10.0, help="--train-bank awgn 일 때의 SNR")
    q.add_argument("--channels", type=int, default=32)
    q.add_argument("--jammer", default="multi")
    q.add_argument("--n-jammers", type=int, default=4)
    q.add_argument("--history", type=int, default=4)
    q.add_argument("--obs-delay", type=int, default=0)
    q.add_argument("--obs-quant", choices=("soft", "binary"), default="binary")
    q.add_argument("--episodes", type=int, default=800)
    q.add_argument("--table-episodes", type=int, default=10000)
    q.add_argument("--episode-len", type=int, default=100)
    q.add_argument("--eval-episodes", type=int, default=100)
    q.add_argument("--seeds", type=int, default=3)
    q.add_argument("--methods", nargs="+", default=["random", "afh", "qtable", "dqn_oracle", "dqn_cnn"])
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--out-tag", default="")
    args = ap.parse_args()
    {"detect": cmd_detect, "dataset": cmd_dataset, "pass": cmd_pass}[args.cmd](args)


if __name__ == "__main__":
    main()
