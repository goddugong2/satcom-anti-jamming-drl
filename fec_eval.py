"""오류정정부호(FEC)를 넣었을 때의 프레임 오류율(FER)과 처리량 분석.

입력: ber_eval.py link가 만든 ber_link_compare_*.json (방법별 피격률 + 재머 종류별 피격 슬롯 BER)

모델 (해석적, 부호기와 복호기를 실제로 돌리지는 않음)
  - 한 슬롯에 부호어(프레임) 한 개를 보낸다. 슬롯이 피격되면 그 슬롯의 모든 비트가 같은 BER b로 깨진다
    (재머가 슬롯 전체를 덮으므로 슬롯 안에서는 오류가 독립이라고 근사). 피격이 아니면 BER은 거의 0.
  - 한계거리 복호: n비트 부호어에서 오류가 t개 이하이면 정정, 초과하면 프레임 실패.
      부호 있음 : P_fail(b) = P( Binomial(n, b) > t )
      부호 없음 : P_fail(b) = 1 - (1-b)^k          (k 정보 비트, 오류 하나라도 있으면 실패)
  - FER = h * P_fail(b_hit) + (1-h) * P_fail(b_clean),  h = 방법별 피격률
  - 슬롯당 정보량(처리량, 재전송 없음) = R * (1 - FER),  R = k/n
  - 부호화로 비트 수가 늘어도 부호화 비트당 에너지와 심볼률은 같다고 가정 (대역 확장)

사용:
  python fec_eval.py --link ber_link_compare_multi4_c32_h4_d0_bin_final.json
  python fec_eval.py --link ... --jam-type sweep --target-fer 1e-3
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

# 대표적인 BCH 부호 (n, k, t). 한계거리 복호 가정
CODES = {
    "uncoded k=64": (64, 64, 0),
    "BCH(63,36,t=5)": (63, 36, 5),
    "BCH(127,64,t=10)": (127, 64, 10),
    "BCH(255,131,t=18)": (255, 131, 18),
}
METHODS = ["random", "fhss", "avoid", "afh", "qtable", "qtable2", "dqn_cnn"]
LABEL = {"single_tone": "Single-tone", "sweep": "Swept chirp", "noise": "Broadband noise",
         "partial_band": "Partial-band noise"}


def p_fail(b, n, k, t):
    b = float(np.clip(b, 0.0, 1.0))
    if t == 0:
        return 1.0 - (1.0 - b) ** k
    return float(binom.sf(t, n, b))


def fer(h, b_hit, b_clean, code):
    n, k, t = code
    return h * p_fail(b_hit, n, k, t) + (1.0 - h) * p_fail(b_clean, n, k, t)


def min_t(h, b_hit, n, target):
    """FER <= target을 만족하는 최소 정정 능력 t (정상 슬롯 항은 무시)."""
    for t in range(0, n + 1):
        if h * float(binom.sf(t, n, b_hit)) <= target:
            return t
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", required=True, help="ber_link_compare_*.json")
    ap.add_argument("--jam-type", default="sweep", choices=list(LABEL), help="그래프에 그릴 재머 종류")
    ap.add_argument("--target-fer", type=float, default=1e-3)
    ap.add_argument("--n", type=int, default=127, help="필요 t 계산에 쓰는 부호어 길이")
    args = ap.parse_args()

    with open(args.link) as f:
        d = json.load(f)
    b_clean = d["ber_clean_theory"]
    methods = [m for m in METHODS if m in d["methods"]]
    jam = list(d["hit_slot_ber"])
    b_hit = {j: d["hit_slot_ber"][j]["ber"] for j in jam}
    h = {m: d["methods"][m]["hit_rate"] for m in methods}
    print(f"[입력] {d.get('source', args.link)} | SNR {d['snr']} dB, J/S {d['jsr']} dB")
    print("피격 슬롯 BER: " + ", ".join(f"{LABEL[j]} {b_hit[j]:.3f}" for j in jam))
    print("방법별 피격률: " + ", ".join(f"{m} {h[m] * 100:.2f}%" for m in methods) + "\n")

    out = {"source": d.get("source"), "codes": {k: list(v) for k, v in CODES.items()}, "fer": {}, "throughput": {},
           "min_t": {}}
    for j in jam:
        print(f"===== 재머: {LABEL[j]} (피격 슬롯 BER {b_hit[j]:.3f}) =====")
        print(f"  {'FER (%)':14s}" + "".join(f"{c:>20s}" for c in CODES))
        out["fer"][j], out["throughput"][j] = {}, {}
        for m in methods:
            fers = {c: fer(h[m], b_hit[j], b_clean, code) for c, code in CODES.items()}
            out["fer"][j][m] = fers
            out["throughput"][j][m] = {c: (CODES[c][1] / CODES[c][0]) * (1 - fers[c]) for c in CODES}
            print(f"  {m:14s}" + "".join(f"{fers[c] * 100:20.3f}" for c in CODES))
        print(f"  {'정보량/슬롯':12s}" + "".join(f"{c:>20s}" for c in CODES))
        for m in methods:
            print(f"  {m:14s}" + "".join(f"{out['throughput'][j][m][c]:20.3f}" for c in CODES))
        print()

    print(f"===== FER <= {args.target_fer:g}을 만족하는 최소 정정 능력 t (n={args.n}) =====")
    print(f"  {'':14s}" + "".join(f"{LABEL[j]:>20s}" for j in jam))
    for m in methods:
        row = {j: min_t(h[m], b_hit[j], args.n, args.target_fer) for j in jam}
        out["min_t"][m] = row
        print(f"  {m:14s}" + "".join(f"{row[j]:20d}" for j in jam))

    # ---- 그림: 방법 x 부호 (선택한 재머), 왼쪽 FER, 오른쪽 슬롯당 정보량 ----
    j = args.jam_type
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5))
    w = 0.8 / len(CODES)
    xs = np.arange(len(methods))
    for i, c in enumerate(CODES):
        off = (i - (len(CODES) - 1) / 2) * w
        a1.bar(xs + off, [max(out["fer"][j][m][c], 1e-7) * 100 for m in methods], w, label=c)
        a2.bar(xs + off, [out["throughput"][j][m][c] for m in methods], w, label=c)
    for a in (a1, a2):
        a.set_xticks(xs); a.set_xticklabels(methods, rotation=20); a.grid(axis="y", ls=":")
    a1.set_yscale("log"); a1.set_ylim(1e-2, 100)
    a1.set_ylabel("Frame error rate (%)"); a1.set_title(f"FER | {LABEL[j]} jammer (hit-slot BER {b_hit[j]:.3f})", fontsize=10)
    a1.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    a2.set_ylabel("Information per slot (rate x success)"); a2.set_ylim(0, 1.05)
    a2.set_title("Throughput without retransmission", fontsize=10)
    plt.tight_layout()
    name = args.link.replace("ber_link_", "fec_").replace(".json", "")
    plt.savefig(name + ".png", dpi=250)
    with open(name + ".json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {name}.png/.json")


if __name__ == "__main__":
    main()
