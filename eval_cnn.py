"""CNN 평가: 혼동행렬, JSR별/SNR별 정확도 곡선.

사용:
    python eval_cnn.py                 # cnn_best.pt, dataset.npz, sweep_tests.npz 사용
출력: confusion_matrix.png, accuracy_curves.png, cnn_eval.json
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_cnn import get_device, load_cnn, predict_proba


def confusion(y, pred, n):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return cm


def metrics(model, x, y, device, n):
    """전체 정확도, 클래스별 정확도, 재밍 유무(이진) 탐지 성능. 클래스 0 = none."""
    pred = predict_proba(model, x, device).argmax(1)
    per_class = [float((pred[y == c] == c).mean()) if (y == c).any() else float("nan")
                 for c in range(n)]
    jam_true, jam_pred = y != 0, pred != 0
    tpr = float(jam_pred[jam_true].mean())          # 재밍을 재밍이라고 맞힌 비율
    fpr = float(jam_pred[~jam_true].mean())         # 정상을 재밍이라고 오탐한 비율
    return {
        "acc": float((pred == y).mean()),
        "per_class": per_class,
        "tpr": tpr,
        "fpr": fpr,
        "det_acc": 0.5 * (tpr + 1 - fpr),            # 균형 탐지 정확도
    }, pred


def plot_confusion(cm, classes, path, title):
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def sweep_eval(model, x, y, levels, device, n):
    rows = []
    for k, lv in enumerate(levels):
        m, _ = metrics(model, x[k], y[k], device, n)
        m["level"] = float(lv)
        rows.append(m)
    return rows


def plot_curves(rows_jsr, rows_snr, classes, d, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, rows, xlabel, note in (
        (axes[0], rows_jsr, "JSR (dB)", f"SNR fixed at {float(d['fixed_snr']):.0f} dB"),
        (axes[1], rows_snr, "SNR (dB)", f"JSR fixed at {float(d['fixed_jsr']):.0f} dB"),
    ):
        lv = [r["level"] for r in rows]
        for c, name in enumerate(classes):
            ax.plot(lv, [r["per_class"][c] for r in rows], lw=1, alpha=0.6, marker=".", label=name)
        ax.plot(lv, [r["acc"] for r in rows], "k-o", lw=2.2, label="overall acc")
        ax.plot(lv, [r["det_acc"] for r in rows], "r--s", lw=1.8, label="jam detection (balanced)")
        ax.set_xlabel(xlabel); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.03)
        ax.set_title(note); ax.grid(ls=":")
    axes[0].legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn_best.pt")
    ap.add_argument("--data", default="dataset.npz")
    ap.add_argument("--sweep", default="sweep_tests.npz")
    args = ap.parse_args()

    device = get_device()
    model, classes = load_cnn(args.model, device)
    n = len(classes)

    d = np.load(args.data)
    m, pred = metrics(model, d["x_test"], d["y_test"], device, n)
    cm = confusion(d["y_test"], pred, n)
    plot_confusion(cm, classes, "confusion_matrix.png",
                   f"Test confusion matrix (acc {m['acc'] * 100:.1f}%)")
    print(f"[test] acc {m['acc'] * 100:.2f}% | jam detection TPR {m['tpr'] * 100:.1f}% "
          f"FPR {m['fpr'] * 100:.1f}%")
    for name, a in zip(classes, m["per_class"]):
        print(f"   {name:13s} {a * 100:.1f}%")

    s = np.load(args.sweep)
    rows_jsr = sweep_eval(model, s["x_jsr"], s["y_jsr"], s["jsr_levels"], device, n)
    rows_snr = sweep_eval(model, s["x_snr"], s["y_snr"], s["snr_levels"], device, n)
    plot_curves(rows_jsr, rows_snr, classes, s, "accuracy_curves.png")

    print(f"\nJSR sweep (SNR={float(s['fixed_snr']):.0f} dB):")
    for r in rows_jsr:
        print(f"   JSR {r['level']:+5.0f} dB | acc {r['acc'] * 100:5.1f}% | "
              f"TPR {r['tpr'] * 100:5.1f}% | FPR {r['fpr'] * 100:5.1f}%")
    print(f"SNR sweep (JSR={float(s['fixed_jsr']):.0f} dB):")
    for r in rows_snr:
        print(f"   SNR {r['level']:+5.0f} dB | acc {r['acc'] * 100:5.1f}% | "
              f"TPR {r['tpr'] * 100:5.1f}% | FPR {r['fpr'] * 100:5.1f}%")

    with open("cnn_eval.json", "w") as f:
        json.dump({"classes": classes, "test": m, "confusion": cm.tolist(),
                   "jsr_sweep": rows_jsr, "snr_sweep": rows_snr}, f, indent=2)
    print("\nsaved confusion_matrix.png, accuracy_curves.png, cnn_eval.json")


if __name__ == "__main__":
    main()
