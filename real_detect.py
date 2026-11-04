"""Jamming classifier trained directly on the public measured IQ dataset (Zenodo 7119040).

Classes: none / tone / noise (3 classes). No known-bit information is used at test time (blind detection).
Evaluation is leave-one-file-out (LOFO): train on all other recordings, test on one recording.
Because windows from the same recording are strongly correlated, splitting by file is the only
leakage-free way to measure generalization to a new session (w19), new jammer hardware (w21~23)
or new distance (w24~31).

Methods
  energy : mean received power threshold (binary none vs jammed only)
  feat   : 5 blind features + multinomial logistic regression
           (log power, M2M4 SNR estimate, spectral flatness, peak-to-2nd-peak ratio, DC-region peak ratio)
  cnn    : the same JammingCNN architecture as the simulation (train_cnn.py), 3 classes,
           input = 32x32 log spectrogram of the complex window (two-sided, fftshift), per-sample standardized

Usage
  1) Extract windows from each .mat once (saves a small cache; the .mat can be deleted afterwards):
       python real_detect.py extract C:\\data\\w1.mat C:\\data\\w12.mat C:\\data\\w15.mat C:\\data\\w19.mat C:\\data\\w21.mat C:\\data\\w27.mat
     Long-frame version (recommended; see image() docstring):
       python real_detect.py extract <same files> --win 32768 --nperseg 2048 --segments 40 --seg-len 262144 --cache-dir real_cache_w32k
  2) Train + evaluate (LOFO over all cached files):
       python real_detect.py train
       python real_detect.py train --epochs 15 --seeds 3 --out-tag lofo
Output: real_cache/real_win_<file>.npz (do not commit), results/real_detect_<tag>.json/png
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import signal
from scipy.ndimage import map_coordinates

VARS = ["Nojamming", "Sine", "Gaussian"]
CLASSES = ["none", "tone", "noise"]
# segments known to be invalid (README 6.16: w19 Nojamming link failure after ~block 111 of 2^20 samples)
EXCLUDE = {("w19", "Nojamming"): 111 * 2 ** 20}
FEAT_NAMES = ["log_power", "m2m4_snr_db", "spectral_flatness", "peak_to_2nd_db", "dc_peak_db"]


# ---------------- features / images ----------------
def features(x):
    p = np.mean(np.abs(x) ** 2)
    m2, m4 = p, np.mean(np.abs(x) ** 4)
    s = np.sqrt(max(2 * m2 ** 2 - m4, 1e-12))
    snr = 10 * np.log10(s / max(m2 - s, 1e-12))
    P = np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(len(x))))) ** 2 + 1e-12
    flat = np.exp(np.mean(np.log(P))) / np.mean(P)
    k = int(np.argmax(P))
    P2 = P.copy()
    P2[max(0, k - 4):k + 5] = 0
    pk2 = 10 * np.log10(P[k] / P2.max())
    c = len(P) // 2
    w = len(P) // 64
    dc = 10 * np.log10(P[c - w:c + w].max() / np.median(P))
    return np.array([10 * np.log10(p), snr, flat, pk2, dc], np.float32)


def image(x, out=32, nperseg=64):
    """32x32 log spectrogram of a complex window (two-sided, fftshift, per-sample standardized).

    Frequency bins are pooled to `out` groups by averaging linear power (not by interpolation, which would
    drop bins). nperseg matters on this dataset: the counting-byte payload gives short frames a strong,
    data-dependent DC component (frame-to-frame +-10 dB with nperseg=64), which looks like a DC tone.
    Frames of 2048 symbols (one payload period) remove it."""
    nov = nperseg // 2 if nperseg >= 256 else 0
    _, _, S = signal.spectrogram(x, fs=1.0, nperseg=nperseg, noverlap=nov, return_onesided=False, mode="psd")
    S = np.fft.fftshift(S, axes=0)
    f, t = S.shape
    if f > out and f % out == 0:
        S = S.reshape(out, f // out, t).mean(1)
    S = 10 * np.log10(S + 1e-12)
    f, t = S.shape
    rr, cc = np.meshgrid(np.linspace(0, f - 1, out), np.linspace(0, t - 1, out), indexing="ij")
    img = map_coordinates(S, [rr, cc], order=1)
    return ((img - img.mean()) / (img.std() + 1e-8)).astype(np.float16)


# ---------------- extract ----------------
def cmd_extract(a):
    from real_explore import MatIQ
    os.makedirs(a.cache_dir, exist_ok=True)
    for path in a.paths:
        tag = os.path.splitext(os.path.basename(path))[0]
        src = MatIQ(path)
        X, F, Y, POS = [], [], [], []
        for ci, name in enumerate(VARS):
            n = src.length(name)
            n_use = min(n, EXCLUDE.get((tag, name), n))
            starts = np.linspace(0, n_use - a.seg_len, a.segments).astype(int)
            for s0 in starts:
                seg = src.read(name, int(s0), int(s0) + a.seg_len)
                for j in range(a.seg_len // a.win):
                    w = seg[j * a.win:(j + 1) * a.win]
                    X.append(image(w, nperseg=a.nperseg))
                    F.append(features(w))
                    Y.append(ci)
                    POS.append(int(s0) + j * a.win)
            print(f"[{tag}] {name}: {a.segments} segments x {a.seg_len // a.win} windows (used samples < {n_use})")
        out = os.path.join(a.cache_dir, f"real_win_{tag}.npz")
        np.savez_compressed(out, x=np.stack(X)[:, None], f=np.stack(F), y=np.array(Y), pos=np.array(POS),
                            win=a.win, nperseg=a.nperseg)
        print(f"saved {out} ({len(Y)} windows)")


# ---------------- models ----------------
def fit_logreg(F, y, n_cls=3, iters=3000, lr=0.5, l2=1e-3):
    mu, sd = F.mean(0), F.std(0) + 1e-8
    Z = np.c_[(F - mu) / sd, np.ones(len(F))]
    W = np.zeros((Z.shape[1], n_cls))
    Yo = np.eye(n_cls)[y]
    for _ in range(iters):
        L = Z @ W
        L -= L.max(1, keepdims=True)
        P = np.exp(L)
        P /= P.sum(1, keepdims=True)
        G = Z.T @ (P - Yo) / len(y) + l2 * W
        W -= lr * G
    return mu, sd, W


def pred_logreg(m, F):
    mu, sd, W = m
    Z = np.c_[(F - mu) / sd, np.ones(len(F))]
    return np.argmax(Z @ W, 1)


def fit_energy(F, y):
    """Best power threshold (and direction) for none vs jammed, balanced accuracy on train."""
    p = F[:, 0]
    jam = y != 0
    best = (-1, 0, 1)
    for th in np.quantile(p, np.linspace(0.01, 0.99, 197)):
        for sgn in (1, -1):
            pr = sgn * (p - th) > 0
            ba = 0.5 * (pr[jam].mean() + (~pr[~jam]).mean())
            if ba > best[0]:
                best = (ba, th, sgn)
    return best[1], best[2]


def pred_energy(m, F):
    th, sgn = m
    return (sgn * (F[:, 0] - th) > 0).astype(int)  # 1 = jammed


def train_cnn_model(X, y, epochs, seed, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from train_cnn import JammingCNN, run_epoch, set_seed
    set_seed(seed)
    model = JammingCNN(num_classes=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    dl = DataLoader(ds, batch_size=128, shuffle=True)
    for ep in range(epochs):
        loss, acc = run_epoch(model, dl, device, opt)
    return model


def predict_cnn(model, X, device):
    from train_cnn import predict_proba
    return predict_proba(model, X.astype(np.float32), device).argmax(1)


# ---------------- metrics ----------------
def summarize(y, pred, binary_only=False):
    jam_t = y != 0
    jam_p = pred != 0
    tpr = float(jam_p[jam_t].mean())
    fpr = float(jam_p[~jam_t].mean())
    r = {"det_tpr": tpr, "det_fpr": fpr, "det_bal_acc": 0.5 * (tpr + 1 - fpr)}
    if not binary_only:
        cm = np.zeros((3, 3), int)
        np.add.at(cm, (y, pred), 1)
        r["acc3"] = float((y == pred).mean())
        r["per_class"] = [float(cm[c, c] / max(cm[c].sum(), 1)) for c in range(3)]
        r["confusion"] = cm.tolist()
    return r


def cmd_train(a):
    files = sorted(glob.glob(os.path.join(a.cache_dir, "real_win_*.npz")))
    if len(files) < 2:
        raise SystemExit("need at least 2 cached files (run extract first)")
    data = {}
    for f in files:
        d = np.load(f)
        data[os.path.basename(f)[9:-4]] = (d["x"], d["f"], d["y"])
    tags = sorted(data, key=lambda t: int(t[1:]) if t[1:].isdigit() else 0)
    test_tags = a.test or tags
    from real_ber import META
    device = None
    if "cnn" in a.methods:
        from train_cnn import get_device
        device = get_device()
    res = {"source": "Zenodo 7119040 (Alhazbi, Sciancalepore, Oligeri, Data in Brief 2023), CC BY 4.0",
           "protocol": "leave-one-file-out", "classes": CLASSES, "files": tags,
           "window": int(np.load(files[0])["win"]),
           "nperseg": int(np.load(files[0])["nperseg"]) if "nperseg" in np.load(files[0]) else 64, "seeds": a.seeds, "epochs": a.epochs, "results": {}}
    for tt in test_tags:
        tr = [t for t in tags if t != tt]
        Xtr = np.concatenate([data[t][0] for t in tr]); Ftr = np.concatenate([data[t][1] for t in tr])
        ytr = np.concatenate([data[t][2] for t in tr])
        Xte, Fte, yte = data[tt]
        r = {"meta": META.get(tt), "n_test": int(len(yte)), "train_files": tr}
        if "energy" in a.methods:
            r["energy"] = summarize(yte, pred_energy(fit_energy(Ftr, ytr), Fte), binary_only=True)
        if "feat" in a.methods:
            r["feat"] = summarize(yte, pred_logreg(fit_logreg(Ftr, ytr), Fte))
        if "cnn" in a.methods:
            runs = []
            for s in range(a.seeds):
                m = train_cnn_model(Xtr, ytr, a.epochs, s, device)
                runs.append(summarize(yte, predict_cnn(m, Xte, device)))
            r["cnn"] = {"acc3_mean": float(np.mean([u["acc3"] for u in runs])),
                        "acc3_std": float(np.std([u["acc3"] for u in runs])),
                        "det_bal_acc_mean": float(np.mean([u["det_bal_acc"] for u in runs])),
                        "per_class_mean": np.mean([u["per_class"] for u in runs], 0).tolist(),
                        "confusion_seed0": runs[0]["confusion"], "runs": runs}
        res["results"][tt] = r
        line = f"[test {tt} {r['meta']}]"
        if "energy" in r: line += f" energy det={r['energy']['det_bal_acc']:.3f}"
        if "feat" in r: line += f" | feat acc3={r['feat']['acc3']:.3f} det={r['feat']['det_bal_acc']:.3f}"
        if "cnn" in r: line += (f" | cnn acc3={r['cnn']['acc3_mean']:.3f}+-{r['cnn']['acc3_std']:.3f} "
                                f"det={r['cnn']['det_bal_acc_mean']:.3f} per-class={np.round(r['cnn']['per_class_mean'], 3)}")
        print(line, flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    js = os.path.join(a.out_dir, f"real_detect_{a.out_tag}.json")
    with open(js, "w") as fh:
        json.dump(res, fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(test_tags)), 4.2))
    xs = np.arange(len(test_tags))
    wbar = 0.27
    series = []
    if "energy" in a.methods:
        series.append(("energy (binary det.)", [res["results"][t]["energy"]["det_bal_acc"] for t in test_tags]))
    if "feat" in a.methods:
        series.append(("features + logreg (3-class)", [res["results"][t]["feat"]["acc3"] for t in test_tags]))
    if "cnn" in a.methods:
        series.append(("CNN (3-class)", [res["results"][t]["cnn"]["acc3_mean"] for t in test_tags]))
    for i, (lab, v) in enumerate(series):
        ax.bar(xs + (i - (len(series) - 1) / 2) * wbar, v, wbar, label=lab)
    lab = []
    for t in test_tags:
        m = META.get(t)
        lab.append(f"{t}\nJ{m[0]} P{m[1]} {m[2]}m" if m else t)
    ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8)
    ax.axhline(1 / 3, color="gray", ls=":", lw=1); ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("accuracy on held-out file")
    ax.set_title(f"Real IQ jamming detection, leave-one-file-out (window {res['window']}, CNN frame {res['nperseg']})")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", ls=":")
    plt.tight_layout()
    png = os.path.join(a.out_dir, f"real_detect_{a.out_tag}.png")
    plt.savefig(png, dpi=150)
    print(f"saved {js}, {png}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("paths", nargs="+")
    e.add_argument("--segments", type=int, default=20, help="segments per variable, spread over the recording")
    e.add_argument("--seg-len", type=int, default=2 ** 17)
    e.add_argument("--win", type=int, default=4096, help="symbols per classification window")
    e.add_argument("--nperseg", type=int, default=64, help="spectrogram frame length for the CNN image")
    e.add_argument("--cache-dir", default="real_cache")
    t = sub.add_parser("train")
    t.add_argument("--cache-dir", default="real_cache")
    t.add_argument("--test", nargs="*", help="files to hold out (default: every file, LOFO)")
    t.add_argument("--methods", nargs="+", default=["energy", "feat", "cnn"])
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--seeds", type=int, default=2)
    t.add_argument("--out-dir", default="results")
    t.add_argument("--out-tag", default="lofo")
    a = ap.parse_args()
    cmd_extract(a) if a.cmd == "extract" else cmd_train(a)


if __name__ == "__main__":
    main()
