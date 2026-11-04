"""Exploratory analysis of the public measured jamming IQ dataset (Zenodo 7119040, Alhazbi et al., 2023).

Neither the README nor the paper gives the absolute sample rate, so everything is computed
in normalized frequency (cycles/sample). Mapping onto the simulation only needs the
ratios (samples per symbol, jammer frequency / bandwidth).

What it measures (per variable: Nojamming / Sine / Gaussian):
  1) power over time (1 s-ish blocks across the full recording) -> on/off segments, level differences
  2) Welch PSD (two-sided, normalized frequency) -> occupied bandwidth, tone position
  3) samples per symbol: period of the repeating bit sequence (|autocorrelation| peak)
     + |x|^2 spectral line (symbol-rate line)
  4) estimated JSR: in-band PSD excess relative to Nojamming

Usage:
    python real_explore.py D:\\data\\w27.mat
    python real_explore.py D:\\data\\w27.mat --seg 4194304 --block 1048576
Output: results/real_explore_<name>.json, results/real_explore_<name>.png
"""
import argparse
import json
import os

import numpy as np
from scipy import signal

VARS = ["Nojamming", "Sine", "Gaussian"]


# ---------------- loading ----------------
class MatIQ:
    """Lazily reads (2, N) or (N, 2) I/Q variables from a .mat (v7.3 = HDF5)."""

    def __init__(self, path):
        import h5py
        self.f = h5py.File(path, "r")

    def names(self):
        return list(self.f.keys())

    def length(self, name):
        s = self.f[name].shape
        return max(s)

    def read(self, name, start, stop):
        ds = self.f[name]
        if ds.shape[0] == 2:
            a = ds[:, start:stop]
            return (a[0] + 1j * a[1]).astype(np.complex64)
        a = ds[start:stop, :]
        return (a[:, 0] + 1j * a[:, 1]).astype(np.complex64)


# ---------------- analysis ----------------
def block_power(src, name, block):
    n = src.length(name)
    out = []
    for s in range(0, n - block + 1, block):
        x = src.read(name, s, s + block)
        out.append(float(np.mean(np.abs(x) ** 2)))
    return np.array(out)


def psd(x, nfft=4096):
    f, p = signal.welch(x, fs=1.0, nperseg=nfft, return_onesided=False, scaling="density")
    i = np.argsort(f)
    return f[i], p[i]


def occupied_bw(f, p, frac=0.99):
    """Bandwidth (normalized) holding frac of the power, centered on the power centroid."""
    c = np.cumsum(p) / np.sum(p)
    lo = f[np.searchsorted(c, (1 - frac) / 2)]
    hi = f[np.searchsorted(c, 1 - (1 - frac) / 2)]
    return float(hi - lo), float(lo), float(hi)


def autocorr_period(x, lo=64, hi=None):
    """Lag of the |autocorrelation| peak over [lo, hi] samples (period of the repeating bit sequence)."""
    n = len(x)
    hi = hi or n // 3
    X = np.fft.fft(x - x.mean(), 2 * n)
    r = np.abs(np.fft.ifft(np.abs(X) ** 2))[: n]
    r /= r[0]
    seg = r[lo:hi]
    k = int(np.argmax(seg)) + lo
    # top 5 peaks (for checking harmonics/multiples)
    pk, _ = signal.find_peaks(seg, distance=max(lo, 16))
    top = pk[np.argsort(seg[pk])[::-1][:5]] + lo
    return k, float(r[k]), [(int(t), float(r[t])) for t in top]


def bw_db(f, p, drop_db=10.0):
    """Width of the region where the PSD is within drop_db of its peak (main-lobe width, normalized)."""
    ps = signal.medfilt(p, 9)
    m = ps >= ps.max() * 10 ** (-drop_db / 10)
    idx = np.where(m)[0]
    return float(f[idx[-1]] - f[idx[0]]), float(f[idx[0]]), float(f[idx[-1]])


def symbol_line(x, fmin=0.01):
    """Strongest non-DC line in the |x|^2 spectrum -> symbol rate (cycles/sample)."""
    e = np.abs(x) ** 2
    E = np.abs(np.fft.rfft((e - e.mean()) * np.hanning(len(e))))
    f = np.fft.rfftfreq(len(e))
    m = f > fmin
    pk, _ = signal.find_peaks(E[m])
    top = pk[np.argsort(E[m][pk])[::-1][:5]]
    med = np.median(E[m])
    return [(float(f[m][t]), float(E[m][t] / med)) for t in top]


def cfo_line(x):
    """For BPSK, x^2 has a line at 2*CFO -> carrier frequency offset (cycles/sample)."""
    y = x ** 2
    Y = np.abs(np.fft.fftshift(np.fft.fft(y * np.hanning(len(y)))))
    f = np.fft.fftshift(np.fft.fftfreq(len(y)))
    k = int(np.argmax(Y))
    return float(f[k] / 2), float(Y[k] / np.median(Y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--seg", type=int, default=2 ** 22, help="analysis segment length (taken from the middle)")
    ap.add_argument("--block", type=int, default=2 ** 20, help="block size for the power-over-time curve")
    ap.add_argument("--no-blocks", action="store_true", help="skip the full power scan (faster)")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = os.path.splitext(os.path.basename(a.path))[0]
    src = MatIQ(a.path)
    names = [v for v in VARS if v in src.names()] or src.names()
    print(f"[{tag}] variables: {names}")

    res = {"file": tag, "source": "Zenodo 7119040 (Alhazbi, Sciancalepore, Oligeri, Data in Brief 2023), CC BY 4.0",
           "note": "all frequencies are normalized (cycles/sample)", "vars": {}}
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    psds = {}
    for name in names:
        n = src.length(name)
        s0 = max(0, n // 2 - a.seg // 2)
        x = src.read(name, s0, s0 + a.seg)
        f, p = psd(x)
        psds[name] = (f, p)
        bw, lo, hi = occupied_bw(f, p)
        bw10, lo10, hi10 = bw_db(f, p)
        lag, rv, tops = autocorr_period(x[: 2 ** 21])
        sl = symbol_line(x)
        cfo, cfo_snr = cfo_line(x[: 2 ** 20])
        r = {"n_samples": int(n), "seg_power": float(np.mean(np.abs(x) ** 2)),
             "dc": [float(np.mean(x).real), float(np.mean(x).imag)],
             "occ_bw_99": bw, "occ_band": [lo, hi], "bw_10db": bw10, "band_10db": [lo10, hi10],
             "psd_peak_freq": float(f[np.argmax(p)]),
             "autocorr_peak_lag": lag, "autocorr_peak_val": rv, "autocorr_top": tops,
             "sps_if_2048bit_period": lag / 2048.0,
             "env_lines": sl, "cfo_est": cfo, "cfo_line_snr": cfo_snr}
        if not a.no_blocks:
            bp = block_power(src, name, a.block)
            r["block_power"] = {"mean": float(bp.mean()), "std": float(bp.std()),
                                "min": float(bp.min()), "max": float(bp.max()), "n_blocks": len(bp)}
            ax[1, 0].plot(10 * np.log10(bp), label=name)
        res["vars"][name] = r
        ax[0, 0].plot(f, 10 * np.log10(p + 1e-20), lw=0.8, label=name)
        print(f"  {name}: N={n}, power={r['seg_power']:.4f}, -10dB BW={bw10:.4f}, PSD peak f={r['psd_peak_freq']:+.4f}, "
              f"autocorr lag={lag} ({rv:.2f}), env line={sl[0][0]:.5f}, CFO~{cfo:+.5f}")

    # estimated in-band JSR (PSD excess relative to Nojamming)
    if "Nojamming" in psds:
        f0, p0 = psds["Nojamming"]
        _, lo, hi = bw_db(f0, p0)
        p_sig = res["vars"]["Nojamming"]["seg_power"]
        band = (f0 >= lo) & (f0 <= hi)
        for name in names:
            if name == "Nojamming":
                continue
            _, pj = psds[name]
            excess = np.sum(np.clip(pj - p0, 0, None)[band])
            sig = np.sum(p0[band])
            res["vars"][name]["jsr_inband_db_est"] = float(10 * np.log10(excess / sig + 1e-12))
            dp = res["vars"][name]["seg_power"] - p_sig
            res["vars"][name]["jsr_total_db_est"] = float(10 * np.log10(dp / p_sig)) if dp > 0 else None
            ax[0, 1].plot(f0, 10 * np.log10(pj / p0), lw=0.8, label=f"{name} / Nojamming")
        ax[0, 1].axvspan(lo, hi, color="gray", alpha=0.15, label="signal -10 dB band")

    # spectrogram of a short slice of each variable
    xs = [src.read(nm, src.length(nm) // 2, src.length(nm) // 2 + 2 ** 16) for nm in names]
    x = np.concatenate(xs)
    ff, tt, S = signal.spectrogram(x, fs=1.0, nperseg=256, noverlap=192, return_onesided=False)
    S = np.fft.fftshift(S, axes=0)
    ff = np.fft.fftshift(ff)
    ax[1, 1].pcolormesh(tt, ff, 10 * np.log10(S + 1e-20), shading="auto", cmap="jet")
    ax[1, 1].set_title(" | ".join(names) + " (each 65536 samples)")
    ax[1, 1].set_xlabel("sample"); ax[1, 1].set_ylabel("norm. freq")

    ax[0, 0].set_title(f"{tag}: Welch PSD"); ax[0, 0].set_xlabel("norm. freq"); ax[0, 0].set_ylabel("dB")
    ax[0, 1].set_title("PSD ratio vs Nojamming"); ax[0, 1].set_xlabel("norm. freq"); ax[0, 1].set_ylabel("dB")
    ax[1, 0].set_title(f"power per block ({a.block} samples)"); ax[1, 0].set_xlabel("block"); ax[1, 0].set_ylabel("dB")
    for k in (ax[0, 0], ax[0, 1], ax[1, 0]):
        k.grid(ls=":"); k.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(a.out_dir, exist_ok=True)
    png = os.path.join(a.out_dir, f"real_explore_{tag}.png")
    js = os.path.join(a.out_dir, f"real_explore_{tag}.json")
    plt.savefig(png, dpi=150)
    with open(js, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"saved {png}, {js}")


if __name__ == "__main__":
    main()
