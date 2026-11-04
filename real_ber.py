"""Measured BER / frame success from the real jamming IQ dataset (Zenodo 7119040).

What real_explore.py found:
  - Nojamming autocorrelation peaks at lag 2048 (0.999) and the PSD has nulls every 1/8 with a peak at -0.4375
    -> exactly matches the spectrum of a 1-sample-per-symbol BPSK stream of the counting bytes 0..255 (2048 bits).
  => The stored IQ is effectively at the symbol rate (1 sample = 1 symbol), and the bit sequence is known.

So this script directly measures, on real received symbols:
  1) BER (symbol timing/phase aligned every 2048 symbols with the known sequence = genie pilot sync)
  2) frame success over 50-bit frames for plain / BCH(50,38,t2) / (50,23,t5) / (50,11,t7)
     (same code family as countermeasure_eval.py; successful if errors <= t)
  3) i.i.d. prediction from the measured BER vs measured frame success -> shows how bursty real errors are
  4) tone cancellation (blind: strongest periodogram peak, LS fit, subtract) as excision on real data
  5) data-aided SINR estimate

Usage:
    python real_ber.py C:\\data\\w27.mat
    python real_ber.py C:\\data\\w1.mat C:\\data\\w15.mat C:\\data\\w27.mat --chunks 8
Output: results/real_ber_<name>.json (one per file), results/real_ber_summary.json/png (when several files)
"""
import argparse
import json
import os
from math import comb

import numpy as np

from real_explore import MatIQ, VARS

PERIOD = 2048
# README.xls of the dataset: file -> (jammer radio id, relative jamming power, distance m)
META = {f"w{i}": v for i, v in enumerate([
    (4, .1, 10), (4, .3, 10), (4, .6, 10), (5, .1, 10), (5, .3, 10), (5, .6, 10), (4, .2, 10), (5, .2, 10),
    (4, .4, 10), (5, .4, 10), (5, .5, 10), (4, .5, 10), (4, .7, 10), (5, .7, 10), (4, .8, 10), (5, .8, 10),
    (5, .6, 10), (4, .6, 10), (4, .5, 10), (5, .5, 10), (6, .5, 10), (7, .5, 10), (8, .5, 10), (4, .4, 3),
    (4, .5, 5), (4, .5, 7), (4, .5, 10), (4, .5, 13), (4, .5, 16), (4, .5, 19), (4, .5, 21)], start=1)}
CODES = [("plain", 50, 0), ("bch_t2", 38, 2), ("bch_t5", 23, 5), ("bch_t7", 11, 7)]  # (name, k, t), n=50
FRAME_N = 50


def ref_sequence(bitorder="big"):
    b = np.unpackbits(np.arange(256, dtype=np.uint8), bitorder=bitorder).astype(np.float64)
    return 2 * b - 1  # bit 1 -> +1, bit 0 -> -1 (overall sign is absorbed into the complex gain)


def align_block(y, R_conj):
    """Circular correlation of one 2048 block with the reference -> (shift, complex gain, peak ratio)."""
    c = np.fft.ifft(np.fft.fft(y) * R_conj)
    a = np.abs(c)
    k = int(np.argmax(a))
    second = np.partition(a, -2)[-2]
    return k, c[k] / PERIOD, float(a[k] / (second + 1e-12))


def cancel_tone(x, block=65536, pad=4, band=None, min_ratio=4.0):
    """Blind single-tone cancellation: per block, strongest periodogram peak -> LS amplitude -> subtract.
    Cancels only if that peak is min_ratio (power, default 6 dB) above every other peak
    (the counting-sequence signal has many equal-height lines, a tone is a single outstanding one)."""
    out = x.copy()
    info = []
    n = np.arange(block)
    for s in range(0, len(x) - block + 1, block):
        b = x[s:s + block]
        X = np.abs(np.fft.fft(b * np.hanning(block), pad * block))
        f = np.fft.fftfreq(pad * block)
        if band is not None:
            X = np.where(np.abs(f) <= band, X, 0)
        k = int(np.argmax(X))
        X2 = X.copy()
        X2[max(0, k - 16):k + 17] = 0
        if X[k] ** 2 < min_ratio * np.max(X2) ** 2:     # no unique peak -> no tone, skip
            info.append((float("nan"), 0.0))
            continue
        # parabolic interpolation
        a, bb, c = X[k - 1], X[k], X[(k + 1) % len(X)]
        d = 0.5 * (a - c) / (a - 2 * bb + c + 1e-12)
        f0 = (k + d) / (pad * block)
        f0 = f0 - 1 if f0 > 0.5 else f0
        e = np.exp(2j * np.pi * f0 * n)
        amp = np.vdot(e, b) / block
        out[s:s + block] = b - amp * e
        info.append((float(f0), float(abs(amp) ** 2)))
    return out, info


def demod(x, ref, mode="track", sub=256, search=3, reacq_ratio=1.6):
    """Align with the known sequence and decide bits (one recorded chunk).

    mode="track" (default): timing tracker. Under jamming the stored symbol stream slips by +-1 symbol
      every few thousand symbols (the shift trace drifts 323 -> 329 -> 307 ...), while it is constant
      without jamming. So the shift is re-estimated every `sub` symbols within +-`search` of the previous one
      (the counting sequence has zero autocorrelation at lags 1..3, so neighbours are unambiguous),
      and gain/phase are estimated on the same sub-block (genie pilot-aided sync). Nothing is excluded:
      errors caused by slips are counted, since they are a real jamming effect on the receiver.
      Re-acquisition: at each 2048 block, if the full-search correlation peak is reacq_ratio times stronger
      than the correlation at the tracked shift, jump there (handles large timing jumps / sample drops that
      a +-3 tracker cannot follow; side peaks of the sequence are <= 0.75 of the main peak).
    mode="per_block": independent argmax on every 2048 block (no tracking), for comparison.
    """
    nb = len(x) // PERIOD
    R_conj = np.conj(np.fft.fft(ref))
    Y = x[: nb * PERIOD].reshape(nb, PERIOD)
    C = np.fft.ifft(np.fft.fft(Y, axis=1) * R_conj[None, :], axis=1)
    k_raw = np.argmax(np.abs(C), axis=1)
    errs, dist = [], []
    if mode == "per_block":
        for i in range(nb):
            rr = np.roll(ref, k_raw[i])
            z = Y[i] / (C[i, k_raw[i]] / PERIOD)
            errs.append(np.sign(z.real) != rr)
            dist.append(np.mean(np.abs(z - rr) ** 2))
        e = np.concatenate(errs)
        return e, dist, {"blocks_total": int(nb), "slips": 0, "reacquisitions": 0,
                         "trace": [int(v) for v in k_raw[:512]]}
    k = int(np.bincount(k_raw[:8]).argmax())        # acquisition: mode over the first 8 blocks
    n_all = np.arange(len(x))
    slips, reacq, trace = 0, 0, []
    ds = np.arange(-search, search + 1)
    for st in range(0, nb * PERIOD - sub + 1, sub):
        if st % PERIOD == 0:
            i = st // PERIOD
            a = np.abs(C[i])
            if k_raw[i] != k and a[k_raw[i]] > reacq_ratio * a[k]:
                k = int(k_raw[i])
                reacq += 1
        y = x[st:st + sub]
        n = n_all[st:st + sub]
        cands = [(np.dot(y, ref[(n - (k + d)) % PERIOD])) for d in ds]
        j = int(np.argmax(np.abs(cands)))
        if ds[j] != 0:
            slips += 1
            k = (k + int(ds[j])) % PERIOD
        g = cands[j] / sub
        rr = ref[(n - k) % PERIOD]
        z = y / g
        errs.append(np.sign(z.real) != rr)
        dist.append(np.mean(np.abs(z - rr) ** 2))
        if st % PERIOD == 0:
            trace.append(int(k))
    e = np.concatenate(errs)
    return e, dist, {"blocks_total": int(nb), "slips": int(slips), "reacquisitions": int(reacq),
                     "trace": trace[:512]}


def frame_stats(e, p):
    nf = len(e) // FRAME_N
    cnt = e[: nf * FRAME_N].reshape(nf, FRAME_N).sum(1)
    out = {}
    for name, k, t in CODES:
        meas = float(np.mean(cnt <= t)) if nf else float("nan")
        iid = sum(comb(FRAME_N, i) * p ** i * (1 - p) ** (FRAME_N - i) for i in range(t + 1))
        out[name] = {"k": k, "t": t, "success_measured": meas, "success_iid_pred": float(iid),
                     "throughput_measured": meas * k / FRAME_N, "throughput_iid_pred": iid * k / FRAME_N}
    hist = np.bincount(cnt, minlength=FRAME_N + 1)
    return out, nf, hist


def burst_stats(e):
    """Lengths of consecutive error runs."""
    if not e.any():
        return {"n_bursts": 0, "mean_len": 0.0, "max_len": 0}
    d = np.diff(np.concatenate([[0], e.astype(int), [0]]))
    lens = np.where(d == -1)[0] - np.where(d == 1)[0]
    return {"n_bursts": int(len(lens)), "mean_len": float(lens.mean()), "max_len": int(lens.max())}


def analyze(xs, ref, mode="track"):
    """xs: list of chunks (each a continuous recording segment)."""
    es, dists, sync, trace = [], [], {}, None
    for x in xs:
        e, d, st = demod(x, ref, mode=mode)
        es.append(e)
        sync.setdefault("per_chunk_ber", []).append(float(e.mean()))
        sync.setdefault("per_chunk_slips", []).append(int(st.get("slips", 0)))
        dists += d
        tr = st.pop("trace")
        trace = tr if trace is None else trace
        for kk, v in st.items():
            if kk not in ("per_chunk_ber", "per_chunk_slips"):
                sync[kk] = sync.get(kk, 0) + v
    e = np.concatenate(es)
    p = float(e.mean()) if len(e) else float("nan")
    sinr_db = float(10 * np.log10(1.0 / (np.mean(dists) + 1e-12))) if dists else float("nan")
    sinr_med_db = float(10 * np.log10(1.0 / (np.median(dists) + 1e-12))) if dists else float("nan")
    fr, nf, hist = frame_stats(e, p)
    return {"n_bits": int(len(e)), "ber": p, "n_errors": int(e.sum()), "sinr_db_est": sinr_db, "sinr_db_median": sinr_med_db,
            "sync": sync, "bursts": burst_stats(e), "n_frames": nf, "frames": fr,
            "frame_err_hist_0to10": [int(v) for v in hist[:11]],
            "shift_trace_chunk0_first512": trace}


def pick_order(src, name):
    """Determine the bit order (MSB/LSB first) by correlation strength."""
    x = src.read(name, 0, PERIOD * 4)
    best = None
    for order in ("big", "little"):
        ref = ref_sequence(order)
        R = np.conj(np.fft.fft(ref))
        r = np.mean([align_block(x[i * PERIOD:(i + 1) * PERIOD], R)[2] for i in range(4)])
        if best is None or r > best[1]:
            best = (order, r)
    return best[0]


def run_file(path, chunks, chunk, tone_band, out_dir):
    tag = os.path.splitext(os.path.basename(path))[0]
    src = MatIQ(path)
    names = [v for v in VARS if v in src.names()]
    order = pick_order(src, names[0])
    ref = ref_sequence(order)
    jm = META.get(tag)
    res = {"file": tag, "jammer_id": jm[0] if jm else None, "rel_power": jm[1] if jm else None,
           "distance_m": jm[2] if jm else None, "bit_order": order, "chunks": chunks, "chunk_len": chunk,
           "assumption": "1 sample = 1 symbol; timing tracked every 256 symbols (+-3 symbol search) and gain/phase estimated per 256 symbols from the known sequence (genie pilot-aided sync); slip errors are counted",
           "source": "Zenodo 7119040 (Alhazbi, Sciancalepore, Oligeri, Data in Brief 2023), CC BY 4.0",
           "vars": {}}
    for name in names:
        n = src.length(name)
        starts = np.linspace(0, n - chunk, chunks).astype(int)
        xs = [src.read(name, int(s), int(s) + chunk) for s in starts]
        r = {}
        r["raw"] = analyze(xs, ref)
        r["raw_resync_every_block"] = analyze(xs, ref, mode="per_block")
        canc = []
        tones = []
        for x in xs:
            y, info = cancel_tone(x, band=tone_band)
            canc.append(y)
            tones += info
        r["tone_cancel"] = analyze(canc, ref)
        tf = np.array(tones)
        det = ~np.isnan(tf[:, 0])
        r["tone_est"] = {"blocks_with_tone": f"{int(det.sum())}/{len(tf)}",
                         "freq_median": float(np.median(tf[det, 0])) if det.any() else None,
                         "power_median": float(np.median(tf[det, 1])) if det.any() else 0.0}
        res["vars"][name] = r
        print(f"[{tag}] {name:9s} BER raw={r['raw']['ber']:.2e} (SINR~{r['raw']['sinr_db_est']:.1f} dB) "
              f"| tone-cancel={r['tone_cancel']['ber']:.2e} (SINR~{r['tone_cancel']['sinr_db_est']:.1f} dB) "
              f"| resync-every-block={r['raw_resync_every_block']['ber']:.2e} "
              f"| slips={r['raw']['sync']['slips']}, reacq={r['raw']['sync']['reacquisitions']} "
              f"\n      per-chunk BER: {' '.join(f'{b:.0e}' for b in r['raw']['sync']['per_chunk_ber'])} "
              f"| t5 frame ok: meas {r['raw']['frames']['bch_t5']['success_measured']:.4f} "
              f"vs iid {r['raw']['frames']['bch_t5']['success_iid_pred']:.4f}")
    os.makedirs(out_dir, exist_ok=True)
    js = os.path.join(out_dir, f"real_ber_{tag}.json")
    with open(js, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"saved {js}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--chunks", type=int, default=8, help="number of chunks spread over the recording")
    ap.add_argument("--chunk", type=int, default=2 ** 20, help="samples per chunk (multiple of 65536)")
    ap.add_argument("--tone-band", type=float, default=None,
                    help="tone search range |f|<=band (normalized). Default: whole band")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    allres = [run_file(p, a.chunks, a.chunk, a.tone_band, a.out_dir) for p in a.paths]
    if len(allres) > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        js = os.path.join(a.out_dir, "real_ber_summary.json")
        summ = [{"file": r["file"], "jammer_id": r["jammer_id"], "rel_power": r["rel_power"],
                 "distance_m": r["distance_m"], **{v: {"ber_raw": r["vars"][v]["raw"]["ber"],
                                           "ber_tone_cancel": r["vars"][v]["tone_cancel"]["ber"],
                                           "ber_resync_every_block": r["vars"][v]["raw_resync_every_block"]["ber"],
                                           "slips": r["vars"][v]["raw"]["sync"]["slips"],
                                           "sinr_db_median": r["vars"][v]["raw"]["sinr_db_median"]}
                                       for v in r["vars"]}} for r in allres]
        with open(js, "w") as fh:
            json.dump(summ, fh, indent=2)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xs = np.arange(len(allres))
        for v, mk in zip(VARS, "os^"):
            ax.semilogy(xs, [max(r["vars"][v]["raw"]["ber"], 1e-7) for r in allres], marker=mk, label=f"{v} raw")
            if v == "Sine":
                ax.semilogy(xs, [max(r["vars"][v]["tone_cancel"]["ber"], 1e-7) for r in allres],
                            marker=mk, ls="--", label=f"{v} + tone cancel")
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{r["file"]}\nJ{r["jammer_id"]} P{r["rel_power"]} {r["distance_m"]}m' if r["jammer_id"]
                            else r["file"] for r in allres], fontsize=8)
        ax.set_ylabel("measured BER (floor 1e-7 = 0 errors)"); ax.grid(ls=":"); ax.legend(fontsize=8)
        ax.set_title("Measured BER on real IQ (Zenodo 7119040)")
        plt.tight_layout()
        png = os.path.join(a.out_dir, "real_ber_summary.png")
        plt.savefig(png, dpi=150)
        print(f"saved {js}, {png}")


if __name__ == "__main__":
    main()
