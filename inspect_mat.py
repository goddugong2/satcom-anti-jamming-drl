"""Inspect the structure of a public IQ dataset .mat file without loading it fully.

Usage:
    python inspect_mat.py path/to/file.mat [--peek 2000]

- MATLAB v7.3 (.mat = HDF5): read lazily with h5py (handles ~4 GB files)
- Older .mat: scipy.io.loadmat (loaded into memory, so watch out for large files)
Prints each variable's name, shape and dtype, plus basic stats of the first --peek samples.
"""
import argparse
import sys

import numpy as np


def peek_stats(x, n):
    x = np.asarray(x[:n]) if x.ndim == 1 else np.asarray(x[:n] if x.shape[0] >= x.shape[-1] else x[..., :n])
    if x.ndim == 2 and 2 in x.shape:  # [N,2] or [2,N] = I/Q columns
        x = x if x.shape[-1] == 2 else x.T
        x = x[:, 0] + 1j * x[:, 1]
    x = x.astype(np.complex64 if np.iscomplexobj(x) else np.float32)
    p = np.mean(np.abs(x) ** 2)
    print(f"    peek {len(x)} samples: mean power {p:.4g} ({10*np.log10(p+1e-30):.1f} dB), "
          f"max |x| {np.max(np.abs(x)):.4g}, mean {np.mean(x):.4g}")


def inspect_h5(path, n):
    import h5py
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape} dtype={obj.dtype}")
                if obj.ndim in (1, 2) and obj.size > 0 and obj.dtype.kind in "fciu":
                    # MATLAB stores data transposed: [2,N] usually means an Nx2 matrix
                    sl = obj[:, :n] if obj.ndim == 2 and obj.shape[0] <= 4 else obj[:n]
                    peek_stats(np.asarray(sl), n)
                elif obj.dtype.names:
                    print(f"    fields: {obj.dtype.names}")
        f.visititems(visit)
        for k, v in f.attrs.items():
            print(f"  attr {k} = {v}")


def inspect_old(path, n):
    from scipy.io import loadmat, whosmat
    for name, shape, cls in whosmat(path):
        print(f"  {name}: shape={shape} class={cls}")
    d = loadmat(path)
    for k, v in d.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind in "fciu" and v.ndim <= 2:
            peek_stats(v if v.ndim == 1 or v.shape[-1] == 2 or v.shape[0] == 2 else v.ravel(), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--peek", type=int, default=2000)
    a = ap.parse_args()
    with open(a.path, "rb") as fh:
        head = fh.read(128)
    print(f"file: {a.path}\nheader: {head[:116].decode('latin1', 'replace').strip()}")
    try:
        import h5py
        if h5py.is_hdf5(a.path):
            print("format: MATLAB v7.3 (HDF5)")
            inspect_h5(a.path, a.peek)
            return
    except ImportError:
        print("(h5py not installed: pip install h5py)")
    print("format: MATLAB v5/v7")
    inspect_old(a.path, a.peek)


if __name__ == "__main__":
    sys.exit(main())
