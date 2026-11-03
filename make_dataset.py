"""CNN 학습/평가용 데이터셋 생성.

출력:
  dataset.npz      train/val/test (70/15/15, 클래스별 층화)
  sweep_tests.npz  JSR별, SNR별 고정 조건 test 세트 (곡선용)
"""
import argparse
import numpy as np
from signal_gen import CLASSES_4, CLASSES_5, generate_sample, get_rng


def make_random(classes, n_per_class, snr_range, jsr_range, rng):
    xs, ys = [], []
    for label, name in enumerate(classes):
        for _ in range(n_per_class):
            snr = rng.uniform(*snr_range)
            jsr = rng.uniform(*jsr_range)
            img, _ = generate_sample(name, snr, jsr, seed=rng)
            xs.append(img)
            ys.append(label)
    x = np.stack(xs)[:, None, :, :].astype(np.float32)
    return x, np.array(ys, dtype=np.int64)


def make_fixed(classes, n_per_class, snr, jsr, rng):
    xs, ys = [], []
    for label, name in enumerate(classes):
        for _ in range(n_per_class):
            img, _ = generate_sample(name, snr, jsr, seed=rng)
            xs.append(img)
            ys.append(label)
    return np.stack(xs)[:, None, :, :].astype(np.float32), np.array(ys, dtype=np.int64)


def stratified_split(y, rng, ratios=(0.7, 0.15, 0.15)):
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_tr = int(len(idx) * ratios[0])
        n_va = int(len(idx) * ratios[1])
        tr.append(idx[:n_tr])
        va.append(idx[n_tr:n_tr + n_va])
        te.append(idx[n_tr + n_va:])
    return [rng.permutation(np.concatenate(p)) for p in (tr, va, te)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=3000)
    ap.add_argument("--n-curve", type=int, default=300, help="곡선용 세트의 클래스당 샘플 수")
    ap.add_argument("--snr-range", type=float, nargs=2, default=[0, 20])
    ap.add_argument("--jsr-range", type=float, nargs=2, default=[0, 15])
    ap.add_argument("--five-class", action="store_true", help="부분대역 잡음 포함 5클래스")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="dataset.npz")
    args = ap.parse_args()

    classes = CLASSES_5 if args.five_class else CLASSES_4
    rng = get_rng(args.seed)

    print(f"[1/2] 메인 데이터셋 생성: {len(classes)}클래스 x {args.n_per_class}장")
    x, y = make_random(classes, args.n_per_class, args.snr_range, args.jsr_range, rng)
    tr, va, te = stratified_split(y, rng)
    np.savez_compressed(args.out,
                        x_train=x[tr], y_train=y[tr],
                        x_val=x[va], y_val=y[va],
                        x_test=x[te], y_test=y[te],
                        classes=np.array(classes))
    print(f"  train {len(tr)} / val {len(va)} / test {len(te)} -> {args.out}")

    print("[2/2] 곡선용 고정 조건 test 세트 생성")
    jsr_levels = np.array([-5, 0, 3, 6, 9, 12, 15], dtype=np.float32)
    snr_levels = np.array([-5, 0, 5, 10, 15, 20], dtype=np.float32)
    fixed_snr, fixed_jsr = 10.0, 8.0   # 한쪽 축을 훑을 때 다른 쪽에 쓰는 값

    x_j, y_j = zip(*[make_fixed(classes, args.n_curve, fixed_snr, j, rng) for j in jsr_levels])
    x_s, y_s = zip(*[make_fixed(classes, args.n_curve, s, fixed_jsr, rng) for s in snr_levels])
    np.savez_compressed("sweep_tests.npz",
                        jsr_levels=jsr_levels, x_jsr=np.stack(x_j), y_jsr=np.stack(y_j),
                        snr_levels=snr_levels, x_snr=np.stack(x_s), y_snr=np.stack(y_s),
                        fixed_snr=fixed_snr, fixed_jsr=fixed_jsr,
                        classes=np.array(classes))
    print("  sweep_tests.npz 저장 완료")
