"""배포 관점 지표 측정: 모델 크기, 연산량, CPU 추론 지연, ONNX 내보내기.

측정 대상
  CNN  : 채널 하나의 스펙트로그램(1x32x32) -> 재밍 분류 (JammingCNN)
  DQN  : 채널 수 C, 이력 k일 때 Q-network (state_dim = k*C + C)
  STFT : 채널 하나의 신호(1000 샘플) -> 32x32 스펙트로그램 전처리
  한 슬롯 전체 = STFT x C채널 + CNN(C채널을 한 배치로) + DQN 한 번

사용:
  python deploy_bench.py --cnn-path cnn_best.pt --channels 32 --history 4
  python deploy_bench.py --cnn-path cnn_best.pt --onnx          (ONNX 파일도 저장)

참고: 지연은 이 PC의 CPU 1스레드 기준 값이다. 임베디드/FPGA 값이 아니며, 그쪽 환경으로 옮길 때의
출발점(모델 크기, 연산량)으로 해석한다.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from dqn import QNetwork
from signal_gen import N, spec_to_image
from train_cnn import load_cnn

torch.set_num_threads(1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_macs(model, x):
    """Conv2d, Linear의 곱셈-덧셈(MAC) 수를 forward hook으로 센다."""
    total = [0]

    def conv_hook(m, inp, out):
        total[0] += out.numel() // out.shape[0] * m.in_channels * m.kernel_size[0] * m.kernel_size[1] // m.groups

    def lin_hook(m, inp, out):
        total[0] += m.in_features * m.out_features

    hs = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return total[0]


def timeit(fn, warmup=50, reps=500):
    """중앙값, 95퍼센타일 (ms)."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-path", default="cnn_best.pt")
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--onnx", action="store_true", help="CNN을 ONNX로 저장(jamming_cnn.onnx)")
    args = ap.parse_args()
    C, k = args.channels, args.history

    cnn, classes = load_cnn(args.cnn_path, torch.device("cpu"))
    cnn.eval()
    dqn = QNetwork(k * C + C, C)
    dqn.eval()
    out = {"channels": C, "history": k, "torch": torch.__version__, "threads": 1}

    x1 = torch.randn(1, 1, 32, 32)
    xc = torch.randn(C, 1, 32, 32)
    s = torch.randn(1, k * C + C)
    rx = np.random.default_rng(0).standard_normal(N)

    p_cnn, p_dqn = count_params(cnn), count_params(dqn)
    m_cnn, m_dqn = count_macs(cnn, x1), count_macs(dqn, s)
    out["cnn"] = {"params": p_cnn, "size_kB_fp32": p_cnn * 4 / 1024, "MACs_per_channel": m_cnn,
                  "file_kB": os.path.getsize(args.cnn_path) / 1024}
    out["dqn"] = {"params": p_dqn, "size_kB_fp32": p_dqn * 4 / 1024, "MACs": m_dqn}

    with torch.no_grad():
        out["lat_ms"] = {
            "stft_1ch": timeit(lambda: spec_to_image(rx), reps=args.reps),
            "cnn_1ch": timeit(lambda: cnn(x1), reps=args.reps),
            f"cnn_{C}ch_batch": timeit(lambda: cnn(xc), reps=args.reps),
            "dqn": timeit(lambda: dqn(s), reps=args.reps),
        }

        def slot():
            for _ in range(C):
                spec_to_image(rx)
            cnn(xc)
            dqn(s)

        out["lat_ms"]["full_slot"] = timeit(slot, warmup=10, reps=max(50, args.reps // 5))

    if args.onnx:
        try:
            torch.onnx.export(cnn, x1, "jamming_cnn.onnx", input_names=["spectrogram"], output_names=["logits"],
                              dynamic_axes={"spectrogram": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
            out["onnx_kB"] = os.path.getsize("jamming_cnn.onnx") / 1024
            print("ONNX 저장: jamming_cnn.onnx")
        except Exception as e:      # onnx/onnxscript 패키지가 없을 때 등
            print(f"ONNX 내보내기 실패: {e}\n  pip install onnx onnxscript 후 다시 실행하세요.")

    print(f"\n[CNN] 파라미터 {p_cnn:,}개 | FP32 {p_cnn * 4 / 1024:.1f} kB | 채널당 {m_cnn / 1e6:.2f} M MACs")
    print(f"[DQN] 파라미터 {p_dqn:,}개 | FP32 {p_dqn * 4 / 1024:.1f} kB | {m_dqn / 1e3:.1f} k MACs (C={C}, k={k})")
    print("\n지연 (CPU 1스레드, 중앙값 / 95퍼센타일, ms)")
    for name, (med, p95) in out["lat_ms"].items():
        print(f"  {name:16s} {med:8.3f} / {p95:8.3f}")
    print(f"\n한 슬롯(센싱 {C}채널 + 판정) 처리 중앙값 {out['lat_ms']['full_slot'][0]:.2f} ms")

    with open("deploy_bench.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved deploy_bench.json")


if __name__ == "__main__":
    main()
