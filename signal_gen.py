"""RF 신호 합성 + STFT 스펙트로그램 생성 모듈.

- 신호 전력 기준으로 SNR, JSR(dB)을 정확히 맞춤
- 재머 파라미터를 샘플마다 랜덤화
- 시간 빈을 충분히 확보한 뒤 32x32로 보간, 샘플별 표준화
"""
import numpy as np
from scipy import signal
from scipy.ndimage import map_coordinates

# ---------------- 기본 설정 ----------------
FS = 2000.0                  # 샘플링 주파수 (Hz)
DURATION = 0.5               # 관측 구간 (s) -> 1000 샘플
N = int(FS * DURATION)
NYQ = FS / 2
T_AXIS = np.arange(N) / FS
SYMBOL_LEN = 20              # 심볼당 샘플 수 (심볼률 100 baud)
NPERSEG = 64                 # 주파수 빈 33개
NOVERLAP = 48                # hop=16 -> 시간 빈 약 59개
OUT_SIZE = 32                # CNN 입력 크기

CLASSES_4 = ["none", "single_tone", "sweep", "noise"]
CLASSES_5 = CLASSES_4 + ["partial_band"]


def get_rng(seed=None):
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


# ---------------- 정상 신호 ----------------
def generate_bpsk(rng):
    """반송파 주파수/위상을 랜덤화한 BPSK. 평균 전력 1로 정규화."""
    fc = rng.uniform(250, 750)
    phase0 = rng.uniform(0, 2 * np.pi)
    bits = rng.choice([-1.0, 1.0], size=N // SYMBOL_LEN + 1)
    baseband = np.repeat(bits, SYMBOL_LEN)[:N]
    tx = baseband * np.cos(2 * np.pi * fc * T_AXIS + phase0)
    return tx / np.sqrt(np.mean(tx ** 2))


# ---------------- 재머 (단위 전력으로 반환 후 JSR로 스케일링) ----------------
def bandlimited_noise(rng, f_lo, f_hi):
    """FFT 영역에서 [f_lo, f_hi] 대역만 남긴 가우시안 잡음."""
    white = rng.standard_normal(N)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(N, d=1 / FS)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return np.fft.irfft(spec * mask, n=N)


def single_tone_jammer(rng):
    f = rng.uniform(50, 950)
    return np.cos(2 * np.pi * f * T_AXIS + rng.uniform(0, 2 * np.pi))


def sweep_jammer(rng):
    """톱니형 반복 선형 처프. 대역, 주기, 방향, 시작 위상 랜덤."""
    f_lo = rng.uniform(30, 400)
    f_hi = rng.uniform(f_lo + 250, 970)
    period = rng.uniform(0.12, 0.5)
    frac = (T_AXIS / period + rng.uniform(0, 1)) % 1.0
    if rng.random() < 0.5:
        frac = 1.0 - frac           # 하향 스윕
    inst_f = f_lo + (f_hi - f_lo) * frac
    phase = 2 * np.pi * np.cumsum(inst_f) / FS + rng.uniform(0, 2 * np.pi)
    return np.cos(phase)


def broadband_noise_jammer(rng):
    """전 대역(나이퀴스트의 70% 이상)을 덮는 잡음."""
    width = rng.uniform(0.7, 1.0) * NYQ
    f_lo = rng.uniform(0, NYQ - width)
    return bandlimited_noise(rng, f_lo, f_lo + width)


def partial_band_jammer(rng):
    """좁은 대역(나이퀴스트의 5~25%) 잡음."""
    width = rng.uniform(0.05, 0.25) * NYQ
    f_lo = rng.uniform(20, NYQ - width - 20)
    return bandlimited_noise(rng, f_lo, f_lo + width)


JAMMER_FUNCS = {
    "single_tone": single_tone_jammer,
    "sweep": sweep_jammer,
    "noise": broadband_noise_jammer,
    "partial_band": partial_band_jammer,
}


def scale_to_power(x, target_power):
    return x * np.sqrt(target_power / (np.mean(x ** 2) + 1e-12))


# ---------------- 스펙트로그램 ----------------
def spec_to_image(rx):
    """수신 신호 -> dB 스펙트로그램 -> 32x32 -> 샘플별 표준화."""
    _, _, sxx = signal.spectrogram(rx, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP, mode="psd")
    sxx_db = 10 * np.log10(sxx + 1e-12)
    f_bins, t_bins = sxx_db.shape
    rr, cc = np.meshgrid(np.linspace(0, f_bins - 1, OUT_SIZE),
                         np.linspace(0, t_bins - 1, OUT_SIZE), indexing="ij")
    img = map_coordinates(sxx_db, [rr, cc], order=1)
    img = (img - img.mean()) / (img.std() + 1e-8)
    return img.astype(np.float32), sxx_db


def generate_sample(jamming_type="none", snr_db=10.0, jsr_db=5.0, seed=None):
    """하나의 수신 샘플 생성.

    SNR(dB) = 10log10(P_신호 / P_잡음), JSR(dB) = 10log10(P_재머 / P_신호).
    반환: (32x32 이미지, 원본 dB 스펙트로그램)
    """
    rng = get_rng(seed)
    tx = generate_bpsk(rng)
    p_sig = np.mean(tx ** 2)

    rx = tx.copy()
    if jamming_type != "none":
        jam = JAMMER_FUNCS[jamming_type](rng)
        rx += scale_to_power(jam, p_sig * 10 ** (jsr_db / 10))

    noise_var = p_sig / 10 ** (snr_db / 10)
    rx += rng.normal(0, np.sqrt(noise_var), size=N)
    return spec_to_image(rx)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = get_rng(0)
    fig, axes = plt.subplots(1, len(CLASSES_5), figsize=(3 * len(CLASSES_5), 3))
    for ax, name in zip(axes, CLASSES_5):
        _, raw = generate_sample(name, snr_db=10, jsr_db=8, seed=rng)
        ax.imshow(raw, origin="lower", aspect="auto", cmap="jet")
        ax.set_title(name)
        ax.set_xlabel("Time bin")
    axes[0].set_ylabel("Freq bin")
    plt.tight_layout()
    plt.savefig("sample_spectrograms.png", dpi=150)
    print("saved sample_spectrograms.png, raw spec shape:", raw.shape)
