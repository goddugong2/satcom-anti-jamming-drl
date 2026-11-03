"""위성 링크 채널 모델 (도플러 편이, 링크 버짓 SNR, Rician 페이딩, 위상 잡음).

기존 signal_gen.py(AWGN + 재머)는 건드리지 않고, 같은 재머 함수를 재사용해
채널 손상이 더해진 수신 샘플을 만든다. 모든 수치는 공개된 일반 물리식과 '가정값'이며
특정 실제 시스템의 사양이 아니다.

기본 가정과 한계
  - 시뮬레이션은 FS=2 kHz, 100 baud 로 스케일된 기저대역 모델이다. 실제 도플러(Hz)는
    '심볼률 대비 비율'로 환산해 넣는다: fd_sim = (fd_real / Rs_real) * 100 baud.
    따라서 결과는 (반송파, 심볼률) 가정에 의존한다.
  - 관측 구간(스케일 시간 0.5 s = 심볼 50개)은 실제로는 짧아 도플러는 구간 안에서 상수로 본다.
  - 페이딩은 구간 안에서 일정한 flat Rician(평균 전력 1). 재머는 페이딩을 받지 않는 것으로 둔다.
  - 전리층 신틸레이션, 강우 감쇠, 안테나 패턴/지향 오차, 다중경로 지연 확산, 비선형 증폭기는 넣지 않았다.
  - 궤도는 관측자 바로 위를 지나는 원궤도(지구 자전 무시)로 가정한다.
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

import signal_gen as sg

C_KMS = 299792.458
RE_KM = 6371.0
MU_KM3S2 = 398600.4418


# ---------------- 궤도 기하 ----------------
def slant_range_km(h_km, el_deg):
    e = np.radians(el_deg)
    return np.sqrt((RE_KM + h_km) ** 2 - (RE_KM * np.cos(e)) ** 2) - RE_KM * np.sin(e)


def radial_speed_kms(h_km, el_deg):
    """원궤도, 관측자 바로 위 통과 시 시선 방향 속도의 크기 = v * Re/(Re+h) * cos(el)."""
    r = RE_KM + h_km
    v = np.sqrt(MU_KM3S2 / r)
    return v * RE_KM / r * np.cos(np.radians(el_deg))


def doppler_hz(h_km, el_deg, fc_ghz, rising=True):
    """상승(접근) 중이면 +, 하강(멀어짐)이면 -."""
    fd = radial_speed_kms(h_km, el_deg) / C_KMS * fc_ghz * 1e9
    return fd if rising else -fd


# ---------------- 링크 조건 ----------------
@dataclass
class ChannelParams:
    h_km: float = 550.0              # 궤도 고도
    fc_ghz: float = 2.2              # 반송파 (가정값, S대역 수준)
    rs_ksps: float = 100.0           # 심볼률 (가정값)
    snr_zenith_db: float = 15.0      # 천정(앙각 90도)에서의 SNR (가정값)
    atm_zenith_db: float = 0.2       # 천정 대기 손실 (가정값), 앙각이 낮으면 1/sin(el) 로 증가
    k_db_low: float = 3.0            # 앙각 10도에서의 Rician K 인수 (가정값)
    k_db_high: float = 15.0          # 앙각 90도에서의 Rician K 인수 (가정값)
    pn_std: float = 0.005            # 위상 잡음 랜덤워크의 샘플당 표준편차 [rad] (가정값)
    doppler_residual: float = 1.0    # 1 = 도플러 보상 없음, 0 = 완전 보상, 0.1 = 90% 보상
    jammer_doppler: bool = False     # True면 재머도 같은 도플러를 받음 (기본: 지상 재머, 도플러 없음)
    fading: bool = True
    rising: bool = True


def link_snr_db(p, el_deg):
    """천정 SNR 기준으로 자유공간 손실 차이와 대기 손실 차이를 반영."""
    rho = slant_range_km(p.h_km, el_deg)
    rho_z = slant_range_km(p.h_km, 90.0)
    fspl_delta = 20 * np.log10(rho / rho_z)
    atm_delta = p.atm_zenith_db * (1.0 / np.sin(np.radians(el_deg)) - 1.0)
    return p.snr_zenith_db - fspl_delta - atm_delta


def rician_k_db(p, el_deg):
    t = np.clip((el_deg - 10.0) / 80.0, 0.0, 1.0)
    return p.k_db_low + (p.k_db_high - p.k_db_low) * t


def fd_sim_hz(p, el_deg):
    """실제 도플러를 심볼률 대비 비율로 환산해 스케일된 시뮬레이션 주파수로 변환."""
    fd = doppler_hz(p.h_km, el_deg, p.fc_ghz, p.rising) * p.doppler_residual
    return fd / (p.rs_ksps * 1e3) * (sg.FS / sg.SYMBOL_LEN)


def describe(p, el_deg):
    return {
        "el_deg": float(el_deg),
        "slant_km": float(slant_range_km(p.h_km, el_deg)),
        "snr_db": float(link_snr_db(p, el_deg)),
        "doppler_real_khz": float(doppler_hz(p.h_km, el_deg, p.fc_ghz, p.rising) / 1e3),
        "doppler_over_rs": float(doppler_hz(p.h_km, el_deg, p.fc_ghz, p.rising) * p.doppler_residual
                                 / (p.rs_ksps * 1e3)),
        "k_db": float(rician_k_db(p, el_deg)),
    }


# ---------------- 샘플 합성 ----------------
def _rician_gain(rng, k_db):
    k = 10 ** (k_db / 10)
    los = np.sqrt(k / (k + 1))
    sc = np.sqrt(1.0 / (2 * (k + 1)))
    return los + sc * (rng.standard_normal() + 1j * rng.standard_normal())


def _doppler_shift_real(x, fd):
    """실수 신호의 스펙트럼 전체를 fd [Hz] 만큼 이동 (해석 신호 사용)."""
    z = sps.hilbert(x) * np.exp(2j * np.pi * fd * sg.T_AXIS)
    return z.real


def generate_sample_ch(jamming_type, jsr_db, p, el_deg, seed=None, snr_db=None):
    """채널 손상이 있는 수신 샘플. snr_db를 주지 않으면 링크 버짓에서 계산한다.

    SNR과 JSR은 '페이딩 전 평균 신호 전력' 기준이므로, 깊은 페이딩은 실효 SNR을 낮추고
    실효 J/S 를 높인다.
    """
    rng = sg.get_rng(seed)
    if snr_db is None:
        snr_db = link_snr_db(p, el_deg)
    fd = fd_sim_hz(p, el_deg)

    fc = rng.uniform(250, 750)
    f_tx = float(np.clip(fc + fd, 20.0, sg.NYQ - 20.0))
    phase0 = rng.uniform(0, 2 * np.pi)
    g = _rician_gain(rng, rician_k_db(p, el_deg)) if p.fading else 1.0 + 0j
    pn = np.cumsum(rng.normal(0, p.pn_std, sg.N)) if p.pn_std > 0 else 0.0

    bits = rng.choice([-1.0, 1.0], size=sg.N // sg.SYMBOL_LEN + 1)
    baseband = np.repeat(bits, sg.SYMBOL_LEN)[:sg.N]
    carrier = np.cos(2 * np.pi * f_tx * sg.T_AXIS + phase0 + np.angle(g) + pn)
    tx_unit = baseband * carrier
    tx_unit = tx_unit / np.sqrt(np.mean((baseband * np.cos(2 * np.pi * fc * sg.T_AXIS + phase0)) ** 2))
    tx = np.abs(g) * tx_unit                      # 평균 전력 1 기준, 페이딩으로 순간 전력 변동
    p_ref = 1.0

    rx = tx.copy()
    if jamming_type != "none":
        jam = sg.JAMMER_FUNCS[jamming_type](rng)
        if p.jammer_doppler:
            jam = _doppler_shift_real(jam, fd)
        rx += sg.scale_to_power(jam, p_ref * 10 ** (jsr_db / 10))
    rx += rng.normal(0, np.sqrt(p_ref / 10 ** (snr_db / 10)), size=sg.N)
    return sg.spec_to_image(rx)


if __name__ == "__main__":
    p = ChannelParams()
    print(f"{'el':>4} {'slant km':>9} {'SNR dB':>7} {'fd kHz':>8} {'fd/Rs':>7} {'K dB':>5}")
    for el in (10, 20, 30, 45, 60, 90):
        d = describe(p, el)
        print(f"{el:4d} {d['slant_km']:9.0f} {d['snr_db']:7.1f} {d['doppler_real_khz']:8.1f} "
              f"{d['doppler_over_rs']:7.3f} {d['k_db']:5.1f}")
