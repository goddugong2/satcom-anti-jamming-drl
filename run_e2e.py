"""End-to-End 실험 실행: RF 신호 -> STFT -> CNN -> DQN 채널 도약.

실험(서브커맨드):
  compare : 한 조건에서 baseline들과 oracle-DQN, CNN-DQN 비교 (학습 곡선 + 최종 성능)
  sweep   : JSR을 바꿔가며 'CNN 탐지 성능 -> 의사결정 성능' 관계 측정
  history : 상태에 이어 붙이는 관측 이력 길이(history)를 바꾸는 ablation
  delay   : 센싱 결과가 d슬롯 늦게 도착하는 상황 (위성 전파 지연 모의) + 이론 상한선
  baselines: 학습 없는 전통 기법(FHSS, AFH 등)을 재머 종류별로 비교 (학습이 없어 빠름)
  scenario: 위성 궤도(LEO/MEO/GEO) 고도, 고각, 슬롯 길이에서 관측 지연 d를 계산해 성능 비교

공통 옵션 예:
  --jammer {sequential,random,reactive,permutation,multi}   (multi는 --n-jammers로 동시 재머 수 지정)
  --obs-quant {soft,binary}   soft: CNN 확률값 그대로 / binary: 0.5 기준 이진 판정만 전달
  --channels 16               채널 수

예)
  python run_e2e.py compare --jammer sequential --jsr 8
  python run_e2e.py sweep --jammer sequential --snr 5 --obs-quant binary --jsr-levels -5 -3 0 2 4 8
  python run_e2e.py compare --jammer multi --n-jammers 3 --jsr 8 --obs-quant binary
  python run_e2e.py history --jammer sequential --snr 5 --jsr -5 --obs-quant binary --histories 1 2 4 8
  python run_e2e.py delay --jammer sequential --jsr 8 --delays 0 1 2 3 5
  python run_e2e.py compare --jammer multi --n-jammers 4 --channels 32 --jsr 8 --obs-quant binary \
         --methods qtable qtable2 --table-episodes 30000      (표만 길게 학습: DQN은 다시 안 돌림)

방법 이름:
  fixed       항상 채널 0
  random      매 슬롯 무작위 채널
  avoid       센싱 결과(CNN)상 재밍 안 된 채널 중 무작위 선택 (학습 없는 센싱 baseline)
  fhss        주파수 도약 확산(FHSS): 송수신이 미리 공유한 의사난수 도약 패턴(채널을 한 번씩 돌며 사용). 센싱 안 씀
  afh         적응형 주파수 도약(AFH): 최근 --afh-window 슬롯 안에 재밍이 감지된 채널을 블랙리스트에 올리고
              나머지 채널 중 무작위로 도약 (블루투스 AFH와 같은 아이디어, 학습 없음)
  qtable      표 기반 Q-learning, 상태 = 가장 최근 관측에서 가장 의심스러운 채널 번호 + 내 직전 채널
  qtable2     위와 같되 최근 2스텝 관측을 상태에 사용 (표 기반의 공정한 강화 버전)
  qtable4     위와 같되 최근 4스텝 관측을 상태에 사용 (채널 8개까지만. DQN의 이력 4와 같은 정보량)
  qmask       표 기반, 상태 = 최근 관측의 감지 비트마스크(0.5 임계) + 내 직전 채널 (채널 16개까지만)
  dqn_oracle  재머 위치를 정확히 아는 DQN (성능 상한)
  dqn_cnn     CNN 관측을 입력으로 쓰는 DQN (실제 시스템)

결과 파일 이름에는 재머, 채널 수, 이력, 지연, 관측 방식이 붙어서 설정이 다른 실험끼리 덮어쓰지 않는다.
"""
import argparse
import copy
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn import DQNAgent, QLearningAgent
from env import AntiJammingEnv, SensingBank, JAMMER_MODELS

torch.set_num_threads(1)

CNN_METHODS = ("avoid", "afh", "qtable", "qtable2", "qtable4", "qmask", "dqn_cnn")
LEARNING_METHODS = ("qtable", "qtable2", "qtable4", "qmask", "dqn_oracle", "dqn_cnn")
ALL_METHODS = ("fixed", "random", "fhss", "avoid", "afh", "qtable", "qtable2", "qtable4", "qmask", "dqn_oracle", "dqn_cnn")
COLORS = {"dqn_oracle": "tab:green", "dqn_cnn": "tab:blue", "qtable": "tab:orange",
          "qtable2": "tab:brown", "qmask": "tab:pink", "qtable4": "tab:olive",
          "avoid": "tab:purple", "random": "crimson", "fixed": "gray",
          "fhss": "tab:cyan", "afh": "tab:red"}
LABELS = {"dqn_oracle": "DQN (oracle observation)", "dqn_cnn": "DQN (CNN observation)",
          "qtable": "Q-table 1-step", "qtable2": "Q-table 2-step", "qtable4": "Q-table 4-step",
          "qmask": "Q-table bitmask",
          "avoid": "Sensing-avoid (no learning)", "random": "Random hopping", "fixed": "Fixed channel",
          "fhss": "FHSS (pre-shared pattern)", "afh": "AFH (blacklist)"}
MARKERS = {"dqn_cnn": "o", "dqn_oracle": "o", "qtable": "D", "qtable2": "v", "qtable4": "X",
           "qmask": "P", "avoid": "^"}
MAX_MASK_CHANNELS = 16     # 2^C 상태 표를 만들 수 있는 최대 채널 수
MAX_QTABLE4_CHANNELS = 8   # 4스텝 표는 (C+1)^5 상태라 채널 8개까지만 (9^5 = 59049)


def active(methods, args):
    """상태 수가 너무 커지는 표 기반 방법은 제외한다."""
    return [m for m in methods
            if not (m == "qmask" and args.channels > MAX_MASK_CHANNELS)
            and not (m == "qtable4" and args.channels > MAX_QTABLE4_CHANNELS)]


def episodes_for(method, args):
    """표 기반 방법은 --table-episodes로 더 오래 학습시킬 수 있다 (상태가 많아 수렴이 느리므로)."""
    if method.startswith("qtable") or method == "qmask":
        return args.table_episodes or args.episodes
    return args.episodes


# ---------------- 정책 어댑터 ----------------
class Policy:
    def act(self, s, greedy=False):
        raise NotImplementedError

    def learn(self, s, a, r, ns, done):
        pass


class FixedPolicy(Policy):
    def act(self, s, greedy=False):
        return 0


class RandomPolicy(Policy):
    def __init__(self, n, seed):
        self.n, self.rng = n, np.random.default_rng(seed)

    def act(self, s, greedy=False):
        return int(self.rng.integers(self.n))


class AvoidPolicy(Policy):
    def __init__(self, env, seed):
        self.env, self.rng = env, np.random.default_rng(seed)

    def act(self, s, greedy=False):
        cand = np.flatnonzero(self.env.newest_obs(s) < 0.5)
        if len(cand) == 0:
            cand = np.arange(self.env.C)
        return int(self.rng.choice(cand))


class FHSSPolicy(Policy):
    """FHSS: 송수신이 미리 공유한 의사난수 도약 패턴. 채널을 한 번씩 돌며(순열 단위) 쓰고 센싱은 쓰지 않는다.
    패턴은 프레임(에피소드)마다 반복된다. 재머가 패턴을 모르면 통계적으로 무작위 도약과 같아진다."""

    def __init__(self, env, seed):
        self.env = env
        rng = np.random.default_rng(seed + 777)
        blocks = -(-env.episode_len // env.C)
        self.pattern = np.concatenate([rng.permutation(env.C) for _ in range(blocks)])

    def act(self, s, greedy=False):
        return int(self.pattern[self.env.time_step % len(self.pattern)])


class AFHPolicy(Policy):
    """AFH: 최근 window 슬롯 안에 재밍이 감지(>=0.5)된 채널을 블랙리스트로 두고, 나머지 중 무작위 도약.
    avoid는 가장 최근 관측만 보지만 AFH는 window 슬롯 동안의 감지를 누적한다."""

    def __init__(self, env, seed, window=4):
        self.env, self.window = env, window
        self.rng = np.random.default_rng(seed)

    def act(self, s, greedy=False):
        w = max(1, min(self.window, self.env.k))
        bad = np.zeros(self.env.C, dtype=bool)
        for j in range(w):
            bad |= self.env.obs_at(s, j) >= 0.5
        cand = np.flatnonzero(~bad)
        if len(cand) == 0:
            cand = np.arange(self.env.C)
        return int(self.rng.choice(cand))


class QTablePolicy(Policy):
    """표 기반 정책.
    mode='argmax': 최근 k스텝 관측 각각에서 p_jam이 가장 큰 채널 번호(감지 없음=C) + 내 직전 채널
    mode='mask'  : 가장 최근 관측을 0.5 임계로 이진화한 감지 비트마스크 + 내 직전 채널
    """

    def __init__(self, env, agent, mode="argmax", k=1):
        self.env, self.agent, self.mode, self.k = env, agent, mode, k

    @staticmethod
    def n_states(C, mode, k):
        return (2 ** C if mode == "mask" else (C + 1) ** k) * (C + 1)

    def _sensed(self, obs):
        if obs.max() <= 0:          # 아무 것도 감지 못함(이진 모드) 또는 시작 시 빈 이력
            return self.env.C
        return int(np.argmax(obs))

    def encode(self, s):
        C = self.env.C
        if self.mode == "mask":
            bits = (self.env.obs_at(s, 0) >= 0.5).astype(np.int64)
            idx = int((bits << np.arange(C)).sum())
        else:
            idx = 0
            for j in reversed(range(self.k)):        # 오래된 관측부터 이어 붙임
                idx = idx * (C + 1) + self._sensed(self.env.obs_at(s, j))
        return idx * (C + 1) + self.env.last_action_of(s) + 1

    def act(self, s, greedy=False):
        return self.agent.select_action(self.encode(s), greedy)

    def learn(self, s, a, r, ns, done):
        self.agent.update(self.encode(s), a, r, self.encode(ns))


class DQNPolicy(Policy):
    def __init__(self, agent):
        self.agent = agent

    def act(self, s, greedy=False):
        return self.agent.select_action(s, greedy)

    def learn(self, s, a, r, ns, done):
        # 100스텝 시간 제한으로 끝나는 것이지 진짜 종료 상태가 아니므로 bootstrap 유지(d=0)
        self.agent.remember(s, a, r, ns, 0.0)
        self.agent.train_step()


# ---------------- 실행 보조 ----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def with_args(args, **kw):
    a = copy.copy(args)
    vars(a).update(kw)
    return a


def jname(args):
    return f"multi{args.n_jammers}" if args.jammer == "multi" else args.jammer


def qtag(args):
    """파일 이름 꼬리: 이진 관측이면 _bin, --out-tag를 줬으면 _그이름 (이전 결과가 덮어쓰이지 않게)."""
    return ("_bin" if args.obs_quant == "binary" else "") + (f"_{args.out_tag}" if args.out_tag else "")


def tag(args):
    return f"{jname(args)}_c{args.channels}_h{args.history}_d{args.obs_delay}{qtag(args)}"


def theory_ceiling(jammer, delay, C, n_jammers=3, p_follow=0.8, p_react=0.7):
    """최적 정책의 이론(근사) 성공률(%). 재머의 무작위 도약 때문에 생기는 피할 수 없는 피격만 센다.

    순차형/순환형: 관측이 d슬롯 늦으면 (d+1)스텝 뒤를 예측해야 한다. 그 사이 한 번도
    무작위 도약이 없을 확률은 p^(d+1)이고, 아니면 재머 위치는 균등분포가 된다.
    최적 정책은 예측 채널을 피하므로 피격 확률은 (1 - p^(d+1)) / C.
    다중형: 재머가 n개이고 서로 독립이라고 보고 (1 - 단일 피격확률)^n (겹침과 안전 채널 부족은 무시한 근사).
    반응형: 지연과 무관(내 직전 채널은 지연 없이 알 수 있음).
    """
    per = (1.0 - p_follow ** (delay + 1)) / C
    if jammer in ("sequential", "permutation"):
        return 100.0 * (1.0 - per)
    if jammer == "multi":
        return 100.0 * (1.0 - per) ** n_jammers
    if jammer == "reactive":
        return 100.0 * (1.0 - (1.0 - p_react) / C)
    return 100.0 * (1.0 - 1.0 / C)


def ceiling_of(args, delay=None):
    d = args.obs_delay if delay is None else delay
    return theory_ceiling(args.jammer, d, args.channels, args.n_jammers)


def make_env(args, obs_mode, bank, seed, pattern_seed=None):
    return AntiJammingEnv(num_channels=args.channels, jammer=args.jammer, obs_mode=obs_mode,
                          bank=bank, history=args.history, episode_len=args.episode_len,
                          switch_cost=args.switch_cost, seed=seed,
                          pattern_seed=seed if pattern_seed is None else pattern_seed,
                          obs_delay=args.obs_delay, n_jammers=args.n_jammers,
                          obs_quant=args.obs_quant)


def run_episode(env, policy, greedy, learn):
    s = env.reset()
    hits, done = 0, False
    while not done:
        a = policy.act(s, greedy)
        ns, r, done, hit = env.step(a)
        if learn:
            policy.learn(s, a, r, ns, done)
        s = ns
        hits += hit
    return 100.0 * (1.0 - hits / env.episode_len)       # 통신 성공률(%)


def build_policy(method, env, args, seed):
    decay = int(0.4 * episodes_for(method, args) * args.episode_len)   # 학습 구간의 40% 동안 epsilon 감소
    if method == "fixed":
        return FixedPolicy()
    if method == "random":
        return RandomPolicy(env.C, seed)
    if method == "avoid":
        return AvoidPolicy(env, seed)
    if method == "fhss":
        return FHSSPolicy(env, seed)
    if method == "afh":
        return AFHPolicy(env, seed, args.afh_window)
    if method in ("qtable", "qtable2", "qtable4", "qmask"):
        mode = "mask" if method == "qmask" else "argmax"
        k = min({"qtable": 1, "qtable2": 2, "qtable4": 4, "qmask": 1}[method], env.k)
        agent = QLearningAgent(QTablePolicy.n_states(env.C, mode, k), env.C,
                               eps_decay_steps=decay, seed=seed)
        return QTablePolicy(env, agent, mode, k)
    agent = DQNAgent(env.state_dim, env.action_dim, eps_decay_steps=decay, seed=seed)
    return DQNPolicy(agent)


def run_method(method, args, bank, seed, verbose=False):
    set_seed(seed)
    obs_mode = "cnn" if method in CNN_METHODS else "oracle"
    env = make_env(args, obs_mode, bank, seed)
    policy = build_policy(method, env, args, seed)

    curve = []
    if method in LEARNING_METHODS:
        n_ep = episodes_for(method, args)
        for ep in range(n_ep):
            curve.append(run_episode(env, policy, greedy=False, learn=True))
            if verbose and (ep + 1) % max(50, n_ep // 6) == 0:
                print(f"    [{method} seed{seed}] ep {ep + 1:4d}/{n_ep} | "
                      f"최근 20회 평균 {np.mean(curve[-20:]):.1f}%")

    # 평가는 별도 환경(다른 난수), 탐험 없음. 재머 패턴(permutation 순서)은 학습 환경과 동일해야 함
    eval_env = make_env(args, obs_mode, bank, seed + 10_000, pattern_seed=seed)
    if hasattr(policy, "env"):
        policy.env = eval_env
    rates = [run_episode(eval_env, policy, greedy=True, learn=False)
             for _ in range(args.eval_episodes)]
    return {"curve": curve, "eval": float(np.mean(rates))}


def moving_avg(x, w=10):
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def summarize(vals):
    return float(np.mean(vals)), float(np.std(vals))


def need_cnn(args):
    import os
    if not os.path.exists(args.cnn_path):
        raise SystemExit(f"CNN 체크포인트가 없습니다: {args.cnn_path}\n"
                         f"먼저 make_dataset.py -> train_cnn.py를 실행하세요.")


def seeds_of(method, args, bank):
    return [run_method(method, args, bank, s)["eval"] for s in range(args.seeds)]


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def clean_args(args):
    return {k: v for k, v in vars(args).items() if k != "fn"}


def bank_line(bank, args):
    return (f"CNN 재밍 탐지(임계 0.5) TPR {bank.tpr * 100:.1f}% FPR {bank.fpr * 100:.1f}% "
            f"(균형 {bank.detect_acc * 100:.1f}%) | 재머 위치 식별률(확률 최대 채널) "
            f"{bank.localization_acc(args.channels) * 100:.1f}% | 관측 방식: {args.obs_quant}")


# ---------------- 실험 1: compare ----------------
def cmd_compare(args):
    need_cnn(args)
    t0 = time.time()
    bank = SensingBank.build(args.cnn_path, args.snr, args.jsr, seed=0)
    print(f"[센싱 은행] SNR {args.snr} dB, JSR {args.jsr} dB | {bank_line(bank, args)}")
    print(f"[설정] 재머={jname(args)}, 채널={args.channels}, 에피소드={args.episodes}, "
          f"시드={args.seeds}, 이력={args.history}, 관측 지연={args.obs_delay}\n")

    ms = active(ALL_METHODS, args)
    if args.methods:     # --methods로 학습 방법을 골랐으면 그것만 실행 (fixed/random/avoid는 항상 포함)
        ms = [m for m in ms if m in ("fixed", "random", "fhss", "avoid", "afh") or m in args.methods]
    res = {m: {"curves": [], "eval": []} for m in ms}
    for seed in range(args.seeds):
        for m in ms:
            out = run_method(m, args, bank, seed, verbose=(seed == 0))
            res[m]["eval"].append(out["eval"])
            if out["curve"]:
                res[m]["curves"].append(out["curve"])
            print(f"  seed {seed} | {m:11s} 최종 평가 성공률 {out['eval']:.1f}%")

    ceiling = ceiling_of(args)
    print(f"\n===== 결과 요약 (재머: {jname(args)}, 채널 {args.channels}, 관측 지연 {args.obs_delay}) =====")
    summary = {}
    for m in ms:
        mean, std = summarize(res[m]["eval"])
        summary[m] = {"mean": mean, "std": std}
        print(f"  {m:11s} {mean:6.2f} ± {std:4.2f} %")
    print(f"  (이론 상한 {ceiling:.2f} %)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    for m in ms:
        if m in LEARNING_METHODS:
            sm = moving_avg(np.mean(res[m]["curves"], axis=0), 10)
            a1.plot(np.arange(len(sm)) + 9, sm, color=COLORS[m], lw=2, label=f"{m} (10-ep MA)")
    for m, ls in (("random", ":"), ("avoid", "--")):
        a1.axhline(summary[m]["mean"], color="gray", ls=ls, lw=1.3, label=f"{m} ({summary[m]['mean']:.1f}%)")
    a1.axhline(ceiling, color="black", ls=(0, (1, 3)), lw=1.2, label=f"theory ceiling ({ceiling:.1f}%)")
    a1.set_xlabel("Training episode"); a1.set_ylabel("Communication success rate (%)")
    a1.set_title(f"Learning curves | {jname(args)}, C={args.channels}, SNR {args.snr} dB, JSR {args.jsr} dB, "
                 f"delay {args.obs_delay}, obs {args.obs_quant}", fontsize=9)
    low = min(summary[m]["mean"] for m in ("random", "avoid")) - 10   # fixed가 너무 낮아도 곡선이 눌리지 않게
    a1.set_ylim(max(0, low), 102)
    a1.grid(ls=":"); a1.legend(loc="lower right", fontsize=7)

    means = [summary[m]["mean"] for m in ms]
    stds = [summary[m]["std"] for m in ms]
    a2.barh(ms, means, xerr=stds, color=[COLORS[m] if m in LEARNING_METHODS else "lightgray" for m in ms],
            capsize=3)
    for i, v in enumerate(means):
        a2.text(min(v + 0.5, 99), i, f"{v:.1f}", va="center", fontsize=9)
    a2.set_xlim(max(0, min(means) - 10), 103)
    a2.set_xlabel("Final eval success rate (%, greedy)"); a2.set_title("Final performance")
    a2.grid(axis="x", ls=":")
    plt.tight_layout()
    plt.savefig(f"compare_{tag(args)}.png", dpi=250)
    dump(f"compare_{tag(args)}.json", {"args": clean_args(args), "bank": {"tpr": bank.tpr, "fpr": bank.fpr},
                                       "ceiling": ceiling, "summary": summary, "raw": res})
    print(f"\nsaved compare_{tag(args)}.png/.json  ({time.time() - t0:.0f}s)")


# ---------------- 실험 2: sweep (JSR) ----------------
def cmd_sweep(args):
    need_cnn(args)
    t0 = time.time()
    print(f"[설정] 재머={jname(args)}, 채널={args.channels}, SNR {args.snr} dB 고정, JSR 수준={args.jsr_levels}, "
          f"에피소드={args.episodes}, 시드={args.seeds}, 이력={args.history}, 지연={args.obs_delay}, "
          f"관측 방식={args.obs_quant}\n")

    ref = {}
    for m in ("dqn_oracle", "random"):
        ref[m] = summarize(seeds_of(m, args, None))[0]
        print(f"  기준선 {m:11s} {ref[m]:.1f}%")

    ms = active(("dqn_cnn", "qtable", "qtable2", "qtable4", "qmask", "avoid"), args)
    rows = []
    for jsr in args.jsr_levels:
        bank = SensingBank.build(args.cnn_path, args.snr, jsr, seed=0)
        loc = bank.localization_acc(args.channels)
        row = {"jsr": jsr, "tpr": bank.tpr, "fpr": bank.fpr, "det_acc": bank.detect_acc, "loc_acc": loc}
        for m in ms:
            row[m] = summarize(seeds_of(m, args, bank))
        rows.append(row)
        print(f"  JSR {jsr:+5.1f} dB | 이진 탐지 {bank.detect_acc * 100:5.1f}% (TPR {bank.tpr * 100:.0f}% "
              f"FPR {bank.fpr * 100:.0f}%) | 위치 식별 {loc * 100:5.1f}% | "
              + " | ".join(f"{m} {row[m][0]:.1f}±{row[m][1]:.1f}" for m in ms))

    x = [r["jsr"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5.3))
    for m in ms:
        ax.errorbar(x, [r[m][0] for r in rows], yerr=[r[m][1] for r in rows], color=COLORS[m],
                    marker=MARKERS[m], lw=2 if m != "avoid" else 1.3, capsize=3, label=LABELS[m])
    ax.axhline(ref["dqn_oracle"], color=COLORS["dqn_oracle"], ls="--", label=f"DQN oracle ({ref['dqn_oracle']:.1f}%)")
    ax.axhline(ref["random"], color=COLORS["random"], ls=":", label=f"Random hopping ({ref['random']:.1f}%)")
    ax.set_xlabel("JSR (dB)"); ax.set_ylabel("Communication success rate (%)")
    ax.set_title(f"Detection quality vs. decision quality | {jname(args)}, C={args.channels}, SNR {args.snr} dB, "
                 f"obs {args.obs_quant}", fontsize=10)
    ax.grid(ls=":")
    ax2 = ax.twinx()
    ax2.plot(x, [r["det_acc"] * 100 for r in rows], color="gray", ls="-.", marker="s", alpha=0.8,
             label="CNN binary detection (balanced acc.)")
    ax2.plot(x, [r["loc_acc"] * 100 for r in rows], color="black", ls="-.", marker="x", alpha=0.8,
             label="Jammed-channel localization (argmax)")
    ax2.set_ylabel("CNN accuracy (%)"); ax2.set_ylim(40, 102)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"jsr_sweep_{tag(args)}.png", dpi=250)
    dump(f"jsr_sweep_{tag(args)}.json", {"args": clean_args(args), "reference": ref, "rows": rows})
    print(f"\nsaved jsr_sweep_{tag(args)}.png/.json  ({time.time() - t0:.0f}s)")


# ---------------- 실험 3: history ablation ----------------
def cmd_history(args):
    need_cnn(args)
    t0 = time.time()
    bank = SensingBank.build(args.cnn_path, args.snr, args.jsr, seed=0)
    print(f"[센싱 은행] SNR {args.snr} dB, JSR {args.jsr} dB | {bank_line(bank, args)}")
    print(f"[설정] 재머={jname(args)}, 채널={args.channels}, 이력 길이={args.histories}, "
          f"시드={args.seeds}, 지연={args.obs_delay}\n")

    base = {}
    for m in active(("qtable", "qtable2", "qtable4", "qmask", "random"), args):
        base[m] = summarize(seeds_of(m, args, None if m == "random" else bank))
        print(f"  baseline {m:8s} {base[m][0]:.1f}±{base[m][1]:.1f}%")

    rows = []
    for h in args.histories:
        a = with_args(args, history=h)
        row = {"history": h, "dqn_cnn": summarize(seeds_of("dqn_cnn", a, bank)),
               "dqn_oracle": summarize(seeds_of("dqn_oracle", a, None))}
        rows.append(row)
        print(f"  history {h:2d} | DQN-CNN {row['dqn_cnn'][0]:.1f}±{row['dqn_cnn'][1]:.1f}% | "
              f"DQN-oracle {row['dqn_oracle'][0]:.1f}±{row['dqn_oracle'][1]:.1f}%")

    x = [r["history"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ("dqn_cnn", "dqn_oracle"):
        ax.errorbar(x, [r[m][0] for r in rows], yerr=[r[m][1] for r in rows], color=COLORS[m],
                    marker="o", lw=2, capsize=3, label=LABELS[m])
    for m in base:
        ax.axhline(base[m][0], color=COLORS[m], ls=":" if m == "random" else "--",
                   label=f"{LABELS[m]} ({base[m][0]:.1f}%)")
    ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels([str(h) for h in x])
    ax.set_xlabel("Observation history length (slots)"); ax.set_ylabel("Communication success rate (%)")
    ax.set_title(f"History ablation | {jname(args)}, C={args.channels}, SNR {args.snr} dB, JSR {args.jsr} dB, "
                 f"obs {args.obs_quant}", fontsize=10)
    ax.grid(ls=":"); ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    name = f"history_ablation_{jname(args)}_c{args.channels}_jsr{args.jsr:g}_d{args.obs_delay}{qtag(args)}"
    plt.savefig(name + ".png", dpi=250)
    dump(name + ".json", {"args": clean_args(args), "baselines": base, "rows": rows})
    print(f"\nsaved {name}.png/.json  ({time.time() - t0:.0f}s)")


# ---------------- 실험 4: delay ----------------
def cmd_delay(args):
    need_cnn(args)
    t0 = time.time()
    bank = SensingBank.build(args.cnn_path, args.snr, args.jsr, seed=0)
    print(f"[센싱 은행] SNR {args.snr} dB, JSR {args.jsr} dB | {bank_line(bank, args)}")
    print(f"[설정] 재머={jname(args)}, 채널={args.channels}, 관측 지연={args.delays}, 이력={args.history}, "
          f"에피소드={args.episodes}, 시드={args.seeds}\n")

    ms = active(("random", "avoid", "qtable", "qtable2", "qtable4", "qmask", "dqn_oracle", "dqn_cnn"), args)
    rows = []
    for d in args.delays:
        a = with_args(args, obs_delay=d)
        row = {"delay": d, "ceiling": ceiling_of(args, d)}
        for m in ms:
            row[m] = summarize(seeds_of(m, a, None if m in ("random", "dqn_oracle") else bank))
        rows.append(row)
        print(f"  delay {d} | " + " | ".join(f"{m} {row[m][0]:.1f}" for m in ms)
              + f" | 이론 상한 {row['ceiling']:.1f}")

    x = [r["delay"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for m in ms:
        if m in ("random",):
            ax.plot(x, [r[m][0] for r in rows], color=COLORS[m], ls=":", label=LABELS[m])
        elif m == "avoid":
            ax.plot(x, [r[m][0] for r in rows], color=COLORS[m], marker="^", lw=1.3, label=LABELS[m])
        else:
            ax.errorbar(x, [r[m][0] for r in rows], yerr=[r[m][1] for r in rows], color=COLORS[m],
                        marker=MARKERS[m], lw=2, capsize=3, label=LABELS[m])
    ax.plot(x, [r["ceiling"] for r in rows], color="black", ls="--", label="Theoretical ceiling (approx.)")
    ax.set_xticks(x)
    ax.set_xlabel("Observation delay (slots)  [satellite propagation delay model]")
    ax.set_ylabel("Communication success rate (%)")
    ax.set_title(f"Effect of delayed sensing | {jname(args)}, C={args.channels}, SNR {args.snr} dB, "
                 f"JSR {args.jsr} dB, h={args.history}, obs {args.obs_quant}", fontsize=9)
    ax.grid(ls=":"); ax.legend(loc="lower left", fontsize=7)
    plt.tight_layout()
    name = f"delay_{jname(args)}_c{args.channels}_h{args.history}_jsr{args.jsr:g}{qtag(args)}"
    plt.savefig(name + ".png", dpi=250)
    dump(name + ".json", {"args": clean_args(args), "rows": rows})
    print(f"\nsaved {name}.png/.json  ({time.time() - t0:.0f}s)")


# ---------------- 실험 5: baselines (학습 없는 전통 기법 비교) ----------------
BASE_METHODS = ("fixed", "random", "fhss", "avoid", "afh")


def cmd_baselines(args):
    need_cnn(args)
    t0 = time.time()
    bank = SensingBank.build(args.cnn_path, args.snr, args.jsr, seed=0)
    print(f"[센싱 은행] SNR {args.snr} dB, JSR {args.jsr} dB | {bank_line(bank, args)}")
    print(f"[설정] 채널={args.channels}, 재머={args.jammers}, 시드={args.seeds}, 지연={args.obs_delay}, "
          f"이력={args.history}, AFH 창={args.afh_window}\n")

    ref = {}
    if args.ref_compare:
        with open(args.ref_compare) as f:
            r = json.load(f)
        ref = {"jammer": r["args"]["jammer"], "n": r["args"].get("n_jammers"),
               "summary": {m: r["summary"][m] for m in ("qtable", "qtable2", "dqn_cnn") if m in r["summary"]}}

    rows = {}
    for j in args.jammers:
        a = with_args(args, jammer=j)
        row = {"ceiling": ceiling_of(a)}
        for m in BASE_METHODS:
            row[m] = summarize(seeds_of(m, a, bank if m in CNN_METHODS else None))
        if ref and ref["jammer"] == j and (j != "multi" or ref["n"] == args.n_jammers):
            for m, v in ref["summary"].items():
                row[m] = (v["mean"], v["std"])
        rows[jname(a)] = row
        print(f"  {jname(a):10s} | " + " | ".join(f"{m} {row[m][0]:.1f}" for m in row if m != "ceiling")
              + f" | 이론 상한 {row['ceiling']:.1f}")

    names = list(rows)
    shown = [m for m in (*BASE_METHODS, "qtable", "qtable2", "dqn_cnn") if any(m in r for r in rows.values())]
    fig, ax = plt.subplots(figsize=(1.9 * len(names) + 4, 5.2))
    w = 0.8 / len(shown)
    for i, m in enumerate(shown):
        xs = [k + (i - (len(shown) - 1) / 2) * w for k, n in enumerate(names) if m in rows[n]]
        ys = [rows[n][m][0] for n in names if m in rows[n]]
        es = [rows[n][m][1] for n in names if m in rows[n]]
        ax.bar(xs, ys, w, yerr=es, color=COLORS[m], capsize=2, label=LABELS[m])
    ax.plot(range(len(names)), [rows[n]["ceiling"] for n in names], "k_", ms=28, mew=1.5, label="Theory ceiling (approx.)")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    lo = min(rows[n][m][0] for n in names for m in BASE_METHODS)
    ax.set_ylim(max(0, lo - 8), 100.5)
    ax.set_ylabel("Communication success rate (%)")
    ax.set_title(f"Conventional anti-jam baselines | C={args.channels}, SNR {args.snr} dB, JSR {args.jsr} dB, "
                 f"delay {args.obs_delay}, obs {args.obs_quant}", fontsize=10)
    ax.grid(axis="y", ls=":")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.07))
    plt.tight_layout()
    name = f"baselines_c{args.channels}_n{args.n_jammers}_jsr{args.jsr:g}_d{args.obs_delay}{qtag(args)}"
    plt.savefig(name + ".png", dpi=250)
    dump(name + ".json", {"args": clean_args(args), "rows": rows})
    print(f"\nsaved {name}.png/.json  ({time.time() - t0:.0f}s)")


# ---------------- 실험 6: scenario (궤도 고도 -> 관측 지연 슬롯 수) ----------------
EARTH_R_KM = 6371.0
C_KM_S = 299792.458
SCENARIOS = {"leo550": 550.0, "leo1200": 1200.0, "meo8000": 8063.0, "geo": 35786.0}   # 공개된 대표 고도(km)
SCENARIO_LABELS = {"leo550": "LEO 550km", "leo1200": "LEO 1200km", "meo8000": "MEO 8063km", "geo": "GEO 35786km"}


def slant_range_km(alt_km, elev_deg):
    """지상 단말에서 고각 elev_deg로 본 위성까지의 사거리(km)."""
    e = np.radians(elev_deg)
    r = EARTH_R_KM
    return float(np.sqrt((r + alt_km) ** 2 - (r * np.cos(e)) ** 2) - r * np.sin(e))


def scenario_delay_slots(alt_km, args, slot_ms):
    """센싱 결과가 에이전트에게 도착하기까지의 지연(ms)을 슬롯 수로 환산: d = ceil(지연 / 슬롯)."""
    one_way = slant_range_km(alt_km, args.elev) / C_KM_S * 1000.0
    path = one_way * (2.0 if args.feedback == "rtt" else 1.0) + args.proc_ms
    return one_way, path, int(np.ceil(path / slot_ms - 1e-9))


def cmd_scenario(args):
    need_cnn(args)
    t0 = time.time()
    slots = args.slot_ms * (len(args.scenarios) if len(args.slot_ms) == 1 else 1)
    assert len(slots) == len(args.scenarios), "--slot-ms는 1개 또는 시나리오 수만큼 주세요"
    plan = []
    print(f"[가정] 고각 {args.elev}도, 피드백 경로={'왕복' if args.feedback == 'rtt' else '편도'}, "
          f"처리 지연 {args.proc_ms} ms")
    print(f"  {'시나리오':12s} {'사거리(km)':>10s} {'편도(ms)':>9s} {'경로(ms)':>9s} {'슬롯(ms)':>9s} {'지연 d':>7s}")
    for name, slot in zip(args.scenarios, slots):
        one_way, path, d = scenario_delay_slots(SCENARIOS[name], args, slot)
        plan.append({"scenario": name, "slot_ms": slot, "one_way_ms": one_way, "path_ms": path, "d": d,
                     "slant_km": slant_range_km(SCENARIOS[name], args.elev)})
        print(f"  {name:12s} {plan[-1]['slant_km']:10.0f} {one_way:9.2f} {path:9.2f} {slot:9.1f} {d:7d}")

    bank = SensingBank.build(args.cnn_path, args.snr, args.jsr, seed=0)
    print(f"\n[센싱 은행] SNR {args.snr} dB, JSR {args.jsr} dB | {bank_line(bank, args)}\n")
    ms = active(("random", "avoid", "qtable", "qtable2", "dqn_oracle", "dqn_cnn"), args)
    cache = {}
    for row in plan:
        d = row["d"]
        if d not in cache:
            a = with_args(args, obs_delay=d)
            res = {"ceiling": ceiling_of(args, d)}
            for m in ms:
                res[m] = summarize(seeds_of(m, a, None if m in ("random", "dqn_oracle") else bank))
            cache[d] = res
            print(f"  d={d:3d} | " + " | ".join(f"{m} {res[m][0]:.1f}" for m in ms)
                  + f" | 이론 상한 {res['ceiling']:.1f}")
        row.update(cache[d])

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    xs = range(len(plan))
    for m in ms:
        ax.errorbar(xs, [r[m][0] for r in plan], yerr=[r[m][1] for r in plan], color=COLORS[m],
                    marker=MARKERS.get(m, "o"), lw=2 if m != "random" else 1.2,
                    ls=":" if m == "random" else "-", capsize=3, label=LABELS[m])
    ax.plot(xs, [r["ceiling"] for r in plan], "k--", label="Theoretical ceiling (approx.)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{SCENARIO_LABELS[r['scenario']]}\npath {r['path_ms']:.1f} ms\nslot {r['slot_ms']:g} ms\n d={r['d']}"
                        for r in plan], fontsize=8)
    ax.set_ylabel("Communication success rate (%)")
    ax.set_title(f"Orbit scenarios (elev {args.elev:g} deg, {'RTT' if args.feedback == 'rtt' else 'one-way'} feedback) "
                 f"| {jname(args)}, C={args.channels}, JSR {args.jsr} dB, obs {args.obs_quant}", fontsize=9)
    ax.grid(ls=":"); ax.legend(loc="lower left", fontsize=7)
    plt.tight_layout()
    name = f"scenario_{jname(args)}_c{args.channels}_el{args.elev:g}_{args.feedback}{qtag(args)}"
    plt.savefig(name + ".png", dpi=250)
    dump(name + ".json", {"args": clean_args(args), "plan": plan})
    print(f"\nsaved {name}.png/.json  ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--jammer", choices=JAMMER_MODELS, default="sequential")
        p.add_argument("--n-jammers", type=int, default=3, help="multi 재머에서 동시에 공격하는 재머 수")
        p.add_argument("--obs-quant", choices=("soft", "binary"), default="soft",
                       help="soft: CNN 확률값 그대로 관측 / binary: 0.5 기준 이진 판정만 관측")
        p.add_argument("--cnn-path", default="cnn_best.pt")
        p.add_argument("--snr", type=float, default=10.0)
        p.add_argument("--channels", type=int, default=8)
        p.add_argument("--history", type=int, default=4, help="상태에 이어 붙일 관측 스텝 수")
        p.add_argument("--obs-delay", type=int, default=0, help="센싱 결과가 늦게 도착하는 슬롯 수")
        p.add_argument("--episodes", type=int, default=300)
        p.add_argument("--table-episodes", type=int, default=0,
                       help="표 기반 방법의 학습 에피소드 수 (0이면 --episodes와 같음). 표는 가벼워서 길게 줘도 됨")
        p.add_argument("--methods", nargs="+", choices=LEARNING_METHODS, default=None,
                       help="compare에서 실행할 학습 방법만 선택 (예: --methods qtable qtable2). 생략하면 전부")
        p.add_argument("--out-tag", default="",
                       help="결과 파일 이름 끝에 붙일 이름 (예: final). 같은 설정을 다시 돌려도 이전 결과가 덮어쓰이지 않음")
        p.add_argument("--afh-window", type=int, default=4,
                       help="AFH 블랙리스트가 기억하는 최근 슬롯 수 (--history 이하로 잘림)")
        p.add_argument("--eval-episodes", type=int, default=100)
        p.add_argument("--episode-len", type=int, default=100)
        p.add_argument("--switch-cost", type=float, default=0.0, help="채널 전환 시 보상 페널티")
        p.add_argument("--seeds", type=int, default=3)

    p1 = sub.add_parser("compare")
    common(p1)
    p1.add_argument("--jsr", type=float, default=8.0)
    p1.set_defaults(fn=cmd_compare)

    p2 = sub.add_parser("sweep")
    common(p2)
    p2.add_argument("--jsr-levels", type=float, nargs="+", default=[-5, 0, 3, 6, 9, 12])
    p2.set_defaults(fn=cmd_sweep)

    p3 = sub.add_parser("history")
    common(p3)
    p3.add_argument("--jsr", type=float, default=-5.0)
    p3.add_argument("--histories", type=int, nargs="+", default=[1, 2, 4, 8])
    p3.set_defaults(fn=cmd_history, snr=5.0)

    p4 = sub.add_parser("delay")
    common(p4)
    p4.add_argument("--jsr", type=float, default=8.0)
    p4.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2, 3, 5])
    p4.set_defaults(fn=cmd_delay)

    p5 = sub.add_parser("baselines")
    common(p5)
    p5.add_argument("--jsr", type=float, default=8.0)
    p5.add_argument("--jammers", nargs="+", choices=JAMMER_MODELS,
                    default=["sequential", "random", "reactive", "permutation", "multi"])
    p5.add_argument("--ref-compare", default=None,
                    help="같은 조건의 compare_*.json을 주면 표 기반과 DQN 결과도 같은 그래프에 겹쳐 그림")
    p5.set_defaults(fn=cmd_baselines)

    p6 = sub.add_parser("scenario")
    common(p6)
    p6.add_argument("--jsr", type=float, default=8.0)
    p6.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    p6.add_argument("--elev", type=float, default=25.0, help="지상 단말이 위성을 보는 고각(도)")
    p6.add_argument("--slot-ms", type=float, nargs="+", default=[10.0],
                    help="도약 슬롯 길이(ms). 1개면 모든 시나리오 공통, 아니면 시나리오 수만큼")
    p6.add_argument("--feedback", choices=("rtt", "oneway"), default="rtt",
                    help="센싱 결과가 돌아오는 경로: rtt=왕복, oneway=편도")
    p6.add_argument("--proc-ms", type=float, default=0.0, help="센싱, 처리 지연(ms). deploy_bench의 한 슬롯 처리 시간을 넣을 수 있음")
    p6.set_defaults(fn=cmd_scenario)

    args = ap.parse_args()
    fn = args.fn
    del args.fn
    fn(args)


if __name__ == "__main__":
    main()
