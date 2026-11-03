"""채널 도약 항재밍 환경.

시간 흐름(슬롯 t):
  1) 에이전트는 '직전 슬롯의 센싱 결과'(상태)를 보고 채널 a_t를 고른다.
  2) 재머가 이동한다 (재머 모델에 따라).
  3) 재머가 공격 중인 채널과 a_t가 같으면 피격(hit), 보상 -1. 아니면 +1.
  4) 새 재머 위치를 센싱한 결과가 다음 상태가 된다 (관측 지연이 있으면 d슬롯 늦게 도착).
     - oracle: 재머가 있는 채널을 정확히 아는 (multi-)one-hot (성능 상한)
     - cnn   : 채널마다 스펙트로그램 -> 학습된 CNN -> 재밍 확률 p_jam (오탐/미탐 포함)
         obs_quant="soft"  : p_jam 값을 그대로 관측으로 사용
         obs_quant="binary": p_jam > 0.5 로 이진 판정한 결과만 관측으로 사용 (판정 오류가 그대로 전달됨)

재머 모델:
  sequential : 확률 p_follow로 다음 채널로 이동, 아니면 무작위 도약
  random     : 매 슬롯 무작위 채널
  reactive   : 에이전트가 직전에 쓴 채널을 확률 p_react로 추적
  permutation: 고정된 임의 순환 순서로 이동 (순서는 pattern_seed로 결정)
  multi      : n_jammers개의 독립 순차형 재머가 동시에 서로 다른 위치를 공격 (다중/부분대역 재밍)

정답(재머 위치)은 hit/보상 계산에만 쓰이고, cnn 모드에서는 상태로 새지 않는다.
"""
from collections import deque

import numpy as np

JAMMER_MODELS = ("sequential", "random", "reactive", "permutation", "multi")


class SensingBank:
    """CNN의 채널별 재밍 확률 p_jam = 1 - P(none) 을 미리 계산해 둔 은행.

    CNN은 학습 후 고정(결정적)이므로, 조건(SNR, JSR)별로 스펙트로그램을 한 번만 만들어
    CNN에 통과시켜 두고 환경에서는 여기서 샘플링한다. 에피소드마다 신호를 합성하고
    CNN을 돌리는 것과 통계적으로 동일하면서 수백 배 빠르다.
    """

    def __init__(self, p_clean, p_jammed):
        self.p_clean = np.asarray(p_clean, dtype=np.float32)     # 재밍 없는 채널의 p_jam
        self.p_jammed = np.asarray(p_jammed, dtype=np.float32)   # 재밍 받는 채널의 p_jam

    @property
    def fpr(self):
        return float((self.p_clean > 0.5).mean())

    @property
    def tpr(self):
        return float((self.p_jammed > 0.5).mean())

    @property
    def detect_acc(self):
        """임계값 0.5로 이진 판정했을 때의 균형 정확도."""
        return 0.5 * (self.tpr + 1 - self.fpr)

    def localization_acc(self, num_channels=8, n=20000, seed=0):
        """재머가 1개일 때, 재밍 채널의 p_jam이 나머지 (C-1)개 정상 채널보다 가장 높은 비율.

        이진 판정(임계 0.5) 정확도와 달리 확률값의 '순위' 정보를 본다. 동률이면 무작위로 고른 셈 친다.
        """
        rng = np.random.default_rng(seed)
        pj = rng.choice(self.p_jammed, n)
        pc = rng.choice(self.p_clean, (n, num_channels - 1))
        m = pc.max(1)
        n_eq = (pc == pj[:, None]).sum(1)
        win = np.where(pj > m, 1.0, np.where(pj == m, 1.0 / (n_eq + 1), 0.0))
        return float(win.mean())

    @classmethod
    def build(cls, cnn_path, snr_db, jsr_db, n_per_class=500, seed=0, device=None):
        from signal_gen import generate_sample
        from train_cnn import get_device, load_cnn, predict_proba

        device = device or get_device()
        model, classes = load_cnn(cnn_path, device)
        assert classes[0] == "none"
        rng = np.random.default_rng(seed)
        imgs, is_jam = [], []
        for name in classes:
            for _ in range(n_per_class):
                img, _ = generate_sample(name, snr_db, jsr_db, seed=rng)
                imgs.append(img)
                is_jam.append(name != "none")
        x = np.stack(imgs)[:, None, :, :]
        p_jam = 1.0 - predict_proba(model, x, device)[:, 0]
        is_jam = np.array(is_jam)
        return cls(p_jam[~is_jam], p_jam[is_jam])


class AntiJammingEnv:
    def __init__(self, num_channels=8, jammer="sequential", obs_mode="oracle", bank=None,
                 history=4, episode_len=100, switch_cost=0.0,
                 p_follow=0.8, p_react=0.7, seed=None, pattern_seed=0, obs_delay=0,
                 n_jammers=3, obs_quant="soft"):
        assert jammer in JAMMER_MODELS, f"jammer must be one of {JAMMER_MODELS}"
        assert obs_mode in ("oracle", "cnn")
        assert obs_quant in ("soft", "binary")
        if obs_mode == "cnn":
            assert bank is not None, "cnn 모드는 SensingBank가 필요합니다"
        if jammer == "multi":
            assert 1 <= n_jammers < num_channels, "n_jammers는 채널 수보다 작아야 합니다"
        self.C = num_channels
        self.jammer_model = jammer
        self.obs_mode = obs_mode
        self.obs_quant = obs_quant
        self.bank = bank
        self.k = history
        self.episode_len = episode_len
        self.switch_cost = switch_cost
        self.p_follow = p_follow        # sequential/permutation/multi: 규칙대로 이동할 확률
        self.p_react = p_react          # reactive: 에이전트의 직전 채널로 이동할 확률
        self.n_jam = n_jammers          # multi: 동시에 공격하는 재머 수
        self.delay = obs_delay          # 센싱 결과가 늦게 도착하는 슬롯 수 (위성 전파 지연 모의)
        self.rng = np.random.default_rng(seed)

        # permutation 재머: pattern_seed로 정해진 고정 순환 순서로 채널을 순회
        # (학습/평가 환경이 같은 pattern_seed를 쓰면 같은 패턴이다)
        order = np.random.default_rng(pattern_seed).permutation(self.C)
        self.successor = np.empty(self.C, dtype=int)
        self.successor[order] = np.roll(order, -1)

        self.state_dim = self.k * self.C + self.C    # 관측 k스텝 + 내 직전 채널(one-hot)
        self.action_dim = self.C

    # ---------------- 센싱 / 상태 ----------------
    def jam_set(self):
        """현재 공격받는 채널들 (정답. hit 판정과 센싱 시뮬레이션에만 사용)."""
        if self.jammer_model == "multi":
            return self.jam_pos
        return np.array([self.jammer])

    def _sense(self):
        jam = self.jam_set()
        if self.obs_mode == "oracle":
            obs = np.zeros(self.C, dtype=np.float32)
            obs[jam] = 1.0
            return obs
        obs = self.rng.choice(self.bank.p_clean, size=self.C).astype(np.float32)
        obs[jam] = self.rng.choice(self.bank.p_jammed, size=len(jam))
        if self.obs_quant == "binary":
            obs = (obs > 0.5).astype(np.float32)
        return obs

    def _push_obs(self):
        """현재 슬롯의 센싱 결과를 만들고, 지연이 있으면 d슬롯 뒤에야 이력(hist)에 들어가게 한다.
        지연 d일 때 에이전트가 보는 가장 최근 관측은 d슬롯 전 재머 위치 정보다."""
        obs = self._sense()
        if self.delay > 0:
            self.pipe.append(obs)
            obs = self.pipe.popleft()
        self.hist.append(obs)

    def _state(self):
        last = np.zeros(self.C, dtype=np.float32)
        if self.last_action >= 0:
            last[self.last_action] = 1.0
        return np.concatenate([np.concatenate(self.hist), last])

    def obs_at(self, state, j=0):
        """상태에서 j번째로 최근인 관측 블록 (j=0이 가장 최근)."""
        b = self.k - 1 - j
        return state[b * self.C:(b + 1) * self.C]

    def newest_obs(self, state):
        return self.obs_at(state, 0)

    def last_action_of(self, state):
        tail = state[self.k * self.C:]
        return int(tail.argmax()) if tail.any() else -1

    # ---------------- 재머 ----------------
    def _move_jammer(self, prev_action):
        m = self.jammer_model
        if m == "sequential":
            if self.rng.random() < self.p_follow:
                self.jammer = (self.jammer + 1) % self.C
            else:
                self.jammer = int(self.rng.integers(self.C))
        elif m == "random":
            self.jammer = int(self.rng.integers(self.C))
        elif m == "reactive":
            if prev_action >= 0 and self.rng.random() < self.p_react:
                self.jammer = prev_action       # 에이전트가 직전에 쓴 채널을 추적
            else:
                self.jammer = int(self.rng.integers(self.C))
        elif m == "permutation":
            if self.rng.random() < self.p_follow:
                self.jammer = int(self.successor[self.jammer])
            else:
                self.jammer = int(self.rng.integers(self.C))
        elif m == "multi":
            follow = self.rng.random(self.n_jam) < self.p_follow
            jump = self.rng.integers(self.C, size=self.n_jam)
            self.jam_pos = np.where(follow, (self.jam_pos + 1) % self.C, jump)
            self.jammer = int(self.jam_pos[0])

    # ---------------- Gym 스타일 API ----------------
    def reset(self):
        self.time_step = 0
        self.jammer = int(self.rng.integers(self.C))
        if self.jammer_model == "multi":
            self.jam_pos = self.rng.integers(self.C, size=self.n_jam)
            self.jammer = int(self.jam_pos[0])
        self.last_action = -1
        self.hist = deque([np.zeros(self.C, dtype=np.float32) for _ in range(self.k)], maxlen=self.k)
        self.pipe = deque([np.zeros(self.C, dtype=np.float32) for _ in range(self.delay)])
        self._push_obs()
        return self._state()

    def step(self, action):
        self.time_step += 1
        prev = self.last_action
        self._move_jammer(prev)
        hit = bool((self.jam_set() == action).any())
        reward = -1.0 if hit else 1.0
        if prev >= 0 and action != prev:
            reward -= self.switch_cost
        self.last_action = int(action)
        self._push_obs()
        done = self.time_step >= self.episode_len
        return self._state(), reward, done, hit
