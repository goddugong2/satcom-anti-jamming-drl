"""DQN(Double DQN) 에이전트와 표 기반 Q-learning 에이전트."""
import random

import numpy as np
import torch
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, capacity, state_dim, seed=None):
        self.cap = capacity
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.ns = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, ns, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.ns[i], self.d[i] = s, a, r, ns, d
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch):
        idx = self.rng.integers(0, self.size, size=batch)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]).unsqueeze(1),
                torch.from_numpy(self.r[idx]).unsqueeze(1), torch.from_numpy(self.ns[idx]),
                torch.from_numpy(self.d[idx]).unsqueeze(1))

    def __len__(self):
        return self.size


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class _EpsilonMixin:
    """스텝 수에 따라 epsilon을 선형 감소 (eps_start -> eps_end)."""

    def epsilon(self):
        frac = min(1.0, self.steps / max(1, self.eps_decay_steps))
        return self.eps_start + (self.eps_end - self.eps_start) * frac


class DQNAgent(_EpsilonMixin):
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.9, batch_size=64,
                 buffer_size=20000, warmup=500, target_update=200, hidden=128,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=10000, seed=None):
        self.action_dim = action_dim
        self.gamma, self.batch_size = gamma, batch_size
        self.warmup, self.target_update = warmup, target_update
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.steps = 0          # 행동 선택 횟수 (epsilon 스케줄용)
        self.train_steps = 0    # 학습 업데이트 횟수 (타깃 동기화용)

        self.q_net = QNetwork(state_dim, action_dim, hidden)
        self.target_net = QNetwork(state_dim, action_dim, hidden)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, state_dim, seed)
        self.rng = random.Random(seed)

    def select_action(self, state, greedy=False):
        if not greedy:
            eps = self.epsilon()
            self.steps += 1
            if self.rng.random() < eps:
                return self.rng.randrange(self.action_dim)
        with torch.no_grad():
            q = self.q_net(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(1).item())

    def remember(self, s, a, r, ns, done):
        self.memory.add(s, a, r, ns, done)

    def train_step(self):
        if len(self.memory) < max(self.batch_size, self.warmup):
            return None
        s, a, r, ns, d = self.memory.sample(self.batch_size)
        q = self.q_net(s).gather(1, a)
        with torch.no_grad():   # Double DQN: 행동은 q_net, 평가는 target_net
            next_a = self.q_net(ns).argmax(1, keepdim=True)
            q_next = self.target_net(ns).gather(1, next_a)
            target = r + self.gamma * (1.0 - d) * q_next
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)


class QLearningAgent(_EpsilonMixin):
    """표 기반 Q-learning. 상태는 정수 인덱스."""

    def __init__(self, n_states, n_actions, alpha=0.2, gamma=0.9,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=10000, seed=None):
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.steps = 0
        self.rng = np.random.default_rng(seed)

    def select_action(self, s, greedy=False):
        if not greedy:
            eps = self.epsilon()
            self.steps += 1
            if self.rng.random() < eps:
                return int(self.rng.integers(self.q.shape[1]))
        row = self.q[s]
        return int(self.rng.choice(np.flatnonzero(row == row.max())))   # 동률이면 무작위

    def update(self, s, a, r, ns):
        self.q[s, a] += self.alpha * (r + self.gamma * self.q[ns].max() - self.q[s, a])
