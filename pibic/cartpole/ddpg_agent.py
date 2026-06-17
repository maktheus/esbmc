"""
ddpg_agent.py — Agente DDPG (Deep Deterministic Policy Gradient) para Cart-Pole.

Controlador contínuo:
  Actor:  state(4) → 24 → 24 → tanh → force ∈ [-10, +10] N
  Critic: (state(4) + action(1)) → 24 → 24 → Q-value

Referência: Lillicrap et al. (2015) — "Continuous control with deep RL"
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cartpole_env import FORCE_MAX

HIDDEN = 24


class ActorNetwork(nn.Module):
    """Rede de política: estado → força contínua."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1), nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state) * FORCE_MAX


class CriticNetwork(nn.Module):
    """Rede Q: (estado, ação) → Q-value."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self._buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self._buf.append((s, a, r, s2, done))

    def sample(self, n):
        batch = random.sample(self._buf, n)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s),  dtype=torch.float32),
            torch.tensor(np.array(a),  dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(r),  dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(np.array(d),  dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self._buf)


class OUNoise:
    """Ornstein-Uhlenbeck noise for exploration."""

    def __init__(self, mu=0.0, sigma=0.3, theta=0.15, dt=1e-2):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.state = mu

    def __call__(self):
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.state += dx
        return self.state

    def reset(self):
        self.state = self.mu


class DDPGAgent:
    def __init__(self, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99,
                 tau=0.005, batch=256):
        self.actor        = ActorNetwork()
        self.actor_target = ActorNetwork()
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic        = CriticNetwork()
        self.critic_target = CriticNetwork()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer(capacity=100_000)
        self.gamma  = gamma
        self.tau    = tau
        self.batch  = batch
        self.noise  = OUNoise(sigma=0.3)

    def select_action(self, state, explore=True):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            a = self.actor(s).item()
        if explore:
            a += self.noise() * FORCE_MAX
        return max(-FORCE_MAX, min(FORCE_MAX, a))

    def store(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def train_step(self):
        if len(self.buffer) < self.batch:
            return None, None

        s, a, r, s2, done = self.buffer.sample(self.batch)

        # ── Critic update ──
        with torch.no_grad():
            a2 = self.actor_target(s2)
            q_next = self.critic_target(s2, a2)
            q_target = r + self.gamma * q_next * (1 - done)

        q_pred = self.critic(s, a)
        critic_loss = nn.functional.mse_loss(q_pred, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor update ──
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── Soft update targets ──
        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return critic_loss.item(), actor_loss.item()
