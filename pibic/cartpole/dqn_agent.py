"""
dqn_agent.py — Agente DQN (Deep Q-Network) para Cart-Pole.

Arquitetura do controlador neural:
  input(4) → Linear(4,24) → ReLU → Linear(24,24) → ReLU → Linear(24,N_ACTIONS)

  N_ACTIONS = 5 (forças: -10, -5, 0, +5, +10 N)
  O DQN emite um Q-value por nível de força. O argmax seleciona a força aplicada.
  Isso é mais realista que a versão binária (±10 N apenas).

Referência:
  MathWorks DQN Cart-Pole:
  https://www.mathworks.com/help/reinforcement-learning/ug/train-dqn-agent-to-balance-discrete-cart-pole.html
"""

import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from cartpole_env import N_ACTIONS

HIDDEN = 24   # neurônios por camada oculta (verificação mais rápida no ESBMC)


class QNetwork(nn.Module):
    """Rede Q — 4 → 24 → 24 → N_ACTIONS."""

    def __init__(self, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self._buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self._buf.append((s, a, r, s2, done))

    def sample(self, n):
        batch = random.sample(self._buf, n)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(s,  dtype=torch.float32),
            torch.tensor(a,  dtype=torch.long),
            torch.tensor(r,  dtype=torch.float32),
            torch.tensor(s2, dtype=torch.float32),
            torch.tensor(d,  dtype=torch.float32),
        )

    def __len__(self):
        return len(self._buf)


class DQNAgent:
    def __init__(self, lr=5e-4, gamma=0.99, batch=128, target_update=100):
        self.policy = QNetwork()
        self.target = QNetwork()
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer    = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer       = ReplayBuffer(capacity=50_000)
        self.gamma        = gamma
        self.batch        = batch
        self.target_update = target_update
        self._steps       = 0

        self.epsilon  = 1.0
        self.eps_min  = 0.01
        self.eps_decay = 0.998

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return int(self.policy(s).argmax(dim=1).item())

    def store(self, *args):
        self.buffer.push(*args)

    def train_step(self):
        if len(self.buffer) < self.batch:
            return None

        s, a, r, s2, done = self.buffer.sample(self.batch)

        q_cur  = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(s2).max(1).values
        q_tgt  = r + self.gamma * q_next * (1 - done)

        loss = nn.functional.mse_loss(q_cur, q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._steps += 1
        if self._steps % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())

        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        return loss.item()
