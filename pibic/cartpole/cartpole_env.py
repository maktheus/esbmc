"""
cartpole_env.py — Simulação do pêndulo invertido com carro (Cart-Pole).

Modelo baseado em:
  Barto, Sutton & Anderson (1983) — equações padrão OpenAI Gym CartPole-v1
  MathWorks: https://www.mathworks.com/help/symbolic/derive-and-simulate-cart-pole-system.html

Estado: s = [x, x_dot, theta, theta_dot]
  x         posição do carro (m)    ∈ [-2.4, 2.4]
  x_dot     velocidade do carro (m/s)
  theta     ângulo do pêndulo (rad) ∈ [-12°, 12°]
  theta_dot velocidade angular (rad/s)

Ação contínua: F ∈ [-10, +10] N (força horizontal no carro)
"""

import math
import random

# ── Parâmetros físicos (MathWorks / OpenAI Gym) ────────────────────────────
GRAVITY   = 9.8          # m/s²
M_CART    = 1.0          # kg
M_POLE    = 0.1          # kg
M_TOTAL   = M_CART + M_POLE
L         = 0.5          # m  (metade do comprimento do pêndulo)
ML        = M_POLE * L
DT        = 0.02         # s  (integração de Euler)

FORCE_MAX = 10.0         # N  (limite de força do atuador)

# ── Espaço de ação discretizado (legado, usado apenas pelo DQN antigo) ────
FORCE_LEVELS = [-10.0, -5.0, 0.0, 5.0, 10.0]  # N
N_ACTIONS    = len(FORCE_LEVELS)                # 5

# ── Limites de falha ───────────────────────────────────────────────────────
X_LIMIT     = 2.4                    # m
THETA_LIMIT = 12.0 * math.pi / 180  # rad ≈ 0.2094

# ── Bounds conservadores para verificação formal ───────────────────────────
STATE_BOUNDS = {
    "x":         (-X_LIMIT,     X_LIMIT),
    "x_dot":     (-5.0,         5.0),
    "theta":     (-THETA_LIMIT, THETA_LIMIT),
    "theta_dot": (-5.0,         5.0),
}


def _physics_step(state, F):
    """Equações de movimento + integração de Euler. Retorna novo estado."""
    x, xd, th, thd = state
    cos_th = math.cos(th)
    sin_th = math.sin(th)

    temp   = (F + ML * thd ** 2 * sin_th) / M_TOTAL
    th_acc = (GRAVITY * sin_th - cos_th * temp) / \
             (L * (4.0 / 3.0 - M_POLE * cos_th ** 2 / M_TOTAL))
    x_acc  = temp - ML * th_acc * cos_th / M_TOTAL

    x   += DT * xd
    xd  += DT * x_acc
    th  += DT * thd
    thd += DT * th_acc
    return (x, xd, th, thd)


def _is_done(state):
    x, _, th, _ = state
    return abs(x) > X_LIMIT or abs(th) > THETA_LIMIT


class CartPoleEnv:
    """Ambiente Cart-Pole determinístico com integração de Euler."""

    def __init__(self, seed=None):
        self._rng = random.Random(seed)

    def reset(self):
        self.state = tuple(self._rng.uniform(-0.05, 0.05) for _ in range(4))
        self.steps = 0
        return self.state

    def step(self, action):
        """Passo com ação discreta (índice em FORCE_LEVELS)."""
        F = FORCE_LEVELS[int(action)]
        return self.step_continuous(F)

    def step_continuous(self, force: float):
        """Passo com força contínua em [-FORCE_MAX, +FORCE_MAX] N."""
        F = max(-FORCE_MAX, min(FORCE_MAX, float(force)))
        self.state = _physics_step(self.state, F)
        self.steps += 1
        done   = _is_done(self.state)
        reward = 0.0 if done else 1.0
        return self.state, reward, done
