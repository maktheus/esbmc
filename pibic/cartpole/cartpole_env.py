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

Ação (discreta, 5 níveis):
  0 = -10 N  (empurrar totalmente à esquerda)
  1 =  -5 N  (empurrar levemente à esquerda)
  2 =   0 N  (sem força)
  3 =  +5 N  (empurrar levemente à direita)
  4 = +10 N  (empurrar totalmente à direita)

  Antes era binário (±10 N). Agora o controlador tem modulação de força,
  o que é mais realista: um motor real não opera apenas em liga/desliga.
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

# ── Espaço de ação discretizado (5 níveis de força) ───────────────────────
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


class CartPoleEnv:
    """Ambiente Cart-Pole determinístico com integração de Euler."""

    def __init__(self, seed=None):
        self._rng = random.Random(seed)

    def reset(self):
        self.state = tuple(self._rng.uniform(-0.05, 0.05) for _ in range(4))
        self.steps = 0
        return self.state

    def step(self, action):
        """Avança um passo de tempo. Retorna (next_state, reward, done)."""
        x, xd, th, thd = self.state
        F = FORCE_LEVELS[int(action)]

        cos_th = math.cos(th)
        sin_th = math.sin(th)

        # Equações de movimento (Barto et al., 1983)
        temp   = (F + ML * thd ** 2 * sin_th) / M_TOTAL
        th_acc = (GRAVITY * sin_th - cos_th * temp) / \
                 (L * (4.0 / 3.0 - M_POLE * cos_th ** 2 / M_TOTAL))
        x_acc  = temp - ML * th_acc * cos_th / M_TOTAL

        # Integração de Euler
        x   += DT * xd
        xd  += DT * x_acc
        th  += DT * thd
        thd += DT * th_acc

        self.state = (x, xd, th, thd)
        self.steps += 1

        done   = abs(x) > X_LIMIT or abs(th) > THETA_LIMIT
        reward = 0.0 if done else 1.0
        return self.state, reward, done
