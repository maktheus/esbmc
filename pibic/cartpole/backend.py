"""
backend.py — FastAPI backend para o webapp Cart-Pole DQN.

Endpoints:
  GET /api/health               → {"status": "ok"}
  GET /api/simulate?seed=42&max_steps=500

Instalação:
    pip install fastapi uvicorn

Execução:
    uvicorn backend:app --reload --port 8000
"""

import os
import sys
import math

import torch
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dqn_agent import QNetwork
from cartpole_env import CartPoleEnv

# ── Configuração ────────────────────────────────────────────────────────────
PTH_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dqn_cartpole.pth")
MAX_STEPS_DEFAULT = 500

# ── Carrega o modelo uma vez no startup ─────────────────────────────────────
_model: QNetwork | None = None


def get_model() -> QNetwork:
    global _model
    if _model is None:
        m = QNetwork()
        state_dict = torch.load(PTH_PATH, map_location="cpu", weights_only=True)
        m.load_state_dict(state_dict)
        m.eval()
        _model = m
    return _model


# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Cart-Pole DQN API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    """Verifica se o backend está funcionando."""
    return {"status": "ok"}


@app.get("/api/simulate")
def simulate(
    seed:      int = Query(default=42,  ge=0,   le=9999, description="Semente aleatória"),
    max_steps: int = Query(default=500, ge=1,   le=2000, description="Máximo de passos"),
):
    """
    Roda um episódio completo do Cart-Pole com o DQN treinado.

    Retorna:
      seed, score, trajectory (lista de frames com estado e Q-values)
    """
    model = get_model()
    env   = CartPoleEnv(seed=seed)

    state       = env.reset()
    trajectory  = []
    total_reward = 0.0

    for _ in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor).squeeze(0)
        q0     = float(q_values[0].item())
        q1     = float(q_values[1].item())
        action = int(q_values.argmax().item())

        x, x_dot, theta, theta_dot = state
        trajectory.append({
            "x":         round(float(x),         6),
            "x_dot":     round(float(x_dot),     6),
            "theta":     round(float(theta),      6),
            "theta_dot": round(float(theta_dot),  6),
            "action":    action,
            "q0":        round(q0, 6),
            "q1":        round(q1, 6),
        })

        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

        if done:
            break

    return {
        "seed":       seed,
        "score":      int(total_reward),
        "trajectory": trajectory,
    }
