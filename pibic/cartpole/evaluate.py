"""
evaluate.py — Avalia o controlador DQN treinado e plota a trajetória.

Uso:
    python evaluate.py

Requer dqn_cartpole.pth gerado pelo train_dqn.py.
"""

import os, sys, math, torch

sys.path.insert(0, os.path.dirname(__file__))
from cartpole_env import CartPoleEnv, THETA_LIMIT
from dqn_agent import QNetwork

N_EPISODES = 20
MAX_STEPS  = 500

model = QNetwork()
model.load_state_dict(
    torch.load(
        os.path.join(os.path.dirname(__file__), "dqn_cartpole.pth"),
        weights_only=True,
    )
)
model.eval()

env    = CartPoleEnv(seed=0)
scores = []
last_traj = []

for ep in range(N_EPISODES):
    state = env.reset()
    total = 0
    traj  = [state]
    for _ in range(MAX_STEPS):
        with torch.no_grad():
            s      = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = int(model(s).argmax(1).item())
        state, reward, done = env.step(action)
        traj.append(state)
        total += reward
        if done:
            break
    scores.append(total)
    last_traj = traj

avg = sum(scores) / len(scores)
print(f"{'='*50}")
print(f"Avaliação — {N_EPISODES} episódios")
print(f"{'='*50}")
print(f"  Score médio : {avg:.1f}")
print(f"  Score mín   : {min(scores):.0f}")
print(f"  Score máx   : {max(scores):.0f}")
print(f"  Duração últ : {len(last_traj)} passos ({len(last_traj)*0.02:.1f} s)")

# ── Plot (opcional) ────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs   = [s[0]                     for s in last_traj]
    xds  = [s[1]                     for s in last_traj]
    ths  = [s[2] * 180 / math.pi     for s in last_traj]
    thds = [s[3]                     for s in last_traj]
    ts   = [i * 0.02                 for i in range(len(last_traj))]

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle("Cart-Pole — Trajetória do Controlador DQN", fontsize=13)

    lim_deg = THETA_LIMIT * 180 / math.pi

    axs[0, 0].plot(ts, xs, "b");            axs[0, 0].set_ylabel("x (m)")
    axs[0, 0].axhline( 2.4, color="r", ls="--", lw=0.8)
    axs[0, 0].axhline(-2.4, color="r", ls="--", lw=0.8)

    axs[0, 1].plot(ts, xds, "b");           axs[0, 1].set_ylabel("ẋ (m/s)")

    axs[1, 0].plot(ts, ths, "g")
    axs[1, 0].axhline( lim_deg, color="r", ls="--", lw=0.8, label=f"±{lim_deg:.1f}°")
    axs[1, 0].axhline(-lim_deg, color="r", ls="--", lw=0.8)
    axs[1, 0].set_ylabel("θ (°)");          axs[1, 0].set_xlabel("t (s)")
    axs[1, 0].legend(fontsize=8)

    axs[1, 1].plot(ts, thds, "g");          axs[1, 1].set_ylabel("θ̇ (rad/s)")
    axs[1, 1].set_xlabel("t (s)")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "cartpole_trajectory.png")
    plt.savefig(out, dpi=150)
    print(f"\nGráfico salvo: {out}")
except ImportError:
    print("matplotlib não disponível — pule a plotagem")
