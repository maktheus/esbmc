"""
train_ddpg.py — Treina controlador contínuo DDPG para Cart-Pole.

Saídas:
    ddpg_actor.pth            — pesos do actor (controlador)
    ddpg_actor_weights.json   — pesos extraídos para inferência no browser
    training_history.json     — curva de aprendizado
"""

import os, sys, json, torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cartpole_env import CartPoleEnv, FORCE_MAX
from ddpg_agent import DDPGAgent

EPISODES    = 2000
MAX_STEPS   = 500
SEED        = 42
SOLVE_SCORE = 470

torch.manual_seed(SEED)
np.random.seed(SEED)

env   = CartPoleEnv(seed=SEED)
agent = DDPGAgent(lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, tau=0.005, batch=256)

print("=" * 55)
print(f"Treinamento DDPG — Cart-Pole contínuo")
print(f"Actor: 4 → 24 → 24 → tanh × {FORCE_MAX}")
print("=" * 55)

scores   = []
history  = []
best_avg = 0.0
out_dir  = os.path.dirname(__file__)

for ep in range(1, EPISODES + 1):
    state = env.reset()
    agent.noise.reset()
    total = 0.0

    for _ in range(MAX_STEPS):
        force = agent.select_action(state, explore=True)
        next_state, r, done = env.step_continuous(force)
        agent.store(state, force, r, next_state, float(done))
        agent.train_step()
        state  = next_state
        total += r
        if done:
            break

    scores.append(total)
    avg = sum(scores[-100:]) / min(len(scores), 100)
    history.append({
        "episode": ep,
        "score":   round(total, 1),
        "avg100":  round(avg, 2),
    })

    if avg > best_avg:
        best_avg = avg
        torch.save(agent.actor.state_dict(),
                    os.path.join(out_dir, "ddpg_actor_best.pth"))

    if ep % 50 == 0:
        print(f"  Ep {ep:4d} | score {total:5.0f} | avg100 {avg:6.1f} | best {best_avg:.1f}")

    if avg >= SOLVE_SCORE:
        print(f"\n✓ Resolvido em {ep} episódios (avg100 = {avg:.1f})")
        break
else:
    print(f"\nTreinamento concluído ({EPISODES} ep). avg100 final = {avg:.1f}")

# ── Salva checkpoint final ─────────────────────────────────────────────────
pth = os.path.join(out_dir, "ddpg_actor.pth")
torch.save(agent.actor.state_dict(), pth)
print(f"Salvo: {pth}")

# ── Salva histórico ────────────────────────────────────────────────────────
hist_path = os.path.join(out_dir, "training_history.json")
with open(hist_path, "w", encoding="utf-8") as f:
    json.dump(history, f, separators=(",", ":"))
print(f"Histórico: {hist_path}  ({len(history)} episódios)")

# ── Extrai pesos para JSON (inferência no browser) ─────────────────────────
best_pth = os.path.join(out_dir, "ddpg_actor_best.pth")
if os.path.exists(best_pth):
    sd = torch.load(best_pth, map_location="cpu", weights_only=True)
    print(f"Usando melhor checkpoint (avg100 = {best_avg:.1f})")
else:
    sd = agent.actor.state_dict()

weights = {}
for key, tensor in sd.items():
    weights[key] = tensor.tolist()

wjson = os.path.join(out_dir, "webapp", "public", "ddpg_weights.json")
os.makedirs(os.path.dirname(wjson), exist_ok=True)
with open(wjson, "w") as f:
    json.dump(weights, f, separators=(",", ":"))
print(f"Pesos browser: {wjson}")
print(f"  Força máxima: ±{FORCE_MAX} N")
