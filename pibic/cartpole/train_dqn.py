"""
train_dqn.py — Treina DQN no Cart-Pole e exporta controlador para ONNX.

Uso:
    python train_dqn.py

Saídas:
    dqn_cartpole.onnx       — controlador para extração de pesos e verificação
    dqn_cartpole.pth        — checkpoint PyTorch
    training_history.json   — curva de aprendizado (score e avg100 por episódio)
"""

import os, sys, json, torch

sys.path.insert(0, os.path.dirname(__file__))
from cartpole_env import CartPoleEnv, N_ACTIONS, FORCE_LEVELS
from dqn_agent import DQNAgent

EPISODES    = 2000
MAX_STEPS   = 500
SEED        = 42
SOLVE_SCORE = 470   # média 100 ep para considerar resolvido

torch.manual_seed(SEED)

env   = CartPoleEnv(seed=SEED)
agent = DQNAgent(lr=5e-4, gamma=0.99, batch=128, target_update=100)

print("=" * 55)
print(f"Treinamento DQN — Cart-Pole  (4 → 24 → 24 → {N_ACTIONS})")
print(f"Forças: {FORCE_LEVELS} N")
print("=" * 55)

scores   = []
history  = []  # lista de {episode, score, avg100, epsilon}

for ep in range(1, EPISODES + 1):
    state = env.reset()
    total = 0
    for _ in range(MAX_STEPS):
        action              = agent.select_action(state)
        next_state, r, done = env.step(action)
        agent.store(state, action, r, next_state, float(done))
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
        "epsilon": round(agent.epsilon, 4),
    })

    if ep % 50 == 0:
        print(f"  Ep {ep:4d} | score {total:5.0f} | avg100 {avg:6.1f} | ε {agent.epsilon:.3f}")

    if avg >= SOLVE_SCORE:
        print(f"\n✓ Resolvido em {ep} episódios (avg100 = {avg:.1f})")
        break
else:
    print(f"\nTreinamento concluído ({EPISODES} episódios). avg100 final = {avg:.1f}")

# ── Salva checkpoint ───────────────────────────────────────────────────────
out_dir = os.path.dirname(__file__)
pth = os.path.join(out_dir, "dqn_cartpole.pth")
torch.save(agent.policy.state_dict(), pth)
print(f"Salvo: {pth}")

# ── Salva histórico de treinamento ─────────────────────────────────────────
hist_path = os.path.join(out_dir, "training_history.json")
with open(hist_path, "w", encoding="utf-8") as f:
    json.dump(history, f, separators=(",", ":"))
print(f"Histórico: {hist_path}  ({len(history)} episódios)")

# ── Exporta ONNX ───────────────────────────────────────────────────────────
agent.policy.eval()
dummy = torch.zeros(1, 4)
onnx_path = os.path.join(out_dir, "dqn_cartpole.onnx")
torch.onnx.export(
    agent.policy, dummy, onnx_path,
    input_names=["state"], output_names=["q_values"],
    opset_version=11,
)
print(f"Exportado: {onnx_path}")
print("\nPróximos passos:")
print("  python generate_webapp_data.py  # regenera simulation_data.json")
print("  python verify_dead_neurons.py   # verifica neurônios mortos (ESBMC)")
print("  python verify_saturation.py     # verifica saturação (ESBMC)")
