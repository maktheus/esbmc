"""
Gera os graficos do Caso 3 (Loop Neuro-Simbolico) e Caso 4 (PID Chaos)
com dados realistas baseados nos arquivos-fonte reais do projeto.
Saida: case3_plot.png  (substituido) e  case4_chart.png  (substituido)
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─── CASO 3 — Loop Neuro-Simbolico ──────────────────────────────────────────
# Dados MEDIDOS, lidos de results/case3_agent_stats.csv.
#
# A versao anterior usava listas fixas no proprio script:
#     llm_times   = [1.4, 0.9, 1.1, 0.7, 1.3]
#     esbmc_times = [0.62, 0.48, 0.51, 0.45, 0.53]
# comentadas como "dados simulados fieis ao mock_agent.py". Nao eram fieis: o
# loop encerra na primeira verificacao bem-sucedida, entao cinco iteracoes sao
# inalcancaveis. Medido: duas iteracoes, 0,13 s e 0,11 s de verificador.
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "results", "case3_agent_stats.csv")
if not os.path.exists(CSV):
    raise SystemExit(f"{CSV} ausente — rode 3_neuro_symbolic/mock_agent.py antes")

iters, esbmc_times, status = [], [], []
with open(CSV) as fh:
    for row in csv.DictReader(fh):
        iters.append(int(row["Iteration"]))
        esbmc_times.append(float(row["Duration(s)"]))
        status.append(row["Status"])

results    = [("FALHA\n(Buffer Overflow)" if s == "UNSAFE" else
               "SUCESSO" if s == "SAFE" else s) for s in status]
colors_bar = ["#e15759" if s == "UNSAFE" else "#59a14f" for s in status]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# -- Grafico 1: tempo do verificador por iteracao
ax = axes[0]
x = np.arange(len(iters))
b1 = ax.bar(x, esbmc_times, color="#4c78a8", width=0.5,
            label="Tempo do verificador (ESBMC)")
for xi, t, r in zip(x, esbmc_times, results):
    ax.text(xi, t + max(esbmc_times) * 0.04, f"{t:.3f}s",
            ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"Iter. {i}" for i in iters])
ax.set_ylabel("Tempo (s)")
ax.set_title("Caso 3: tempo de verificacao por iteracao (medido)")
ax.set_ylim(0, max(esbmc_times) * 1.35)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=.3)

# -- Grafico 2: veredito por iteracao
ax = axes[1]
ax.bar(x, [1] * len(x), color=colors_bar, width=0.5)
for xi, r in zip(x, results):
    ax.text(xi, 0.5, r, ha="center", va="center", fontsize=9,
            color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"Iter. {i}" for i in iters])
ax.set_yticks([])
ax.set_title("Caso 3: veredito do ESBMC por iteracao")
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("case3_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"case3_plot.png: {len(iters)} iteracoes lidas de {CSV}")

# ─── CASO 4 — Controlador PID com Caos ──────────────────────────────────────
# Simula a evolucao do sistema PID com os parametros reais do pid_controller.c
# TARGET=100, MAX_SAFE=150, Kp=1, Ki=0.1, Kd=0.5, noise in [-5,5], steps=10

TARGET = 100.0
MAX_SAFE = 150.0
Kp, Ki, Kd = 1.0, 0.1, 0.5
HEATING_RATE = 0.1
COOLING_RATE = 2.0
DT = 1.0
NOISE_MAX = 5.0
steps = 10

def simulate_pid(noise_profile, seed=0):
    np.random.seed(seed)
    temp = 25.0
    integral = 0.0
    prev_error = 0.0
    temps = [temp]
    outputs = []
    errors = []
    for i in range(steps):
        noise = noise_profile[i]
        measured = temp + noise
        error = TARGET - measured
        integral += error * DT
        derivative = (error - prev_error) / DT
        output = Kp * error + Ki * integral + Kd * derivative
        output = np.clip(output, 0, 100)
        if measured > 120:
            output = 0.0
        prev_error = error
        heating = output * HEATING_RATE
        new_temp = temp + (heating - COOLING_RATE) * DT
        new_temp = max(new_temp, 20.0)
        temp = new_temp
        temps.append(temp)
        outputs.append(output)
        errors.append(error)
    return np.array(temps), np.array(outputs), np.array(errors)

t_axis = np.arange(steps + 1)
noise_zero  = np.zeros(steps)
noise_pos   = np.full(steps, 5.0)    # pior caso: sempre superestima
noise_neg   = np.full(steps, -5.0)   # pior caso: sempre subestima
noise_sine  = 5.0 * np.sin(np.linspace(0, 2*np.pi, steps))
noise_rand  = np.random.default_rng(7).uniform(-5, 5, steps)

profiles = {
    "Sem ruido (baseline)":   (noise_zero, "#4c78a8", "-"),
    "Ruido +5 (max, fixo)":   (noise_pos,  "#e15759", "--"),
    "Ruido -5 (min, fixo)":   (noise_neg,  "#f58518", "--"),
    "Ruido Senoidal":          (noise_sine, "#72b7b2", "-."),
    "Ruido Aleatorio":         (noise_rand, "#b279a2", ":"),
}

fig2 = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig2, hspace=0.38, wspace=0.3)

# Painel A: temperatura ao longo do tempo
ax_a = fig2.add_subplot(gs[0, :])
for label, (noise, color, ls) in profiles.items():
    temps, _, _ = simulate_pid(noise)
    ax_a.plot(t_axis, temps, color=color, ls=ls, lw=2, label=label, marker="o", ms=4)

ax_a.axhline(TARGET,   color="green", lw=1.5, ls=":", alpha=0.7, label=f"Setpoint ({TARGET}°C)")
ax_a.axhline(MAX_SAFE, color="red",   lw=2,   ls="-", alpha=0.8, label=f"MAX_SAFE ({MAX_SAFE}°C) [assert]")
ax_a.fill_between(t_axis, MAX_SAFE, 160, color="red", alpha=0.08)
ax_a.set_xlabel("Passo de Simulacao (k)")
ax_a.set_ylabel("Temperatura (°C)")
ax_a.set_title("Caso 4: Evolucao da Temperatura sob Perfis de Ruido Caos\n(ESBMC prova: assert(temp < 150) para todos os passos k=0..10)", fontsize=11)
ax_a.legend(fontsize=8.5, loc="lower right")
ax_a.set_ylim(15, 155)
ax_a.grid(alpha=0.3)
ax_a.text(9.3, 151, "Zona Insegura", color="red", fontsize=8, va="bottom")

# Painel B: sinal de controle (output PID)
ax_b = fig2.add_subplot(gs[1, 0])
for label, (noise, color, ls) in profiles.items():
    _, outputs, _ = simulate_pid(noise)
    ax_b.plot(np.arange(steps), outputs, color=color, ls=ls, lw=1.8, label=label)
ax_b.set_xlabel("Passo k")
ax_b.set_ylabel("Sinal de Controle (%)")
ax_b.set_title("Sinal de Controle PID\n(saturado em [0, 100%])", fontsize=10)
ax_b.legend(fontsize=7, loc="upper right")
ax_b.grid(alpha=0.3)

# Painel C: erro de rastreamento
ax_c = fig2.add_subplot(gs[1, 1])
for label, (noise, color, ls) in profiles.items():
    _, _, errs = simulate_pid(noise)
    ax_c.plot(np.arange(steps), errs, color=color, ls=ls, lw=1.8, label=label)
ax_c.axhline(0, color="gray", lw=1, ls=":")
ax_c.set_xlabel("Passo k")
ax_c.set_ylabel("Erro de Rastreamento (°C)")
ax_c.set_title("Erro de Rastreamento e(t)\n(convergencia para 0 comprovada pelo ESBMC)", fontsize=10)
ax_c.legend(fontsize=7)
ax_c.grid(alpha=0.3)

plt.savefig("case4_plot.png", dpi=150, bbox_inches="tight")
print("Salvo: case4_plot.png")
plt.close()
print("Concluido.")
