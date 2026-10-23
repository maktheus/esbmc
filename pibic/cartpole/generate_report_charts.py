#!/usr/bin/env python3
"""
generate_report_charts.py — Gera os graficos academicos de alta resolucao (300 DPI)
para o Relatorio de Mestrado (5 Paginas).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "relatorio_assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuracao de estilo academico limpo
plt.rcParams['font.sans-serif'] = 'Segoe UI, Arial, Helvetica, sans-serif'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 0.8

# ─── 1. GRAFICO DO CICLO LIMITE E HOMEOSTASE (ESPACO DE FASE) ────────────────
def generate_limit_cycle_chart():
    # Carrega simulador
    with open(os.path.join(HERE, "webapp", "public", "ddpg_weights_q88.json"), "r") as f:
        W = json.load(f)

    def td(v): return int(v / 256)
    def relu(v): return v if v > 0 else 0

    def step(x, xd, th, thd):
        h1 = [relu(td(r[0]*x) + td(r[1]*xd) + td(r[2]*th) + td(r[3]*thd) + b) for r, b in zip(W['w1'], W['b1'])]
        h2 = [relu(sum((td(w * h) for w, h in zip(ws, h1))) + b) for ws, b in zip(W['w2'], W['b2'])]
        z = sum((td(w * h) for w, h in zip(W['w_out'][0], h2))) + W['b_out'][0]
        za = abs(z)
        if za <= 64: ta = td(za * 252)
        elif za <= 192: ta = 62 + td((za - 64) * 200)
        elif za <= 384: ta = 162 + td((za - 192) * 92)
        elif za <= 768: ta = 231 + td((za - 384) * 16)
        else: ta = 255
        tz = ta if z >= 0 else -ta
        FQ = tz * 10
        tacc = td(4040 * th - 375 * FQ)
        return x + td(5 * xd), xd + td(5 * FQ), th + td(5 * thd), thd + td(5 * tacc)

    # Roda trajetoria nominal
    x, xd, th, thd = 0, 0, 0, 0
    th_hist, thd_hist = [], []
    x_hist, xd_hist = [], []
    for _ in range(3000):
        th_hist.append(th / 256 * 180 / np.pi)
        thd_hist.append(thd / 256 * 180 / np.pi)
        x_hist.append(x / 256)
        xd_hist.append(xd / 256)
        x, xd, th, thd = step(x, xd, th, thd)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

    # Subplot 1: Espaco de Fase do Pendulo (Theta vs Theta_dot)
    ax1.plot(th_hist[:397], thd_hist[:397], color='#94a3b8', linestyle=':', label='Transiente (k < 397)', alpha=0.7)
    ax1.plot(th_hist[397:], thd_hist[397:], color='#2563eb', linewidth=1.5, label='Atrator Periodico (Homeostase)')
    ax1.axvline(-12.0, color='#dc2626', linestyle='--', linewidth=1, label='Limite Seguro (±12°)')
    ax1.axvline(12.0, color='#dc2626', linestyle='--', linewidth=1)
    ax1.set_title('Espaço de Fase do Pêndulo (θ vs θ̇)', fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel('Ângulo θ (graus)', fontsize=9)
    ax1.set_ylabel('Velocidade Angular θ̇ (graus/s)', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize=8, loc='upper right')

    # Subplot 2: Espaco de Fase do Carrinho (x vs x_dot)
    ax2.plot(x_hist[:397], xd_hist[:397], color='#94a3b8', linestyle=':', label='Transiente (k < 397)', alpha=0.7)
    ax2.plot(x_hist[397:], xd_hist[397:], color='#059669', linewidth=1.5, label='Atrator Periódico (x ∈ [-2.32m, 2.07m])')
    ax2.axvline(-2.4, color='#dc2626', linestyle='--', linewidth=1, label='Limite dos Trilhos (±2.4m)')
    ax2.axvline(2.4, color='#dc2626', linestyle='--', linewidth=1)
    ax2.set_title('Espaço de Fase do Carrinho (x vs ẋ)', fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel('Posição x (metros)', fontsize=9)
    ax2.set_ylabel('Velocidade ẋ (m/s)', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig1_ciclo_limite_fase.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Salvo: {out_path}")

# ─── 2. GRAFICO DE ESCALABILIDADE BMC (HORIZONTE K) ───────────────────────────
def generate_bmc_scalability_chart():
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=300)

    # Dados reais medidos
    k_btor = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    t_btor = [0.01, 0.01, 0.01, 0.01, 302.8, 0.01, 0.01, 0.01, 0.01, 1322.0, 3123.0]

    k_esbmc = [1, 2, 3, 4]
    t_esbmc = [0.3, 77.0, 300.0, 909.0]

    k_abc = [1, 2, 3, 4, 5]
    t_abc = [0.05, 0.2, 0.8, 3.5, 901.0]

    ax.plot(k_btor, t_btor, marker='s', color='#2563eb', linewidth=2, label='BtorMC (Word-Level BTOR2)')
    ax.plot(k_esbmc, t_esbmc, marker='o', color='#dc2626', linewidth=2, linestyle='--', label='ESBMC (C AST unrolling)')
    ax.plot(k_abc, t_abc, marker='^', color='#d97706', linewidth=2, linestyle='-.', label='ABC bmc3 (Bit-Level AIGER)')

    ax.scatter([5], [1000], marker='x', color='#dc2626', s=80, zorder=5, label='ESBMC Timeout (K≥5)')
    ax.scatter([6], [1000], marker='x', color='#d97706', s=80, zorder=5, label='ABC Timeout (Frame≥6)')

    ax.set_yscale('log')
    ax.set_title('Escalabilidade Temporal do Bounded Model Checking (BMC)', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Horizonte de Passos (K)', fontsize=9)
    ax.set_ylabel('Tempo de Resolução (segundos, log)', fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig2_bmc_escalabilidade.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Salvo: {out_path}")

# ─── 3. GRAFICO DO BENCHMARK FATIA A FATIA (Z3 VS ESBMC) ─────────────────────
def generate_slicing_comparison_chart():
    json_path = os.path.join(HERE, "comparativo_z3_vs_esbmc_28cores.json")
    if not os.path.exists(json_path):
        return
    with open(json_path, "r") as f:
        data = json.load(f)

    slices = [s["slice"].replace("Slice_", "F") for s in data["slices"]]
    t_esbmc = [s["esbmc_time"] for s in data["slices"]]
    t_z3 = [s["z3_time"] for s in data["slices"]]

    fig, ax = plt.subplots(figsize=(11, 3.8), dpi=300)
    x = np.arange(len(slices))
    width = 0.35

    ax.bar(x - width/2, t_esbmc, width, label='ESBMC (C AST)', color='#93c5fd', edgecolor='#2563eb')
    ax.bar(x + width/2, t_z3, width, label='Z3 (Bit-Vectors QF_BV)', color='#86efac', edgecolor='#059669')

    ax.axhline(90.0, color='#dc2626', linestyle='--', linewidth=1, label='Timeout (90s)')
    ax.set_title('Benchmark 28 Cores: Tempo por Fatia Angular (Domain Slicing)', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Fatia Angular (F01 a F28)', fontsize=9)
    ax.set_ylabel('Tempo (segundos)', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(slices, fontsize=7.5)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.legend(fontsize=8, loc='upper center')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig3_z3_vs_esbmc_barras.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Salvo: {out_path}")

if __name__ == "__main__":
    generate_limit_cycle_chart()
    generate_bmc_scalability_chart()
    generate_slicing_comparison_chart()
