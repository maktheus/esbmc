# -*- coding: utf-8 -*-
"""
Gera a Figura 8: Certificação Formal Ponta-a-Ponta no Modelo Qwen 3.8 27B
(27.32B Parâmetros, 65 Camadas Híbridas 3:1, d = 5120) no Servidor GPU de Alto Desempenho.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8.5,
    'axes.labelsize': 9.5,
    'axes.titlesize': 10.5,
    'xtick.labelsize': 8.0,
    'ytick.labelsize': 8.0,
    'legend.fontsize': 8.0,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

JSON_PATH = r"C:\uchoa\esbmc\llm_verification\resultado_qwen38_27b_verification.json"
OUT_IMG = r"C:\uchoa\esbmc\llm_verification\assets\fig8_qwen38_27b_benchmark_e2e.png"

def plot_fig8():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    slices = data["parallel_benchmark_28cores"]["slices"]
    btor_inst = data["btor2_vs_z3_head_to_head"]["instances"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 8.5))

    # ----------------------------------------------------
    # Subplot 1: Latência Head-to-Head Btor2/QF_BV vs Z3 QF_LRA (Escala Log)
    # ----------------------------------------------------
    tokens_b = [b["token"] for b in btor_inst]
    t_bv_safe = [b["z3_bv_safe_ms"] for b in btor_inst]
    t_bv_sat = [b["z3_bv_sat_ms"] for b in btor_inst]
    t_lra_safe = [b["z3_lra_safe_ms"] for b in btor_inst]
    t_lra_sat = [b["z3_lra_sat_ms"] for b in btor_inst]

    x = np.arange(len(tokens_b))
    w = 0.2

    ax1.bar(x - 1.5*w, t_bv_safe,  w, label='Btor2 / QF_BV (Safe/UNSAT: 1.160,8 ms)', color='#b91c1c', alpha=0.9, edgecolor='black', linewidth=0.5)
    ax1.bar(x - 0.5*w, t_bv_sat,   w, label='Btor2 / QF_BV (Adv/SAT: 2.145,9 ms)',    color='#ea580c', alpha=0.9, edgecolor='black', linewidth=0.5)
    ax1.bar(x + 0.5*w, t_lra_safe, w, label='Z3 QF_LRA (Safe/UNSAT: 10,85 ms)',       color='#1e40af', alpha=0.9, edgecolor='black', linewidth=0.5)
    ax1.bar(x + 1.5*w, t_lra_sat,  w, label='Z3 QF_LRA (Adv/SAT: 8,29 ms)',           color='#3b82f6', alpha=0.9, edgecolor='black', linewidth=0.5)

    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tokens_b, rotation=35, ha='right', fontweight='bold')
    ax1.set_ylabel('Tempo de Resolução (milissegundos, log)')
    ax1.set_title('(A) Latência Head-to-Head: Btor2 (Bit-Vector Q16.16) vs. Z3 QF_LRA (d = 5120)')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=7.2)

    # ----------------------------------------------------
    # Subplot 2: Raios Críticos Certificados de Entrada eps_in* (28 Classes)
    # ----------------------------------------------------
    tok_names_28 = [s["token"].replace("_fallback", "") for s in slices]
    eps_in = [s["eps_critical_input"] * 1e4 for s in slices] # em 10^-4

    x28 = np.arange(len(slices))
    colors_bar = ['#047857' if s['verdict_e2e'] == 'UNSAT' else '#b91c1c' for s in slices]

    ax2.bar(x28, eps_in, color=colors_bar, edgecolor='black', linewidth=0.5, alpha=0.85)
    ax2.axhline(np.mean(eps_in), color='#b91c1c', linestyle='--', linewidth=1.2, label=f'Média $\\bar{{\\epsilon}}^* = {np.mean(eps_in):.2f} \\times 10^{{-4}}$')
    ax2.set_xticks(x28)
    ax2.set_xticklabels(tok_names_28, rotation=70, ha='right', fontsize=6.5)
    ax2.set_ylabel('Raio Crítico Certificado $\\epsilon_{\\mathrm{in}}^* \\,(\\times 10^{-4})$')
    ax2.set_title('(B) Raio Crítico Certificado de Entrada para 28 Classes de Ameaça (100% UNSAT)')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')

    # ----------------------------------------------------
    # Subplot 3: Comparativo de Escala de Modelos (7B, 9B, 14B e 27B)
    # ----------------------------------------------------
    models_labels = ['Qwen2.5-7B\n(28 camadas)', 'Nimble-9B\n(32 camadas)', 'Qwen3-14B\n(40 camadas)', 'Qwen3.8-27B\n(65 camadas)']
    params_b = [7.61, 8.98, 14.77, 27.32]
    expansions = [12.50, 8.23, 14.82, 19.95]

    xm = np.arange(len(models_labels))
    ax3_twin = ax3.twinx()

    ax3.bar(xm - 0.15, params_b, width=0.3, label='Parâmetros (Bilhões)', color='#4338ca', alpha=0.85, edgecolor='black')
    ax3_twin.plot(xm + 0.15, expansions, marker='o', color='#b45309', linewidth=2, label='Fator Lipschitz $\\mathcal{G}_{\\mathrm{total}}$ (x)', markersize=7)

    for i, v in enumerate(params_b):
        ax3.text(i - 0.15, v + 0.6, f'{v:.2f}B', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    for i, v in enumerate(expansions):
        ax3_twin.text(i + 0.15, v + 0.6, f'{v:.2f}x', ha='center', va='bottom', fontsize=7.5, color='#b45309', fontweight='bold')

    ax3.set_xticks(xm)
    ax3.set_xticklabels(models_labels, fontweight='bold')
    ax3.set_ylabel('Parâmetros Totais (Bilhões)')
    ax3_twin.set_ylabel('Fator Lipschitz Acumulado $\\mathcal{G}_{\\mathrm{total}}$')
    ax3.set_title('(C) Escalabilidade Formal: Evolução 7,6B -> 9,0B -> 14,8B -> 27,3B')
    ax3.grid(axis='y', linestyle='--', alpha=0.4)
    ax3.set_ylim(0, 32)
    ax3_twin.set_ylim(0, 24)

    # ----------------------------------------------------
    # Subplot 4: Matriz de Descobertas Formais e Métricas Consolidadas
    # ----------------------------------------------------
    ax4.axis('off')
    table_data = [
        ["Métrica de Engenharia Formal", "Nimble-9B (Q8_0)", "Qwen3-14B", "Qwen 3.8 27B (Híbrido)"],
        ["Parâmetros do Modelo", "8,98 Bilhões", "14,77 Bilhões", "27,32 Bilhões (+85%)"],
        ["Camadas Transformer / SSM", "32 Blocos", "40 Blocos", "65 Blocos (49 Attn + 16 SSM)"],
        ["Dimensão Latente (d)", "4.096", "5.120", "5.120"],
        ["Vocabulário Auditado (|V|)", "248.320 tokens", "151.936 tokens", "248.320 tokens (100%)"],
        ["Expansão Lipschitz (G_total)", "8,2338x", "14,8210x", "19,9533x"],
        ["Raio Crítico Médio (eps_in*)", "1,67 x 10^-3", "8,38 x 10^-4", "6,53 x 10^-4"],
        ["Paridade Btor2 vs Z3 SMT", "100% UNSAT", "100% UNSAT / SAT", "100% UNSAT / SAT"],
        ["Latência Z3 QF_LRA (Safe)", "6,80 ms", "9,30 ms", "10,85 ms"],
        ["Tempo Benchmark 28 Threads", "1,92 s", "1,98 s", "2,16 s (Wall-Clock)"],
        ["Infraestrutura de Execução", "Estação Local", "Estação Local", "Servidor HPC GPU (RTX 2000)"]
    ]

    t = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(7.2)
    t.scale(1.0, 1.35)

    for (row, col), cell in t.get_celld().items():
        if row == 0:
            cell.set_facecolor('#0f2d69')
            cell.set_text_props(color='white', weight='bold')
        elif col == 3:
            cell.set_facecolor('#ecfdf5')
            cell.set_text_props(weight='bold')
        elif row % 2 == 1:
            cell.set_facecolor('#f8fafc')

    ax4.set_title('(D) Matriz Consolidada da Família de Modelos Verificados', pad=12)

    plt.tight_layout()
    plt.savefig(OUT_IMG)
    plt.close()
    print(f"[OK] Gráfico consolidado salvo em {OUT_IMG}")

if __name__ == "__main__":
    plot_fig8()
