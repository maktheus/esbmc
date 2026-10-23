# -*- coding: utf-8 -*-
"""
Gera a Figura 6: Comparativo Formal entre Btor2 (Word-Level Bit-Vector) e Z3 SMT
no Maior Modelo Disponivel: Qwen3-14B (14.8B Parametros, d = 5120).
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

JSON_PATH = r"C:\uchoa\esbmc\llm_verification\btor_vs_z3_14b_comparison.json"
OUT_IMG = r"C:\uchoa\esbmc\llm_verification\assets\fig6_btor_vs_z3_14b_benchmark.png"

def plot_fig6():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    comp = data["comparison"]
    sc = data["dimensional_scalability"]
    
    tokens = [c["token"] for c in comp]
    eps_stars = [c["eps_critical"] for c in comp]
    
    # Latencias
    t_bv_safe = [c["z3_qf_bv"]["time_safe_ms"] for c in comp]
    t_bv_sat  = [c["z3_qf_bv"]["time_sat_ms"] for c in comp]
    t_lra_safe = [c["z3_qf_lra"]["time_safe_ms"] for c in comp]
    t_lra_sat  = [c["z3_qf_lra"]["time_sat_ms"] for c in comp]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 8.5))
    
    # ----------------------------------------------------
    # Subplot 1: Latencia Head-to-Head por Token (Escala Logaritmica)
    # ----------------------------------------------------
    x = np.arange(len(tokens))
    w = 0.2
    
    b1 = ax1.bar(x - 1.5*w, t_bv_safe,  w, label='Btor2 / QF_BV (Safe/UNSAT)', color='#b91c1c', alpha=0.9, edgecolor='black', linewidth=0.5)
    b2 = ax1.bar(x - 0.5*w, t_bv_sat,   w, label='Btor2 / QF_BV (Adv/SAT)',    color='#ea580c', alpha=0.9, edgecolor='black', linewidth=0.5)
    b3 = ax1.bar(x + 0.5*w, t_lra_safe, w, label='Z3 QF_LRA (Safe/UNSAT)',     color='#1e40af', alpha=0.9, edgecolor='black', linewidth=0.5)
    b4 = ax1.bar(x + 1.5*w, t_lra_sat,  w, label='Z3 QF_LRA (Adv/SAT)',      color='#3b82f6', alpha=0.9, edgecolor='black', linewidth=0.5)
    
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tokens, rotation=35, ha='right', fontweight='bold')
    ax1.set_ylabel('Latência de Verificação (ms - log)')
    ax1.set_title('(A) Latência Head-to-Head: Btor2 (Word-Level BV) vs Z3 (Real Simplex) [d=5120]', fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', alpha=0.4, axis='y')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(0.5, 30000)
    
    # Anotacao de speedup
    avg_speedup = np.mean(t_bv_safe) / np.mean(t_lra_safe)
    ax1.text(0.03, 0.92, f'Speedup QF_LRA vs QF_BV: ~{avg_speedup:.0f}x\nParidade de Veredito: 100% (20/20)', 
             transform=ax1.transAxes, bbox=dict(boxstyle='round,pad=0.4', fc='#fef3c7', ec='#d97706', lw=1.2), fontsize=8.5)
             
    # ----------------------------------------------------
    # Subplot 2: Escalabilidade Dimensional (d = 128 a 5120)
    # ----------------------------------------------------
    dims = [s["dimension"] for s in sc]
    t_sc_bv = [s["time_qf_bv_ms"] for s in sc]
    t_sc_lra = [s["time_qf_lra_ms"] for s in sc]
    
    ax2.plot(dims, t_sc_bv, 'o-', color='#b91c1c', linewidth=2, markersize=6, label='Btor2 / QF_BV (Word-Level Bit-Vector)')
    ax2.plot(dims, t_sc_lra, 's-', color='#1e40af', linewidth=2, markersize=6, label='Z3 QF_LRA (Continuous Linear Real)')
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Dimensão Latente / Embedding (d)')
    ax2.set_ylabel('Tempo de Prova UNSAT (ms - log)')
    ax2.set_title('(B) Escalabilidade Dimensional de d=128 até d=5120 (Qwen3-14B)', fontweight='bold')
    ax2.set_xticks(dims)
    ax2.grid(True, which='both', linestyle='--', alpha=0.4)
    ax2.legend(loc='center right', framealpha=0.9)
    
    for d_val, tb, tl in zip(dims, t_sc_bv, t_sc_lra):
        ax2.annotate(f'{tb:.0f}ms', (d_val, tb), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=7.5, color='#b91c1c', fontweight='bold')
        ax2.annotate(f'{tl:.1f}ms', (d_val, tl), textcoords="offset points", xytext=(0, -12), ha='center', fontsize=7.5, color='#1e40af', fontweight='bold')

    # ----------------------------------------------------
    # Subplot 3: Raio Critico de Perturbacao (eps*) por Token
    # ----------------------------------------------------
    colors_bar = plt.cm.plasma(np.linspace(0.2, 0.85, len(tokens)))
    bars = ax3.barh(tokens, eps_stars, color=colors_bar, edgecolor='black', linewidth=0.6)
    ax3.set_xlabel(r'Raio Crítico Formal $\epsilon^*$ (Margem $L_\infty$ Máxima)')
    ax3.set_title('(C) Raio Crítico de Robustez Formal $\epsilon^*$ no Qwen3-14B', fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.4, axis='x')
    
    for bar in bars:
        w_val = bar.get_width()
        ax3.text(w_val + 0.0003, bar.get_y() + bar.get_height()/2, f'{w_val:.5f}', 
                 va='center', ha='left', fontsize=8, fontweight='bold')
    ax3.set_xlim(0, max(eps_stars) * 1.25)
    
    # ----------------------------------------------------
    # Subplot 4: Matriz de Comparacao Metodologica Btor2 vs Z3
    # ----------------------------------------------------
    ax4.axis('off')
    
    table_data = [
        ["Critério de Avaliação", "Btor2 (Word-Level BV)", "Z3 SMT (QF_LRA Simplex)"],
        ["Espaço de Representação", "Vetores de Bits (Q16.16)", "Contínuo Real (R^5120)"],
        ["Precisão Aritmética", "Bit-Exata (sem arredondamento)", "Contínua com Relaxação"],
        ["Tempo Médio UNSAT (Safe)", f"{np.mean(t_bv_safe):.1f} ms (~5.0 s)", f"{np.mean(t_lra_safe):.1f} ms (< 0.01 s)"],
        ["Tempo Médio SAT (Adv)", f"{np.mean(t_bv_sat):.1f} ms (~1.5 s)", f"{np.mean(t_lra_sat):.1f} ms (< 0.01 s)"],
        ["Complexidade Algorítmica", "Bit-Blasting / BV-Lemma (NP-c)", "Simplex Primal-Dual (P-médio)"],
        ["Acordo de Veredito", "100% (20/20 instâncias)", "100% (20/20 instâncias)"],
        ["Formato de Intercâmbio", "Arquivo .btor2 aberto", "API SMT-LIB2 / Solver Nativo"],
        ["Garantia Formal", "Cota Exata em Inteiros", "Cota Teórica Conservadora"]
    ]
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.38, 0.31, 0.31])
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.45)
    
    # Estilizar cabecalho e celulas da tabela
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1e293b')
            cell.set_text_props(color='white', fontweight='bold')
        elif col == 0:
            cell.set_facecolor('#f1f5f9')
            cell.set_text_props(fontweight='bold')
        else:
            if row % 2 == 1:
                cell.set_facecolor('#ffffff')
            else:
                cell.set_facecolor('#f8fafc')
        cell.set_edgecolor('#cbd5e1')
        
    ax4.set_title('(D) Síntese Comparativa em Paridade de Parâmetros', fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(OUT_IMG)
    plt.close()
    print(f"[OK] Figura salva com sucesso em: {OUT_IMG}")

if __name__ == "__main__":
    plot_fig6()
