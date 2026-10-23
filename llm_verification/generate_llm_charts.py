# -*- coding: utf-8 -*-
"""
Gera os gráficos científicos em alta resolução (300 DPI) para o relatório
de Verificação Formal em Modelos de Linguagem de 8B (Qwen2.5-7B / Nimble-9B).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuração de estilo visual acadêmico
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

OUT_DIR = r"C:\uchoa\esbmc\llm_verification\assets"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# FIGURA 1: Retrato de Fase e Ciclo Limite Autoregressivo (P = 3)
# -------------------------------------------------------------------------
def plot_fig1_attractor():
    with open(r"C:\uchoa\esbmc\llm_verification\resultado_atratores_llm.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    
    # Subfigura 1A: Trajetória dos Tokens e Captura no Ciclo Limite
    # Dados do Regime 2 (Prompt Degenerado com Ciclo P=3)
    # Tokens: [198, 7141, 13440, 21619, 8287, 13440, 21619, 8287, 13440, 21619]
    tokens = [198, 7141, 13440, 21619, 8287, 13440, 21619, 8287, 13440, 21619]
    steps = list(range(len(tokens)))
    
    # Plot da série temporal de IDs
    ax1.plot(steps, tokens, color='#1e3a8a', marker='o', linewidth=2, label='Trajetória Token ID')
    ax1.axvspan(1.5, 9.5, color='#dbeafe', alpha=0.5, label='Bacia do Ciclo Limite (T = 3)')
    
    # Destacar os pontos do ciclo periódico
    for k in range(2, len(tokens)):
        tok = tokens[k]
        name = "beta" if tok == 13440 else ("gamma" if tok == 21619 else "alpha")
        ax1.annotate(f" {name}", (k, tok), fontsize=8, color='#1e40af', fontweight='bold', va='bottom')
        
    ax1.set_title('(A) Dinâmica Autoregressiva e Captura no Ciclo Limite (P = 3)', fontweight='bold', pad=10)
    ax1.set_xlabel('Passo Autoregressivo ($k$)')
    ax1.set_ylabel('Token ID na Camada de Saída')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', framealpha=0.9)
    
    # Subfigura 1B: Matriz de Recorrência de Fase (Recurrence Plot)
    # Mostra a identidade de fase R_{i,j} = (w_i == w_j)
    rec_matrix = np.zeros((len(tokens), len(tokens)))
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if tokens[i] == tokens[j]:
                rec_matrix[i, j] = 1.0
                
    im = ax2.imshow(rec_matrix, cmap='Blues', interpolation='nearest', origin='lower')
    ax2.set_title('(B) Matriz de Recorrência de Estados ($R_{i,j} \\equiv w_i = w_j$)', fontweight='bold', pad=10)
    ax2.set_xlabel('Índice Temporal $i$')
    ax2.set_ylabel('Índice Temporal $j$')
    
    # Adicionar anotação de periodicidade
    ax2.text(5, 5, 'Padrão Diagonal Periódico\n(Período P = 3)', color='#b91c1c', 
             fontweight='bold', fontsize=9, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#ef4444', alpha=0.9))
             
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig1_llm_ciclo_limite_atrator.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Gerado: {out_path}")

# -------------------------------------------------------------------------
# FIGURA 2: Geometria do Polítopo SMT e Raio Crítico Certificado em R^3584
# -------------------------------------------------------------------------
def plot_fig2_polytope():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    
    # Desenho conceitual 2D da projeção do espaço latente R^3584
    # Hiperplano separador: w_bad^T h = w_safe^T h  => (w_bad - w_safe)^T h = 0
    # Centro: h0 (Estado Nominal)
    
    h0 = np.array([2.0, 2.0])
    
    # Hiperplano: 0.8 * x + 0.6 * y = 4.2
    # Normal do hiperplano (diff)
    diff = np.array([0.8, 0.6])
    c = 4.2
    
    x_line = np.linspace(-0.5, 4.5, 100)
    y_line = (c - diff[0] * x_line) / diff[1]
    
    # Região Unsafe (acima do hiperplano)
    ax.fill_between(x_line, y_line, 5.0, color='#fee2e2', alpha=0.6, label='Região Insegura / Bad Token (SAT)')
    # Região Safe (abaixo do hiperplano)
    ax.fill_between(x_line, -0.5, y_line, color='#ecfdf5', alpha=0.6, label='Região Segura / Safe Token (UNSAT)')
    
    # Linha da Fronteira SMT
    ax.plot(x_line, y_line, color='#b91c1c', linestyle='--', linewidth=2, label='Fronteira SMT: $(w_{bad} - w_{safe})^T h = 0$')
    
    # Ponto h0
    ax.scatter([h0[0]], [h0[1]], color='#1e3a8a', s=100, zorder=5, label='Estado Nominal $h_0$ (Resposta Alinhada)')
    
    # Bola de Certeza Formal B_{eps*} (tocando exatamente a fronteira)
    # Distância de h0 ao hiperplano: |diff . h0 - c| / ||diff|| = |0.8*2 + 0.6*2 - 4.2| / 1.0 = |2.8 - 4.2| = 1.4
    eps_star = 1.4
    circle_crit = patches.Circle(h0, eps_star, fill=False, edgecolor='#059669', linewidth=2.2, linestyle='-', label='Raio Crítico Certificado $\\mathcal{B}_{\\epsilon^*}$ (UNSAT Seguro)')
    ax.add_patch(circle_crit)
    
    # Bola Vulnerável (violando a fronteira)
    circle_vuln = patches.Circle(h0, eps_star * 1.35, fill=False, edgecolor='#dc2626', linewidth=1.5, linestyle=':', label='Perturbação Adversarial $\\epsilon > \\epsilon^*$ (SAT Jailbreak)')
    ax.add_patch(circle_vuln)
    
    # Vetor de distância mínima (direção do contraexemplo)
    closest_pt = h0 + diff * eps_star
    ax.annotate('', xy=closest_pt, xytext=h0, arrowprops=dict(arrowstyle='->', color='#059669', lw=2))
    ax.text(2.6, 2.9, '$\\epsilon^* = \\frac{(w_{safe} - w_{bad})^T h_0}{\\|w_{bad} - w_{safe}\\|_1}$', 
            fontsize=10, fontweight='bold', color='#047857', bbox=dict(boxstyle='round,pad=0.2', facecolor='#d1fae5', edgecolor='none'))
            
    ax.set_xlim(0, 4.2)
    ax.set_ylim(0, 4.5)
    ax.set_title('Geometria do Polítipo de Decisão em $\\mathbb{R}^{3584}$ e Certificação Formal SMT', fontweight='bold', pad=12)
    ax.set_xlabel('Dimensão Latente Ativa $h_{(i)}$ (Direção de Maior Variância)')
    ax.set_ylabel('Dimensão Latente Ativa $h_{(j)}$ (Direção Ortogonal)')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower left', framealpha=0.95, fontsize=8)
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig2_llm_smt_barreira_polytopo.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Gerado: {out_path}")

# -------------------------------------------------------------------------
# FIGURA 3: Benchmark em 28 Núcleos Nativos: Raios Críticos e Latência Z3
# -------------------------------------------------------------------------
def plot_fig3_benchmark():
    with open(r"C:\uchoa\esbmc\llm_verification\benchmark_llm_28cores.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    results = data["results"]
    tokens = [r["token"].replace("_fallback", "") for r in results]
    eps_stars = [r["eps_critical"] * 1000 for r in results] # Em unidades de 10^-3
    times_ms = [r["time_sec"] * 1000 for r in results] # Em milissegundos
    
    x = np.arange(len(tokens))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    
    # Subgráfico Superior: Raio Crítico Certificado epsilon*
    bars1 = ax1.bar(x, eps_stars, color='#2563eb', width=0.6, edgecolor='#1d4ed8')
    ax1.axhline(np.mean(eps_stars), color='#b91c1c', linestyle='--', linewidth=1.5, label=f'Média: $\\bar{{\\epsilon}}^* = {np.mean(eps_stars):.2f} \\times 10^{{-3}}$')
    ax1.set_ylabel('Raio Crítico $\\epsilon^*$ ($\\times 10^{-3}$)', fontweight='bold')
    ax1.set_title('(A) Raio Crítico Certificado Formalmente contra 28 Tokens de Exploit / Jailbreak', fontweight='bold', pad=8)
    ax1.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax1.legend(loc='upper right')
    
    # Subgráfico Inferior: Tempo de Resolução SMT por Núcleo
    bars2 = ax2.bar(x, times_ms, color='#059669', width=0.6, edgecolor='#047857')
    ax2.axhline(np.mean(times_ms), color='#d97706', linestyle='--', linewidth=1.5, label=f'Média de CPU: {np.mean(times_ms):.1f} ms / fatia')
    ax2.set_ylabel('Latência SMT (ms)', fontweight='bold')
    ax2.set_xlabel('Fatias de Tokens Verificadas em Paralelo (28 Núcleos)')
    ax2.set_title('(B) Latência de Verificação SMT com Abstração CEGAR (Saturação 28 Threads)', fontweight='bold', pad=8)
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax2.legend(loc='upper right')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig3_llm_28cores_benchmark.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Gerado: {out_path}")

if __name__ == "__main__":
    plot_fig1_attractor()
    plot_fig2_polytope()
    plot_fig3_benchmark()
