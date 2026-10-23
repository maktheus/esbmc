# -*- coding: utf-8 -*-
"""
Gera a Figura 4: Curva de Propagação Ponta-a-Ponta nas 28 Camadas do Qwen2.5-7B
e os Raios Críticos Certificados de Entrada (Input Prompt Embeddings).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

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

def plot_fig4_e2e():
    json_path = r"C:\uchoa\esbmc\llm_verification\benchmark_end_to_end_28cores.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    layers = data["layer_propagation"]
    results = data["results"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    
    # Subgráfico 1: Evolução do Envelope de Propagação através das 28 Camadas
    layer_indices = [l["layer"] for l in layers]
    gains = [l["cumulative_gain"] for l in layers]
    attn_norms = [l["attn_norm_mean"] for l in layers]
    ffn_norms = [l["ffn_norm_mean"] for l in layers]
    
    ax1.plot(layer_indices, gains, color='#7c3aed', marker='s', linewidth=2, label='Fator Cumulativo Residual $\\mathcal{G}_l$')
    ax1.plot(layer_indices, attn_norms, color='#2563eb', linestyle='--', linewidth=1.5, label='Média RMSNorm Attention ($\\gamma_{attn}$)')
    ax1.plot(layer_indices, ffn_norms, color='#059669', linestyle=':', linewidth=1.5, label='Média RMSNorm FFN ($\\gamma_{ffn}$)')
    
    ax1.axhline(data["total_expansion_factor"], color='#b91c1c', linestyle='-.', linewidth=1.5, 
                label=f'Expansão Global Final: {data["total_expansion_factor"]:.2f}x')
                
    ax1.set_title('(A) Propagação de Envelopes Abstratos nas 28 Camadas Transformer', fontweight='bold', pad=10)
    ax1.set_xlabel('Camada do Transformer ($l = 0 \\dots 27$)')
    ax1.set_ylabel('Fator de Expansão / Norma')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', framealpha=0.9)
    
    # Subgráfico 2: Raio Crítico Certificado na Entrada (Embedding do Prompt)
    tokens = [r["token"].replace("_fallback", "") for r in results]
    eps_in_values = [r["eps_critical_input"] * 1000 for r in results] # Em unidades de 10^-3
    
    x = np.arange(len(tokens))
    bars = ax2.bar(x, eps_in_values, color='#8b5cf6', width=0.6, edgecolor='#6d28d9')
    mean_val = float(np.mean(eps_in_values))
    ax2.axhline(mean_val, color='#b91c1c', linestyle='--', linewidth=1.5, label=f'Média: $\\bar{{\\epsilon}}_{{in}}^* = {mean_val:.3f} \\times 10^{{-3}}$')
    
    ax2.set_title('(B) Raio Crítico Certificado Ponta-a-Ponta na Entrada (Prompt)', fontweight='bold', pad=10)
    ax2.set_ylabel('$\\epsilon_{in}^*$ no Espaço de Embedding ($\\times 10^{-3}$)')
    ax2.set_xlabel('28 Classes de Exploit Provadas UNSAT Ponta-a-Ponta')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tokens, rotation=45, ha='right', fontsize=7.5)
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax2.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig4_end_to_end_28camadas.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Gerado: {out_path}")

if __name__ == "__main__":
    plot_fig4_e2e()
