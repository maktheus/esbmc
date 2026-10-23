# -*- coding: utf-8 -*-
"""
Gera a Figura 7: Roadmap Estratégico e Próximos Passos da Verificação Formal de LLMs.
Diagrama vetorial com design acadêmico de alta fidelidade (300 DPI).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

OUT_PATH = r"C:\uchoa\esbmc\llm_verification\assets\fig7_roadmap_futuro_verificacao_llm.png"

def plot_roadmap():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Fundo sutil
    bg = patches.Rectangle((0, 0), 100, 100, facecolor='#f8fafc', edgecolor='none', zorder=0)
    ax.add_patch(bg)
    
    # Titulo e subtitulo
    ax.text(50, 95, "ROADMAP CIENTÍFICO E TECNOLÓGICO: PRÓXIMOS PASSOS NA VERIFICAÇÃO FORMAL DE LLMs", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='#0f172a')
    ax.text(50, 91, "Da Certificação Estática de Redes Densas à Salvaguarda Simbólica Dinâmica em Silício e Sistemas de Grande Porte", 
            ha='center', va='center', fontsize=9.5, color='#475569')

    # Secao 1: Estado Atual (Conquistas Deste Trabalho)
    # Caixa base esquerda
    card_current = patches.FancyBboxPatch((4, 10), 28, 75, boxstyle="round,pad=1.5,rounding_size=2", 
                                         facecolor='#eff6ff', edgecolor='#3b82f6', linewidth=1.5)
    ax.add_patch(card_current)
    ax.text(18, 81, "ESTADO ATUAL CONSOLIDADO", ha='center', va='center', fontsize=10, fontweight='bold', color='#1d4ed8')
    ax.text(18, 78, "(Mestrado / PPGI-UFAM)", ha='center', va='center', fontsize=8, color='#3b82f6')
    
    current_items = [
        ("Modelos Verificados:", "7B (28 cam.), 9B (32 cam.),\n14.8B (40 cam., d=5120)"),
        ("Dinâmica Autoregressiva:", "Atratores de Ciclo Limite (P=3)\ne Homeostase de Ponto Fixo"),
        ("Teorema de Envelopes:", "Contrações Lipschitz ponta-a-ponta\ne raio crítico certificado eps*"),
        ("Paridade Btor2 vs Z3:", "100% de acordo formal (20/20)\nem ponto fixo Q16.16"),
        ("Desempenho Bare-Metal:", "28 núcleos paralelos (< 2,0s)\nSpeedup 500x com Z3 Simplex")
    ]
    
    y_pos = 71
    for title, desc in current_items:
        ax.text(6, y_pos, f"• {title}", fontsize=8.2, fontweight='bold', color='#0f172a')
        ax.text(7.5, y_pos - 3.8, desc, fontsize=7.5, color='#334155')
        y_pos -= 11.5
        
    # Seta de transicao
    ax.annotate('', xy=(38, 48), xytext=(33, 48),
                arrowprops=dict(arrowstyle="simple,head_length=1.0,head_width=1.0,tail_width=0.4", 
                                color='#2563eb', alpha=0.8))

    # Secao 2: Os 4 Pilares de Proximos Passos
    steps = [
        {
            "num": "PASSO 1",
            "title": "Síntese de Hardware & Btor2 em Silício",
            "subtitle": "Boolector / Bitwuzla em NPU & FPGA",
            "desc": "Transpilação dos modelos .btor2 para circuitos lógicos Verilog/VHDL. Implementação de verificadores formais Bit-Vector embarcados diretamente nas NPUs de inferência para garantia física contra transientes e jailbreaks de hardware.",
            "color": "#7c3aed",
            "bg": "#f5f3ff",
            "border": "#8b5cf6"
        },
        {
            "num": "PASSO 2",
            "title": "Guardrail Simbólico Streaming (Zero Latência)",
            "subtitle": "Verificação Online em Tempo Real (< 10ms)",
            "desc": "Integração do solver Z3 QF_LRA no pipeline de decodificação token-a-token da engine Ollama/vLLM. Interrupção garantida via prova matemática sem impacto perceptual ao usuário, substituindo filtros heurísticos frágeis.",
            "color": "#059669",
            "bg": "#ecfdf5",
            "border": "#10b981"
        },
        {
            "num": "PASSO 3",
            "title": "Escala para Mixture of Experts (MoE) & > 70B",
            "subtitle": "Roteamento Booleano e Modelos Esparsos",
            "desc": "Extensão do formalismo para arquiteturas MoE (ex: Mixtral 8x7B, Qwen-MoE). Verificação das funções de roteamento via SMT combinacional e isolamento formal de sub-redes especialistas seguras.",
            "color": "#d97706",
            "bg": "#fffbeb",
            "border": "#f59e0b"
        },
        {
            "num": "PASSO 4",
            "title": "BMC Interativo Multi-Turn com k-Indução",
            "subtitle": "Model Checking de Contexto Longo",
            "desc": "Transição de verificação de passo único para Bounded Model Checking temporal. Prova indutiva de invariantes de segurança ao longo de conversas infinitas (k passos de diálogo) sob perturbações adversariais cumulativas.",
            "color": "#dc2626",
            "bg": "#fef2f2",
            "border": "#ef4444"
        }
    ]
    
    positions = [
        (40, 50, 27, 35),
        (70, 50, 27, 35),
        (40, 10, 27, 35),
        (70, 10, 27, 35)
    ]
    
    for step, (x, y, w, h) in zip(steps, positions):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.2,rounding_size=2", 
                                    facecolor=step["bg"], edgecolor=step["border"], linewidth=1.4)
        ax.add_patch(box)
        
        # Badge
        badge = patches.FancyBboxPatch((x + 1.5, y + h - 5.5), 8.5, 3.8, boxstyle="round,pad=0.3,rounding_size=1",
                                     facecolor=step["color"], edgecolor='none')
        ax.add_patch(badge)
        ax.text(x + 5.75, y + h - 3.6, step["num"], ha='center', va='center', fontsize=7.5, fontweight='bold', color='white')
        
        ax.text(x + 11.5, y + h - 3.6, step["title"], ha='left', va='center', fontsize=8.2, fontweight='bold', color='#0f172a')
        ax.text(x + 2, y + h - 8.5, step["subtitle"], fontsize=7.2, fontweight='bold', color=step["color"])
        
        # Texto da descricao
        ax.text(x + 2, y + h - 13, step["desc"], fontsize=7.2, color='#334155', wrap=True, va='top', 
                linespacing=1.35)

    plt.tight_layout()
    plt.savefig(OUT_PATH)
    plt.close()
    print(f"[OK] Roadmap Figura 7 gerada em: {OUT_PATH}")

if __name__ == "__main__":
    plot_roadmap()
