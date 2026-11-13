# -*- coding: utf-8 -*-
"""
Gera a Figura 11: Macro-Arquitetura Metodologica (Passo-a-Passo) e Mapa de Gatilhos
de Prompts/Entradas que Acionam as 4 Falhas Intrinsecas do Qwen 3.8 27B.
"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"


def draw_box(ax, x, y, w, h, title, body, fc="#FFFFFF", ec="#1E293B", title_color="#0F172A", lw=1.6):
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h - 0.025,
        title,
        ha="center",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + w / 2,
        y + h / 2 - 0.022,
        body,
        ha="center",
        va="center",
        fontsize=7.75,
        color="#1E293B",
        linespacing=1.26,
    )


def main():
    fig = plt.figure(figsize=(16.8, 11.2), dpi=300)
    fig.patch.set_facecolor("#F8FAFC")

    fig.suptitle(
        "Macro-Arquitetura de Descoberta Formal e Mapa de Gatilhos de Entrada/Prompts no Qwen 3.8 27B",
        fontsize=15.2,
        fontweight="bold",
        color="#0F172A",
        y=0.975,
    )
    fig.text(
        0.5,
        0.944,
        "Pipeline Completo em 6 Estágios (Mestrado PPGEE/UFAM) Conectando o Binário GGUF (16,81 GB) às Provas Z3 SMT e aos Prompts Reais de Ativação",
        ha="center",
        fontsize=10.2,
        color="#334155",
        style="italic",
    )

    # Top panel: Macro-Architecture Flowchart
    ax_top = fig.add_axes([0.02, 0.51, 0.96, 0.41])
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")

    ax_top.text(
        0.5,
        0.96,
        "PARTE I — MACRO-ARQUITETURA DO MOTOR DE AUDITORIA FORMAL (PASSO A PASSO PARA CHEGAR ÀS DESCOBERTAS)",
        ha="center",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color="#0F172A",
    )

    # Row 1 of boxes (Stages 1, 2, 3)
    draw_box(
        ax_top,
        0.015,
        0.52,
        0.29,
        0.34,
        "ESTÁGIO 1: Ingestão Zero-Copy (mmap)",
        "• Binário GGUF qwen3.8:27b (16,81 GB)\n"
        "• Mapeamento virtual O(1) sem explodir RAM\n"
        "• Extração dos 863 tensores (65 blocos qwen35:\n"
        "  48 SSM + 16 Full-Attn + 1 NextN) + Jinja2",
        fc="#EFF6FF",
        ec="#0284C7",
        title_color="#0369A1",
    )

    draw_box(
        ax_top,
        0.355,
        0.52,
        0.29,
        0.34,
        "ESTÁGIO 2: Extrator de Invariantes Duplo",
        "• Nível Macro: Envelope Lipschitz (G=19,95x)\n"
        "  e Estabilidade de Hurwitz nos 48 blocos SSM\n"
        "• Nível Micro: Scanner Bit-a-Bit IEEE-754 de\n"
        "  Posto (γ=0), Fase QK-Norm e Horizonte FP32",
        fc="#F5F3FF",
        ec="#7C3AED",
        title_color="#6D28D9",
    )

    draw_box(
        ax_top,
        0.695,
        0.52,
        0.29,
        0.34,
        "ESTÁGIO 3: Isolamento em Duplo Venv",
        "• Venv I (HPC Remoto): venv_flaw_audit\n"
        "  (Python 3.14 isolado — Extração + QF_NRA)\n"
        "• Venv II (Local): venv_local_flaw_check\n"
        "  (Python 3.12 isolado — Confirmação QF_FP)",
        fc="#ECFDF5",
        ec="#059669",
        title_color="#047857",
    )

    # Row 2 of boxes (Stages 4, 5, 6)
    draw_box(
        ax_top,
        0.015,
        0.06,
        0.29,
        0.34,
        "ESTÁGIO 4: Codificação Multi-Teoria SMT",
        "• QF_FP (FPSort(8,24) IEEE-754 binary32)\n"
        "• QF_NRA (Aritmética Real Não-Linear RMS)\n"
        "• QF_BV (Btor2 Q16.16 64-bit Hardware)\n"
        "• QF_LRA & QF_LIA (Dual-Tail CEGAR & Liveness)",
        fc="#FFFBEB",
        ec="#D97706",
        title_color="#B45309",
    )

    draw_box(
        ax_top,
        0.355,
        0.06,
        0.29,
        0.34,
        "ESTÁGIO 5: Veredito Z3 & Contraexemplos",
        "• Certificação Macro: 248.320 saídas UNSAT\n"
        "  (Pior caso global ID 3966 Ġsol em 22,38 ms)\n"
        "• Descoberta Micro: 4 Falhas Intrínsecas SAT\n"
        "  (FLAW-01, FLAW-02, FLAW-03 e FLAW-04)",
        fc="#FEF2F2",
        ec="#DC2626",
        title_color="#B91C1C",
    )

    draw_box(
        ax_top,
        0.695,
        0.06,
        0.29,
        0.34,
        "ESTÁGIO 6: Gatilhos de Prompt & Reparo",
        "• Mapeamento de quais Prompts/Entradas reais\n"
        "  acionam cada uma das 4 falhas na prática\n"
        "• Saneamento Cirúrgico de Pesos (Patching)\n"
        "  e Contrato de API (think=False / Clamping)",
        fc="#F0FDF4",
        ec="#16A34A",
        title_color="#15803D",
    )

    # Arrows connecting Stages 1 -> 2 -> 3 -> 4 -> 5 -> 6
    arrow_kw = dict(arrowstyle="->,head_width=0.35,head_length=0.35", color="#334155", lw=2.0)
    ax_top.annotate("", xy=(0.350, 0.69), xytext=(0.310, 0.69), arrowprops=arrow_kw)
    ax_top.annotate("", xy=(0.690, 0.69), xytext=(0.650, 0.69), arrowprops=arrow_kw)
    ax_top.annotate(
        "",
        xy=(0.16, 0.415),
        xytext=(0.84, 0.505),
        arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", color="#475569", lw=1.8),
    )
    ax_top.annotate("", xy=(0.350, 0.23), xytext=(0.310, 0.23), arrowprops=arrow_kw)
    ax_top.annotate("", xy=(0.690, 0.23), xytext=(0.650, 0.23), arrowprops=arrow_kw)

    # Bottom panel: 4 Cards showing EXACT PROMPTS / INPUTS that trigger each flaw
    ax_bot = fig.add_axes([0.02, 0.08, 0.96, 0.41])
    ax_bot.set_xlim(0, 1)
    ax_bot.set_ylim(0, 1)
    ax_bot.axis("off")

    ax_bot.text(
        0.5,
        0.96,
        "PARTE II — QUE TIPO DE PROMPT OU ENTRADA OCASIONA CADA FALHA NA PRÁTICA? (MAPA DE GATILHOS REAIS)",
        ha="center",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color="#991B1B",
    )

    # 4 vertical trigger cards side by side
    draw_box(
        ax_bot,
        0.01,
        0.04,
        0.232,
        0.84,
        "GATILHO DA FALHA 1 (FLAW-01)\nPesos Zero (0x00000000) na RMSNorm",
        "• Onde está:\n"
        "  blk.7.post_attn_norm[3994] = 0\n"
        "  blk.43.attn_k_norm[32] = 0\n"
        "  blk.59.attn_q_norm[1] = 0\n\n"
        "• Que entrada ocasiona o erro?\n"
        "  1) Embeddings Multimodais / RAG /\n"
        "     Soft-Prompts Adversariais que\n"
        "     injetam energia nos canais 3994,\n"
        "     32 ou 1.\n"
        "  2) Prompts com tokens cuja projeção\n"
        "     ativa fortemente a dimensão 3994.\n\n"
        "• O que acontece no modelo?\n"
        "  A informação do canal é apagada\n"
        "  (0x0) e o denominador RMS infla,\n"
        "  suprimindo em até 96,81% os outros\n"
        "  255 canais da cabeça de atenção!",
        fc="#FEF2F2",
        ec="#DC2626",
        title_color="#991B1B",
    )

    draw_box(
        ax_bot,
        0.259,
        0.04,
        0.232,
        0.84,
        "GATILHO DA FALHA 2 (FLAW-02)\nInversão -180° QK-Norm (RoPE)",
        "• Onde está:\n"
        "  blk.31.attn_k_norm[0] = -0,1484\n"
        "  blk.47.attn_k_norm[0] = -0,0391\n"
        "  blk.59.attn_k_norm[32] = -0,0078\n\n"
        "• Que prompt ocasiona o erro?\n"
        "  Prompts com repetição contígua de\n"
        "  identificadores, variáveis de código\n"
        "  ou números idênticos (Δpos = 0..1):\n"
        "  Ex: \"Compare x_0 == x_0 e ID_99 == ID_99\"\n\n"
        "• O que acontece no modelo?\n"
        "  Os canais 0 e 32 são a frequência\n"
        "  posicional máxima do RoPE (1 rad/tok).\n"
        "  Como γ_q > 0 e γ_k < 0, tokens\n"
        "  idênticos recebem produto escalar\n"
        "  NEGATIVO (-0,2814), repelindo a\n"
        "  autoatenção em vez de atraí-la!",
        fc="#FFF7ED",
        ec="#EA580C",
        title_color="#9A3412",
    )

    draw_box(
        ax_bot,
        0.508,
        0.04,
        0.232,
        0.84,
        "GATILHO DA FALHA 3 (FLAW-03)\nOverflow/Underflow FP32 em ssm_a",
        "• Onde está:\n"
        "  blk.12.ssm_a[41] = -139,42 (0xC30B6BD8)\n"
        "  blk.12.ssm_a[9]  = -102,00 (0xC2CC016C)\n\n"
        "• Que prompt ocasiona o erro?\n"
        "  Prompts de Contexto Longo onde um\n"
        "  dado crítico (senha, código, tensão)\n"
        "  é fornecido no início e perguntado\n"
        "  mais de 2 a 16 tokens adiante.\n\n"
        "• O que acontece no modelo?\n"
        "  A Cabeça 41 perde 99,81% da memória\n"
        "  em 1 único token (e sofre Underflow\n"
        "  total para 0x00000000 em 16 tokens!\n"
        "  Já a Cabeça 9 opera em regime\n"
        "  Subnormal IEEE-754 (0x00000004),\n"
        "  perdendo 21 dos 24 bits de mantissa!",
        fc="#F5F3FF",
        ec="#7C3AED",
        title_color="#5B21B6",
    )

    draw_box(
        ax_bot,
        0.757,
        0.04,
        0.232,
        0.84,
        "GATILHO DA FALHA 4 (FLAW-04)\nStarvation por reasoning='xhigh'",
        "• Onde está:\n"
        "  Metadado tokenizer.chat_template\n"
        "  (default reasoning_effort = 'xhigh')\n\n"
        "• Que prompt ocasiona o erro?\n"
        "  Qualquer prompt direto de produção\n"
        "  com orçamento curto (num_predict ≤ 128):\n"
        "  1) \"Responda SIM ou NAO: rm -rf /\"\n"
        "  2) \"Retorne o JSON {status: OK}\"\n"
        "  3) \"Quanto é 2+2?\"\n\n"
        "• O que acontece no modelo?\n"
        "  O modelo gasta 100% dos tokens\n"
        "  divagando em <think> e retorna\n"
        "  response = \"\" (STRING VAZIA!),\n"
        "  derrubando parsers e guardrails!",
        fc="#FEF2F2",
        ec="#B91C1C",
        title_color="#7F1D1D",
    )

    # Footer legend
    fig.text(
        0.5,
        0.032,
        "Guia Semântico de Cores Incorporado na Imagem:\n"
        "[Azul #0284C7] Ingestão Binária GGUF   |   [Roxo #7C3AED] Extração Simbólica & Estado Subnormal   |   "
        "[Verde #059669] Ambientes Virtuais Isolados & Mitigação\n"
        "[Laranja #D97706] Formulação SMT & Fase RoPE   |   [Vermelho #DC2626] Contraexemplos SAT & Gatilhos de Entrada/Prompts Reais",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.42", fc="#E2E8F0", ec="#94A3B8", lw=1.0),
    )

    out_file = r"C:\uchoa\esbmc\llm_verification\assets\fig11_macro_arquitetura_auditoria_prompts.png"
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print("FIGURA 11 GERADA COM SUCESSO:", out_file)


if __name__ == "__main__":
    main()
