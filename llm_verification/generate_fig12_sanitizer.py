# -*- coding: utf-8 -*-
"""
Gera a Figura 12: Reparo Cirurgico Formal (Formal Weight Sanitizer) do Qwen 3.8 27B
Comparativo ANTES (Z3: SAT - Vulneravel) vs. DEPOIS (Z3: UNSAT - 100% Seguro) nas 4 Falhas.
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 1.0


def main():
    fig = plt.figure(figsize=(16.5, 10.5), dpi=300)
    fig.patch.set_facecolor("#F8FAFC")

    fig.suptitle(
        "Reparo Cirúrgico Formal (Formal Weight Sanitizer) no Qwen 3.8 27B: Antes (Z3 SAT) vs. Depois (Z3 UNSAT)",
        fontsize=14.8,
        fontweight="bold",
        color="#0F172A",
        y=0.972,
    )
    fig.text(
        0.5,
        0.938,
        "Certificação Matemática Sem Retreinamento no Ambiente Virtual Isolado (venv_local_flaw_check) em 1.003,35 ms — 4/4 Vulnerabilidades Eliminadas (UNSAT)",
        ha="center",
        fontsize=10.1,
        color="#334155",
        style="italic",
    )

    gs = fig.add_gridspec(2, 2, left=0.08, right=0.97, top=0.88, bottom=0.13, wspace=0.26, hspace=0.36)

    # =========================================================================
    # Painel (A): PATCH-01 - Saneamento de Posto (0x00000000 -> 0x3E000000) & Condicionamento RMS
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    labels_p1 = [
        "blk.7.post_attn[3994]\n(Retenção de Ganho)",
        "blk.43.k_norm[32]\n(Retenção de Ganho)",
        "blk.59.q_norm[1]\n(Retenção de Ganho)",
    ]
    before_p1 = [3.19, 3.19, 3.19]
    after_p1 = [70.64, 70.64, 70.64]
    x1 = np.arange(len(labels_p1))
    w1 = 0.34

    b1_pre = ax1.bar(x1 - w1 / 2, before_p1, w1, color="#DC2626", edgecolor="#991B1B", label="ANTES: γ = 0x00000000 (Z3: SAT — 96,81% Perda)")
    b1_post = ax1.bar(x1 + w1 / 2, after_p1, w1, color="#16A34A", edgecolor="#15803D", label="DEPOIS: γ = 0x3E000000 (+0,125) & Clip (Z3: UNSAT)")

    for b in b1_pre:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 1.8, f"{h:.2f}%\n(SAT)", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#991B1B")
    for b in b1_post:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 1.8, f"{h:.2f}%\n(UNSAT)", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#15803D")

    ax1.set_title(
        "(A) PATCH-01: Restauração de Posto (det J > 0) e Proteção RMS",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels_p1, fontsize=8.6)
    ax1.set_ylabel("Ganho Retido nos Canais Ativos (%)", fontsize=9.2)
    ax1.set_ylim(0, 96)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=8.0, frameon=True, facecolor="#F8FAFC")

    # =========================================================================
    # Painel (B): PATCH-02 - Retificação Antipodal de Fase QK-Norm (Bit 31: 1 -> 0)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    labels_p2 = [
        "blk.31 (Canal RoPE 0)\n0xBE180000 → 0x3E180000",
        "blk.47 (Canal RoPE 0)\n0xBD200000 → 0x3D200000",
        "blk.59 (Canal RoPE 32)\n0xBC000000 → 0x3C000000",
    ]
    contrib_before = [-0.281356, -0.012340, -0.133274]
    contrib_after = [+0.281356, +0.012340, +0.133274]
    x2 = np.arange(len(labels_p2))

    b2_pre = ax2.bar(x2 - w1 / 2, contrib_before, w1, color="#DC2626", edgecolor="#991B1B", label="ANTES: γ_k < 0 (Autossimilaridade < 0 — Z3: SAT)")
    b2_post = ax2.bar(x2 + w1 / 2, contrib_after, w1, color="#16A34A", edgecolor="#15803D", label="DEPOIS: γ_k = |γ_k| > 0 (Coerência RoPE — Z3: UNSAT)")
    ax2.axhline(0, color="#0F172A", linewidth=1.2)

    for b in b2_pre:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h - 0.018, f"{h:.4f}\n(SAT)", ha="center", va="top", fontsize=7.9, fontweight="bold", color="#991B1B")
    for b in b2_post:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.012, f"+{h:.4f}\n(UNSAT)", ha="center", va="bottom", fontsize=7.9, fontweight="bold", color="#15803D")

    ax2.set_title(
        "(B) PATCH-02: Retificação de Fase 180° no QK-Norm (Δpos = 0)",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels_p2, fontsize=8.3)
    ax2.set_ylabel("Contribuição de Autoatenção C_i (q_i = k_i = 8,99)", fontsize=9.1)
    ax2.set_ylim(-0.38, 0.39)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="#F8FAFC")

    # =========================================================================
    # Painel (C): PATCH-03 - Clamping Espectral IEEE-754 em blk.12.ssm_a (Cabeças 41 e 9)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#FFFFFF")
    labels_p3 = [
        "Cabeça 41 (blk.12.ssm_a)\n-139,42 (0x00000000) → -20,0",
        "Cabeça 9 (blk.12.ssm_a)\n-102,00 (0x00000004) → -20,0",
    ]
    ret_before_p3 = [0.188, 3.41]
    ret_after_p3 = [40.64, 51.55]
    x3 = np.arange(len(labels_p3))

    b3_pre = ax3.bar(x3 - w1 / 2, ret_before_p3, w1, color="#DC2626", edgecolor="#991B1B", label="ANTES: Underflow / Subnormal FP32 (Z3: SAT)")
    b3_post = ax3.bar(x3 + w1 / 2, ret_after_p3, w1, color="#16A34A", edgecolor="#15803D", label="DEPOIS: ssm_a = -20,0 Normal FP32 (Z3: UNSAT)")

    for b in b3_pre:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width() / 2, h + 1.2, f"{h:.2f}%\n(SAT: Subnormal/0)", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#991B1B")
    for b in b3_post:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width() / 2, h + 1.2, f"{h:.2f}%\n(UNSAT: 24-bit Normal)", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#15803D")

    ax3.set_title(
        "(C) PATCH-03: Clamping IEEE-754 FP32 e Restauração de Memória SSM",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels_p3, fontsize=8.6)
    ax3.set_ylabel("Retenção de Memória por Passo A_bar (%)", fontsize=9.2)
    ax3.set_ylim(0, 66)
    ax3.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left", fontsize=8.0, frameon=True, facecolor="#F8FAFC")

    # =========================================================================
    # Painel (D): PATCH-04 - Eliminação de Starvation no Chat Template (xhigh -> none)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#FFFFFF")
    labels_p4 = [
        "Prompt Guardrail\n('rm -rf /' B=64)",
        "Prompt Automação\n(JSON status B=64)",
        "Prompt Aritmético\n('Quanto é 2+2?' B=64)",
    ]
    vis_before_p4 = [0, 0, 0]
    vis_after_p4 = [64, 64, 64]
    x4 = np.arange(len(labels_p4))

    b4_pre = ax4.bar(x4 - w1 / 2, vis_before_p4, w1, color="#DC2626", edgecolor="#991B1B", label="ANTES: default('xhigh') → 0 Visíveis (Z3: SAT Starvation)")
    b4_post = ax4.bar(x4 + w1 / 2, vis_after_p4, w1, color="#16A34A", edgecolor="#15803D", label="DEPOIS: default('none') → 100% Orçamento Visível (Z3: UNSAT)")

    for b in b4_pre:
        ax4.text(b.get_x() + b.get_width() / 2, 2.0, "0 tokens\n(response=\"\")\nSAT", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#991B1B")
    for b in b4_post:
        h = b.get_height()
        ax4.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h} tokens úteis\n(UNSAT)", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#15803D")

    ax4.set_title(
        "(D) PATCH-04: Garantia de Liveness no Template (Eliminação de Starvation)",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    ax4.set_xticks(x4)
    ax4.set_xticklabels(labels_p4, fontsize=8.5)
    ax4.set_ylabel("Orçamento Garantido de Tokens Visíveis (B=64)", fontsize=9.1)
    ax4.set_ylim(0, 82)
    ax4.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left", fontsize=8.0, frameon=True, facecolor="#F8FAFC")

    # Semantic color guide footer
    fig.text(
        0.5,
        0.033,
        "Guia Semântico de Cores Incorporado na Imagem:\n"
        "[Vermelho #DC2626] Estado Original Vulnerável do Qwen 3.8 27B (Veredito Z3: SAT — Contraexemplo de Falha Encontrado)\n"
        "[Verde Esmeralda #16A34A] Estado Pós-Reparo Cirúrgico Formal pelo Formal Weight Sanitizer (Veredito Z3: UNSAT — 100% Provado Seguro Sem Retreinamento)",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.44", fc="#E2E8F0", ec="#94A3B8", lw=1.0),
    )

    out_file = r"C:\uchoa\esbmc\llm_verification\assets\fig12_reparo_cirurgico_formal_antes_depois.png"
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print("FIGURA 12 GERADA COM SUCESSO:", out_file)


if __name__ == "__main__":
    main()
