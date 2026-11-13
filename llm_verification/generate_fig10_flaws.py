# -*- coding: utf-8 -*-
"""
Gera a Figura 10: Auditoria Formal de Falhas Intrinsecas no Qwen 3.8 27B (qwen3.8:27b)
Validadas em Duplo Ambiente Virtual Isolado (venv_flaw_audit & venv_local_flaw_check) via Z3 SMT.
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 1.0


def main():
    fig = plt.figure(figsize=(16.5, 10.4), dpi=300)
    fig.patch.set_facecolor("#F8FAFC")

    # Header banner
    fig.suptitle(
        "Auditoria de Falhas Intrínsecas no Qwen 3.8 27B (qwen3.8:27b — 65 Blocos / 16,81 GB GGUF)",
        fontsize=15.0,
        fontweight="bold",
        color="#0F172A",
        y=0.972,
    )
    fig.text(
        0.5,
        0.938,
        "Validação Formal Independente em Duplo Ambiente Virtual Isolado (venv_flaw_audit & venv_local_flaw_check) via Z3 SMT (QF_FP / QF_NRA / QF_LIA)",
        ha="center",
        fontsize=10.2,
        color="#334155",
        style="italic",
    )

    gs = fig.add_gridspec(2, 2, left=0.115, right=0.975, top=0.88, bottom=0.13, wspace=0.33, hspace=0.35)

    # =========================================================================
    # Painel (A): FLAW-01 - Zero-Weight Rank Collapse & RMS Denominator Poisoning
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    z_dead_vals = np.linspace(0, 500, 300)
    d_head = 256.0
    eps = 1e-6
    rms_base = np.sqrt(((d_head - 1.0) * 1.0 + 0.0) / d_head + eps)
    rms_curve = np.sqrt(((d_head - 1.0) * 1.0 + z_dead_vals**2) / d_head + eps)
    gain_retention_256 = (rms_base / rms_curve) * 100.0

    d_res = 5120.0
    rms_base_5120 = np.sqrt(((d_res - 1.0) * 1.0 + 0.0) / d_res + eps)
    rms_curve_5120 = np.sqrt(((d_res - 1.0) * 1.0 + z_dead_vals**2) / d_res + eps)
    gain_retention_5120 = (rms_base_5120 / rms_curve_5120) * 100.0

    ax1.plot(
        z_dead_vals,
        gain_retention_256,
        color="#DC2626",
        linewidth=2.6,
        label="QK-Norm (d=256): blk.43.k_norm[32]=0 & blk.59.q_norm[1]=0",
    )
    ax1.plot(
        z_dead_vals,
        gain_retention_5120,
        color="#D97706",
        linewidth=2.2,
        linestyle="--",
        label="Post-Attn Norm (d=5120): blk.7.post_attn_norm[3994]=0",
    )
    ax1.scatter([100, 500], [gain_retention_256[60], gain_retention_256[-1]], color="#991B1B", zorder=5, s=45)
    ax1.annotate(
        "z_dead = 100 → Retenção = 15,8%\n(Supressão de 84,2% nos 255 canais ativos!)",
        xy=(100, 15.81),
        xytext=(140, 38),
        arrowprops=dict(arrowstyle="->", color="#991B1B", lw=1.3),
        fontsize=8.3,
        fontweight="bold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#DC2626", lw=1),
    )
    ax1.annotate(
        "z_dead = 500 → Retenção = 3,19%\n(Supressão de 96,81% — Z3 SAT)",
        xy=(500, 3.19),
        xytext=(230, 12),
        arrowprops=dict(arrowstyle="->", color="#991B1B", lw=1.3),
        fontsize=8.3,
        fontweight="bold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#DC2626", lw=1),
    )
    ax1.set_title(
        "(A) Falha 1: Colapso de Posto (γ = 0x00000000) e Supressão RMS",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    ax1.set_xlabel("Magnitude de Perturbação no Canal Morto |z_dead| (Saída γ·z_dead = 0)", fontsize=9.2)
    ax1.set_ylabel("Ganho Retido nos Canais Ativos (%)", fontsize=9.2)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="#F8FAFC")

    # =========================================================================
    # Painel (B): FLAW-02 - Sign / Phase Inversion in QK-Norm (RoPE Channels 0 & 32)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    categories = [
        "blk.31 (Canal RoPE 0)\nq=+0.0234 | k=-0.1484",
        "blk.47 (Canal RoPE 0)\nq=+0.0039 | k=-0.0391",
        "blk.59 (Canal RoPE 32)\nq=+0.2109 | k=-0.0078",
    ]
    q_weights = [0.0234375, 0.00390625, 0.2109375]
    k_weights = [-0.1484375, -0.0390625, -0.0078125]
    x_pos = np.arange(len(categories))
    width = 0.34

    bars_q = ax2.bar(x_pos - width / 2, q_weights, width, label="γ_q (attn_q_norm > 0)", color="#0284C7", edgecolor="#0369A1")
    bars_k = ax2.bar(x_pos + width / 2, k_weights, width, label="γ_k (attn_k_norm < 0 — Inversão 180°)", color="#DC2626", edgecolor="#991B1B")
    ax2.axhline(0, color="#0F172A", linewidth=1.2)

    for b in bars_q:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.006, f"+{h:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#0369A1")
    for b in bars_k:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h - 0.018, f"{h:.4f}", ha="center", va="top", fontsize=8, fontweight="bold", color="#991B1B")

    ax2.text(
        0.48,
        0.54,
        "Violação de Monotonicidade Angular RoPE (Z3 QF_FP: SAT)\nPara vetores idênticos (q_i = k_i ≠ 0, Δpos = 0):\nContribuição ao Produto Escalar = γ_q · γ_k · q_i² < 0\n(Refletor Antipodal det = -1 no Subespaço RoPE)",
        transform=ax2.transAxes,
        ha="center",
        va="center",
        fontsize=8.3,
        color="#7F1D1D",
        bbox=dict(boxstyle="round,pad=0.38", fc="#FEF2F2", ec="#DC2626", lw=1.1),
    )

    ax2.set_title(
        "(B) Falha 2: Inversão Antipodal de Fase (180°) no QK-Norm (RoPE)",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, fontsize=8.6)
    ax2.set_ylabel("Peso de Escala QK-Norm (γ)", fontsize=9.2)
    ax2.set_ylim(-0.19, 0.26)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8.2, frameon=True, facecolor="#F8FAFC")

    # =========================================================================
    # Painel (C): FLAW-03 - IEEE-754 Float32 Exponential Horizon Violation in blk.12.ssm_a
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#FFFFFF")
    heads_labels = [
        "Média da Camada\n(46 Cabeças)",
        "Limiar Overflow\nIEEE-754 FP32",
        "Cabeça 9 (blk.12)\nSubnormal (0x00000004)",
        "Limiar Underflow\nTotal FP32",
        "Cabeça 41 (blk.12)\nUnderflow (0x00000000)",
    ]
    vals_abs_a = [6.2195, 88.7228, 102.0028, 103.2789, 139.4213]
    colors_c = ["#0284C7", "#D97706", "#EA580C", "#7C3AED", "#DC2626"]

    bars_c = ax3.barh(heads_labels, vals_abs_a, color=colors_c, height=0.54, edgecolor="#1E293B")
    ax3.axvline(88.7228, color="#D97706", linestyle="--", linewidth=1.8, label="ln(FLT_MAX) = 88,7228")
    ax3.axvline(103.2789, color="#7C3AED", linestyle=":", linewidth=1.8, label="|ln(FLT_TRUE_MIN)| = 103,2789")

    for b, v in zip(bars_c, vals_abs_a):
        ax3.text(v + 2.0, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", ha="left", fontsize=8.6, fontweight="bold", color="#0F172A")

    ax3.set_title(
        "(C) Falha 3: Violação do Horizonte Exponencial FP32 em blk.12.ssm_a",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    ax3.set_yticks(np.arange(len(heads_labels)))
    ax3.set_yticklabels(heads_labels, fontsize=8.3)
    ax3.set_xlabel("Magnitude da Taxa de Decaimento |ssm_a| (Escala Logarítmica natural)", fontsize=9.2)
    ax3.set_xlim(0, 165)
    ax3.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax3.legend(loc="lower right", fontsize=8.0, frameon=True, facecolor="#F8FAFC")

    # =========================================================================
    # Painel (D): FLAW-04 - Chat Template reasoning_effort='xhigh' Output Starvation
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#FFFFFF")
    configs = [
        "Template Default\n(xhigh, B=64)",
        "Template Default\n(xhigh, B=128)",
        "Mitigação Formal\n(think=False, B=64)",
    ]
    think_tokens = [64, 128, 0]
    visible_tokens = [0, 0, 5]
    y_pos = np.arange(len(configs))

    ax4.barh(y_pos, think_tokens, height=0.48, color="#DC2626", edgecolor="#991B1B", label="Tokens Consumidos em <think> (Ocultos)")
    ax4.barh(y_pos, visible_tokens, left=think_tokens, height=0.48, color="#16A34A", edgecolor="#15803D", label="Tokens Visíveis em response (Saída Útil)")

    ax4.text(67, 0, "response = \"\" (0 visíveis — STARVATION!)", va="center", ha="left", color="#991B1B", fontsize=8.3, fontweight="bold")
    ax4.text(64, 1, "response = \"\" (0 visíveis — STARVATION!)", va="center", ha="center", color="#FFFFFF", fontsize=8.3, fontweight="bold")
    ax4.text(9, 2, "response = \"4\" (5 tokens, 100% útil — SAFE)", va="center", ha="left", color="#15803D", fontsize=8.6, fontweight="bold")

    ax4.set_title(
        "(D) Falha 4: Inanição de Liveness (Starvation) por reasoning_effort='xhigh'",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(configs, fontsize=8.5)
    ax4.set_xlabel("Orçamento de Tokens Gerados (num_predict)", fontsize=9.2)
    ax4.set_xlim(0, 152)
    ax4.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="#F8FAFC")

    # Semantic color guide footer inside the figure (3 lines so it fits cleanly within margins)
    fig.text(
        0.5,
        0.036,
        "Guia Semântico de Cores Incorporado na Imagem:\n"
        "[Vermelho #DC2626] Violação Formal Provada (Z3 SAT: Colapso de Posto γ=0, Inversão -180° QK-Norm, Overflow/Underflow FP32, Starvation)\n"
        "[Laranja #EA580C / Roxo #7C3AED] Regime Subnormal e Limiares Críticos IEEE-754   |   [Azul #0284C7] Parâmetro Nominal   |   [Verde #16A34A] Mitigação Segura",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.45", fc="#E2E8F0", ec="#94A3B8", lw=1.0),
    )

    out_file = r"C:\uchoa\esbmc\llm_verification\assets\fig10_qwen38_27b_flaws_audit.png"
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print("FIGURA 10 REGERADA COM SUCESSO:", out_file)


if __name__ == "__main__":
    main()
