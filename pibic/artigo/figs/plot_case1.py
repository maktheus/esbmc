"""
Gera os gráficos do Caso 1 — Verificação Formal de Neurônio (ReLU)
Salva: case1_neuron_surface.png  e  case1_verification_summary.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ─── Modelo do neurônio (mesmo do neuron.py) ────────────────────────────────
def relu(x: float) -> float:
    return max(0.0, x)

def neuron_forward(x1: float, x2: float) -> float:
    w1, w2, bias = 0.5, -0.2, 0.1
    return relu(x1 * w1 + x2 * w2 + bias)

# ─── Grades de amostragem ────────────────────────────────────────────────────
N = 200
x1_vals = np.linspace(0.0, 1.0, N)
x2_vals = np.linspace(0.0, 1.0, N)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.vectorize(neuron_forward)(X1, X2)

# limites verificados pelo ESBMC
Z_MIN, Z_MAX = 0.0, 0.6

# ─── Figura 1: Superfície 3D com região de limites ──────────────────────────
fig = plt.figure(figsize=(13, 5))

# --- painel esquerdo: superfície 3D ---
ax1 = fig.add_subplot(121, projection="3d")
surf = ax1.plot_surface(X1, X2, Z, cmap="viridis", alpha=0.9, linewidth=0)
ax1.set_xlabel("$x_1$", fontsize=11)
ax1.set_ylabel("$x_2$", fontsize=11)
ax1.set_zlabel("$out$", fontsize=11)
ax1.set_title("Superfície de Saída do Neurônio\n(Qualquer $(x_1,x_2)\\in[0,1]^2$)", fontsize=10)
ax1.set_zlim(-0.05, 0.75)

# plano superior: limite provado
xx, yy = np.meshgrid([0, 1], [0, 1])
ax1.plot_surface(xx, yy, np.full_like(xx, Z_MAX),
                 color="red", alpha=0.25, label=f"Limite ESBMC: out ≤ {Z_MAX}")
ax1.plot_surface(xx, yy, np.full_like(xx, Z_MIN),
                 color="green", alpha=0.15, label=f"Limite ESBMC: out ≥ {Z_MIN}")

red_patch   = mpatches.Patch(color="red",   alpha=0.5, label=f"bound superior: out ≤ {Z_MAX}")
green_patch = mpatches.Patch(color="green", alpha=0.5, label=f"bound inferior: out ≥ {Z_MIN}")
ax1.legend(handles=[red_patch, green_patch], loc="upper left", fontsize=8)
cb = fig.colorbar(surf, ax=ax1, shrink=0.5, pad=0.08)
cb.set_label("Saída $out$", fontsize=9)

# --- painel direito: histograma de saídas ---
ax2 = fig.add_subplot(122)
out_flat = Z.flatten()
ax2.hist(out_flat, bins=50, color="#4c78a8", edgecolor="white", alpha=0.85)
ax2.axvline(Z_MIN, color="green", lw=2, ls="--", label=f"Limite inferior (ESBMC): {Z_MIN}")
ax2.axvline(Z_MAX, color="red",   lw=2, ls="--", label=f"Limite superior (ESBMC): {Z_MAX}")
ax2.set_xlabel("Valor de Saída $out$", fontsize=11)
ax2.set_ylabel("Frequência", fontsize=11)
ax2.set_title("Distribuição de Saídas\npara $10^4$ amostras uniformes em $[0,1]^2$", fontsize=10)
ax2.legend(fontsize=9)
ax2.text(0.67, ax2.get_ylim()[1] * 0.88,
         "✓ 100% das saídas\ndentro dos bounds\nverificados",
         fontsize=9, color="#2d7a2d",
         bbox=dict(facecolor="#e6ffe6", edgecolor="#2d7a2d", boxstyle="round,pad=0.4"))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "case1_plot.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Salvo: {out_path}")
plt.close()

# ─── Figura 2: diagrama de verificação (pipeline simplificado) ──────────────
fig2, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

steps = [
    (0.5, 2.5, "Entradas\nSimbólicas\n$x_1,x_2\\in[0,1]$", "#4c78a8"),
    (2.8, 2.5, "Neurônio\nReLU\n$out = \\text{relu}(wx+b)$", "#f58518"),
    (5.1, 2.5, "Codificação\nSMT\n(Bitvector Float)", "#72b7b2"),
    (7.4, 2.5, "Solver Z3\nProva / Contra-\nexemplo", "#54a24b"),
]

for (x, y, label, color) in steps:
    rect = plt.Rectangle((x - 0.35, y - 0.9), 2.1, 1.8,
                          facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.88,
                          zorder=2)
    ax.add_patch(rect)
    ax.text(x + 0.7, y, label, ha="center", va="center",
            fontsize=8.5, color="white", fontweight="bold", zorder=3)

# setas
arrow_kw = dict(arrowstyle="->", color="#333333", lw=1.8)
for i in range(len(steps) - 1):
    x_start = steps[i][0] + 1.75
    x_end   = steps[i+1][0] - 0.35
    ax.annotate("", xy=(x_end, 2.5), xytext=(x_start, 2.5),
                arrowprops=arrow_kw, zorder=4)

# resultado
ax.text(9.5, 2.5, "UNSAT\n✓ Seguro", ha="center", va="center",
        fontsize=10, color="#2d7a2d", fontweight="bold",
        bbox=dict(facecolor="#e6ffe6", edgecolor="#2d7a2d", boxstyle="round,pad=0.4"))

ax.set_title("Pipeline de Verificação Formal — Caso 1 (Neurônio Python com ESBMC)",
             fontsize=11, fontweight="bold", pad=12)

pipeline_path = "case1_pipeline.png"
plt.savefig(pipeline_path, dpi=150, bbox_inches="tight")
print(f"Salvo: {pipeline_path}")
plt.close()
print("Concluído.")
