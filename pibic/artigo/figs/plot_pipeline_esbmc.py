import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4)
ax.axis("off")

steps = [
    (1.0, 2.0, "Código Fonte\n(C/C++/Python)\nContendo Assertions", "#4c78a8"),
    (3.5, 2.0, "Frontend do ESBMC\nTradução para\nGOTO-Programs", "#f58518"),
    (6.0, 2.0, "Execução Simbólica\n(Desenrolamento\nde Laços)", "#e15759"),
    (8.5, 2.0, "Gerador de Fórmulas\n(Codificação em\nBitvectores)", "#72b7b2"),
    (11.0, 2.0, "Solver SMT\n(Z3 / Bitwuzla)", "#54a24b"),
]

for (x, y, label, color) in steps:
    rect = mpatches.FancyBboxPatch((x - 1.0, y - 0.75), 2.0, 1.5, boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=3)

arrow_kw = dict(arrowstyle="->", color="#333333", lw=2.0)
for i in range(len(steps) - 1):
    x_start = steps[i][0] + 1.1
    x_end   = steps[i+1][0] - 1.1
    ax.annotate("", xy=(x_end, 2.0), xytext=(x_start, 2.0), arrowprops=arrow_kw, zorder=4)

ax.annotate("", xy=(12.3, 2.5), xytext=(11.0, 2.85), arrowprops=dict(arrowstyle="<-", color="#2d7a2d", lw=2), zorder=4)
ax.text(12.3, 2.85, "UNSAT\n(Seguro)", ha="left", va="center", fontsize=9, color="#2d7a2d", fontweight="bold")

ax.annotate("", xy=(12.3, 1.5), xytext=(11.0, 1.15), arrowprops=dict(arrowstyle="<-", color="#e15759", lw=2), zorder=4)
ax.text(12.3, 1.15, "SAT\n(Falha + Contra-exemplo)", ha="left", va="center", fontsize=9, color="#e15759", fontweight="bold")

ax.set_title("Pipeline de Verificação Formal Baseado em SMT (ESBMC)", fontsize=13, fontweight="bold", pad=10)

plt.tight_layout()
plt.savefig("/home/uchoa/esbmc/pibic/artigo/figs/fluxo-esbmc.png", dpi=150, bbox_inches="tight")
print("Salvo: /home/uchoa/esbmc/pibic/artigo/figs/fluxo-esbmc.png")
