"""
Caso 1 — mapa de calor dos pesos da MLP verificada.

DADOS REAIS. A versão anterior deste script usava matrizes inventadas:

    W1 = np.array([[0.5, -0.2, 0.8], [-0.1, 0.9, -0.4]])
    W2 = np.array([[0.6], [-0.5], [0.3]])

e o artigo legendava a figura resultante como "atestando a origem topológica
das VCCs" — atribuindo significado de verificação a números que ninguém tinha
verificado, de uma rede que não existe. Agora lê `teste_mlp/mlp_weights.h`, que
é a rede efetivamente submetida ao ESBMC.

Saída: plot_case1_weights.png
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
PESOS = os.path.join(AQUI, "..", "..", "teste_mlp", "mlp_weights.h")


def le_array(fonte, nome):
    """Extrai um array float do header C, preservando a forma declarada."""
    m = re.search(rf"float {nome}((?:\[\d+\])+)\s*=\s*\{{(.*?)\}};",
                  fonte, re.S)
    if not m:
        raise SystemExit(f"{nome} não encontrado em {PESOS}")
    dims = [int(d) for d in re.findall(r"\[(\d+)\]", m.group(1))]
    vals = [float(v) for v in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", m.group(2))]
    return np.array(vals).reshape(dims)


with open(PESOS) as fh:
    src = fh.read()

w_hidden = le_array(src, "w_hidden")     # (4, 2)
w_out = le_array(src, "w_out").reshape(-1, 1)   # (4, 1)

fig, axes = plt.subplots(1, 2, figsize=(9, 4),
                         gridspec_kw={"width_ratios": [2, 1]})
lim = max(abs(w_hidden).max(), abs(w_out).max())

for ax, W, titulo, xlabel, ylabel in (
    (axes[0], w_hidden, "Camada oculta  $W_1$  (4×2)", "entrada", "neurônio"),
    (axes[1], w_out, "Saída  $W_2$  (4×1)", "saída", "neurônio"),
):
    im = ax.imshow(W, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(range(W.shape[1]))
    ax.set_yticks(range(W.shape[0]))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            v = W[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > lim * 0.55 else "black")

fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04, label="valor do peso")
fig.suptitle("Caso 1: pesos da MLP verificada (teste_mlp/mlp_weights.h)",
             fontsize=11)
plt.savefig(os.path.join(AQUI, "plot_case1_weights.png"), dpi=150,
            bbox_inches="tight")
plt.close()
print(f"plot_case1_weights.png: W1{w_hidden.shape} W2{w_out.shape} "
      f"lidos de {os.path.relpath(PESOS, AQUI)}")
