"""
Caso 6 — distribuição das ações da política RL contra os limites do atuador.

DADOS DA POLÍTICA REAL. A versão anterior amostrava uma normal inventada:

    actions = np.random.normal(0, 1.5, 1000)

sem seed, de modo que a figura não regenerava igual, e a distribuição não tinha
relação com a política verificada. Agora a ação é calculada pela **mesma
expressão** de `cases/ai_model_checking/rl_policy.c`, sobre a mesma caixa de
estados que o harness declara em seus `__ESBMC_assume`:

    steering = pos_y * 0.01 + velocity * 0.05
    pos_y    in [0, 100]        velocity in [-10, 10]

O contra-exemplo que o ESBMC encontra é assim visível na propria figura: o
canto `pos_y = 100, velocity = 10` produz `1.5`, meio volante alem do limite
declarado de `1.0`. A regiao violadora nao e ruido de amostragem — e um setor
determinado da caixa de estados.

Saída: plot_rl_shield.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
LIMITE = 1.0

# grade determinística sobre a caixa declarada no harness — sem amostragem
# aleatória, portanto sem necessidade de seed
pos_y = np.linspace(0.0, 100.0, 400)
velocity = np.linspace(-10.0, 10.0, 400)
PY_, VEL = np.meshgrid(pos_y, velocity)
acoes = PY_ * 0.01 + VEL * 0.05          # rl_policy.c

planas = acoes.ravel()
seguras = planas[np.abs(planas) <= LIMITE]
violadas = planas[np.abs(planas) > LIMITE]
frac = 100.0 * violadas.size / planas.size

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# — histograma
ax = axes[0]
bins = np.linspace(planas.min(), planas.max(), 60)
ax.hist(seguras, bins=bins, color="#59a14f", alpha=.85,
        label=f"Dentro dos limites ({100 - frac:.1f}%)")
ax.hist(violadas, bins=bins, color="#e15759", alpha=.85,
        label=f"Excede o atuador ({frac:.1f}%)")
for v in (-LIMITE, LIMITE):
    ax.axvline(v, color="black", ls="--", lw=1.6)
ax.set_xlabel("Sinal de controle (volante)")
ax.set_ylabel("Frequência na grade de estados")
ax.set_title("Caso 6: distribuição da ação sobre a caixa declarada")
ax.legend(fontsize=9)
ax.grid(alpha=.3)

# — onde no espaço de estados a violação ocorre
ax = axes[1]
m = ax.pcolormesh(pos_y, velocity, acoes, cmap="RdBu_r",
                  vmin=-abs(acoes).max(), vmax=abs(acoes).max(), shading="auto")
ax.contour(pos_y, velocity, np.abs(acoes), levels=[LIMITE],
           colors="black", linewidths=2)
ax.set_xlabel("$pos_y$"); ax.set_ylabel("$velocity$")
ax.set_title("Região violadora (contorno em $|a| = 1{,}0$)")
fig.colorbar(m, ax=ax, label="ação")

pior = acoes.max()
ax.plot(100, 10, "k*", ms=14)
ax.annotate(f"máx = {pior:.2f}", xy=(100, 10), xytext=(62, 6.4),
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=1.2))

plt.tight_layout()
plt.savefig(os.path.join(AQUI, "plot_rl_shield.png"), dpi=150,
            bbox_inches="tight")
plt.close()
print(f"plot_rl_shield.png: ação máxima {pior:.2f} (limite {LIMITE}); "
      f"{frac:.1f}% da grade excede o atuador")
