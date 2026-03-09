import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
ax.set_title('Máquina de Estados: Loop Neuro-Simbólico LLM-ESBMC (Ralph Loop)', fontsize=12, fontweight='bold', pad=20)

colors = ['#C0D6E4', '#FFDDC1', '#C5E1A5', '#FFF59D', '#FFAB91']
labels = [
    "LLM (Agente)\n[Gera Código C]",
    "Ralph Orchestrator\n[Injeta __ESBMC_assume]",
    "ESBMC core\n[Verificação SMT]",
    "Sucesso\n[Deploy]",
    "Falha (Trace Parser)\n[Envia Trace como Prompt]"
]
positions = [(0.1, 0.5), (0.4, 0.5), (0.7, 0.5), (0.7, 0.2), (0.4, 0.8)]

for i, (pos, label, color) in enumerate(zip(positions, labels, colors)):
    rect = patches.FancyBboxPatch((pos[0]-0.1, pos[1]-0.1), 0.2, 0.2, boxstyle="round,pad=0.03", 
                                  linewidth=1.5, edgecolor='black', facecolor=color, alpha=0.9)
    ax.add_patch(rect)
    ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=9, fontweight='bold')

# Arrows
ax.annotate('', xy=(0.3, 0.5), xytext=(0.2, 0.5), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
ax.annotate('', xy=(0.6, 0.5), xytext=(0.5, 0.5), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
ax.annotate('', xy=(0.7, 0.3), xytext=(0.7, 0.4), arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=8))
ax.annotate('', xy=(0.5, 0.7), xytext=(0.6, 0.6), arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8))
ax.annotate('', xy=(0.2, 0.6), xytext=(0.3, 0.8), arrowprops=dict(facecolor='orange', shrink=0.05, width=2, headwidth=8, connectionstyle="arc3,rad=-0.2"))

plt.tight_layout()
plt.savefig('plot_agent_statemachine.png', dpi=300, bbox_inches='tight')
print("Saved plot_agent_statemachine.png")
