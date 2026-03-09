import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Config
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis('off')
ax.set_title('Pipeline de Tradução: Python para SMT via ast2json e ESBMC', fontsize=12, fontweight='bold', pad=20)

# Colors
colors = ['#4B8BBE', '#FFE873', '#FFD43B', '#646464', '#FF5733']
labels = [
    "Python\nFrontend\n(Módulo Dinâmico)",
    "ast2json\nParser\n(Extração AST)",
    "GOTO-Program\n(Transpilação C++)",
    "ESBMC core\n(Execução Simbólica)",
    "Z3 / Bitwuzla\n(Solver SMT)"
]

x_start = 0.05
y = 0.4
width = 0.14
height = 0.3
spacing = 0.05

for i, (label, color) in enumerate(zip(labels, colors)):
    x = x_start + i * (width + spacing)
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.03", 
                                  linewidth=1.5, edgecolor='black', facecolor=color, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    if i < len(labels) - 1:
        ax.annotate('', xy=(x + width + spacing, y + height/2), 
                    xytext=(x + width, y + height/2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))

plt.tight_layout()
plt.savefig('plot_python_to_smt.png', dpi=300, bbox_inches='tight')
print("Saved plot_python_to_smt.png")
