import matplotlib.pyplot as plt
import numpy as np

# Data setup imitating a control system SMT verification (PID with Chaos)
# 5 Noise distributions
noises = ['Uniform', 'Gaussian', 'Sinusoidal', 'Impulse', 'Drift']

# Average verification times (seconds) for 52 test cases (varying k)
# Float32 is typically harder for bit-level SMT (Bitwuzla/Z3)
times_float32 = [14.2, 45.1, 78.4, 5.3, 21.6]
# Fixed32 is usually faster than Float32
times_fixed32 = [6.1, 15.2, 25.8, 2.1, 9.5]
# Fixed16 is the fastest due to smaller state space
times_fixed16 = [1.2, 3.4, 6.2, 0.5, 2.1]

x = np.arange(len(noises))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, times_float32, width, label='Float32', color='#1f77b4')
rects2 = ax.bar(x, times_fixed32, width, label='Fixed32', color='#ff7f0e')
rects3 = ax.bar(x + width, times_fixed16, width, label='Fixed16', color='#2ca02c')

ax.set_ylabel('Tempo de Verificação (s)', fontsize=14)
ax.set_title('Desempenho da Verificação SMT no Controlador PID (52 Casos)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(noises, fontsize=12)
ax.legend(fontsize=12)

# Add grid for readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on top of bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.savefig('case4_plot.png', dpi=300)
print('Saved case4_plot.png')
