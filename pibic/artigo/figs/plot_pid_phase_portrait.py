import numpy as np
import matplotlib.pyplot as plt

# Simulate chaos PID for phase portrait
t = np.linspace(0, 10, 100)
e = np.exp(-0.5 * t) * np.cos(2 * t) + np.random.normal(0, 0.1, 100)
de_dt = np.gradient(e, t)

plt.figure(figsize=(6, 6))
plt.plot(e, de_dt, color='purple', alpha=0.6, marker='o', markersize=3, label='Trajetória em Malha Fechada')
plt.scatter(e[0], de_dt[0], color='green', s=100, label='Início', zorder=5)
plt.scatter(e[-1], de_dt[-1], color='red', s=100, marker='X', label='Convergência (Atraindo \para e=0)', zorder=5)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.title('Retrato de Fase PID sob Caos', fontweight='bold')
plt.xlabel('Erro $e(t)$')
plt.ylabel('Taxa de Variação $de/dt$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_pid_phase_portrait.png', dpi=300, bbox_inches='tight')
print("Saved plot_pid_phase_portrait.png")
