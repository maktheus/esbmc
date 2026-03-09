import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-6, 6, 400)
silu_exact = z / (1 + np.exp(-z))
silu_approx = np.clip(0.25 * z + 0.5, 0, 1) * z

plt.figure(figsize=(8, 5))
plt.plot(z, silu_exact, label=r'Exato: $z \cdot \sigma(z)$', color='blue', linewidth=2)
plt.plot(z, silu_approx, label=r'Aproximação ESBMC: $z \cdot \text{clip}(0.25z+0.5, 0, 1)$', color='red', linestyle='--', linewidth=2)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.title('Substituição Transcendental no ESBMC: SiLU vs Aproximação Linear', fontweight='bold')
plt.xlabel('Input $z$')
plt.ylabel('Ativação')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_silu_approx.png', dpi=300, bbox_inches='tight')
print("Saved plot_silu_approx.png")
