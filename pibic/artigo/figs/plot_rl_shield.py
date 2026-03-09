import matplotlib.pyplot as plt
import numpy as np

# RL Action distribution pre and post shield
actions = np.random.normal(0, 1.5, 1000)
safe_actions = actions[(actions >= -1.0) & (actions <= 1.0)]
pruned_actions = actions[(actions < -1.0) | (actions > 1.0)]

plt.figure(figsize=(8, 5))
plt.hist(safe_actions, bins=30, color='green', alpha=0.7, label='Ações Válidas (Aprovadas)')
plt.hist(pruned_actions, bins=30, color='red', alpha=0.7, label='Ações Podadas (Safety Shield Mapped)')
plt.axvline(-1.0, color='black', linestyle='--', linewidth=2, label='Limites do Atuador')
plt.axvline(1.0, color='black', linestyle='--', linewidth=2)
plt.title('Distribuição SMT: Ações Permitidas vs Interceptação pré-Física', fontweight='bold')
plt.xlabel('Magnitude do Sinal de Controle (Volante)')
plt.ylabel('Frequência das Amostragens')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_rl_shield.png', dpi=300, bbox_inches='tight')
print("Saved plot_rl_shield.png")
