import numpy as np
import matplotlib.pyplot as plt

W1 = np.array([[0.5, -0.2, 0.8], [-0.1, 0.9, -0.4]]) 
W2 = np.array([[0.6], [-0.5], [0.3]])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [3, 1]})

cax1 = ax1.matshow(W1, cmap='RdYlGn', vmin=-1, vmax=1)
ax1.set_title('Pesos Ocultos (W1: 2x3)', pad=15)
ax1.set_xticks([0, 1, 2])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['$H_1$', '$H_2$', '$H_3$'])
ax1.set_yticklabels(['$In_1$', '$In_2$'])
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        ax1.text(j, i, f"{W1[i,j]:.1f}", ha='center', va='center', color='black', fontweight='bold')

cax2 = ax2.matshow(W2.T, cmap='RdYlGn', vmin=-1, vmax=1)
ax2.set_title('Pesos de Saída (W2: 3x1)', pad=15)
ax2.set_xticks([0, 1, 2])
ax2.set_yticks([0])
ax2.set_xticklabels(['$H_1$', '$H_2$', '$H_3$'])
ax2.set_yticklabels(['$Out$'])
for i in range(W2.T.shape[0]):
    for j in range(W2.T.shape[1]):
        ax2.text(j, i, f"{W2.T[i,j]:.1f}", ha='center', va='center', color='black', fontweight='bold')

plt.colorbar(cax1, ax=[ax1, ax2], orientation='vertical', fraction=0.02, pad=0.04)
plt.suptitle('Heatmap de Pesos MLP: Origem das 32 VCCs Simbolicas', fontweight='bold')
plt.savefig('plot_case1_weights.png', dpi=300, bbox_inches='tight')
print("Saved plot_case1_weights.png")
