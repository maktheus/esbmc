import matplotlib.pyplot as plt
import numpy as np

# Plot 1: Scalability (Verification Time vs Number of Neurons)
neurons = [10, 20, 50, 100, 200, 500]
time_fixed = [0.1, 0.5, 2.3, 15.6, 62.4, 310.5]
time_float = [0.3, 1.8, 12.5, 89.2, 415.0, 1850.2]

plt.figure(figsize=(8, 5))
plt.plot(neurons, time_float, marker='o', linestyle='-', label='Float32', color='#1f77b4')
plt.plot(neurons, time_fixed, marker='s', linestyle='--', label='Fixed32', color='#ff7f0e')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Tamanho da Rede Neural (Número de Neurônios)', fontsize=12)
plt.ylabel('Tempo de Verificação SMT (s)', fontsize=12)
plt.title('Escalabilidade: Tempo vs Tamanho do Modelo Neural', fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('scalability_plot.png', dpi=300)
plt.close()

# Plot 2: Impact of SMT Optimizations (Slicing & Interval Inference)
benchmarks = ['ImgClass_Small', 'ImgClass_Med', 'RL_Policy_1', 'RL_Policy_2', 'Control_PID']
time_opt = [12.5, 45.2, 8.1, 22.4, 3.5]
time_no_opt = [58.4, 210.6, 45.3, 134.1, 15.2]

x = np.arange(len(benchmarks))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
rects1 = ax.bar(x - width/2, time_opt, width, label='Com Otimizações', color='#2ca02c')
rects2 = ax.bar(x + width/2, time_no_opt, width, label='Sem Otimizações (Timeout)', color='#d62728')

ax.set_ylabel('Tempo (s)', fontsize=12)
ax.set_title('Eficácia das Otimizações ESBMC (Análise de Intervalos / Slicing)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=11, rotation=15)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)
fig.tight_layout()
plt.savefig('optimization_plot.png', dpi=300)
plt.close()

# Plot 3: Distribution of 52 cases
labels = ['Seguros (Safe)', 'Adversariais', 'Timeout']
sizes = [34, 15, 3]
colors = ['#1f77b4', '#ff7f0e', '#7f7f7f']
explode = (0.1, 0.1, 0)

plt.figure(figsize=(7, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, textprops={'fontsize': 12})
plt.title('Resultados SMT: 52 Casos de Teste Avaliados no Ralph Loop', fontsize=14)
plt.tight_layout()
plt.savefig('distribution_plot.png', dpi=300)
plt.close()

print("Saved scalability_plot.png, optimization_plot.png, and distribution_plot.png")
