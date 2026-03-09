import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Measured times from the report
N_measured = np.array([2, 3, 4, 5, 6])
time_measured = np.array([0.5, 2.0, 5.0, 8.0, 15.0]) # seconds

# Fit an exponential curve: time = a * exp(b * N)
# ln(time) = ln(a) + b * N
p = np.polyfit(N_measured, np.log(time_measured), 1)
a = np.exp(p[1])
b = p[0]

# Extrapolate to N=60
N_extrapolated = np.arange(2, 61)
time_extrapolated = a * np.exp(b * N_extrapolated)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(N_extrapolated, time_extrapolated, 'r--', label='Tempo Projetado (Exponencial)', alpha=0.7)
ax1.plot(N_measured, time_measured, 'o-', color='blue', label='Tempo Medido (N$\\leq$6)', markersize=8)

ax1.set_xlabel("Dimensão da Matriz ($N \\times N$)", fontsize=12)
ax1.set_ylabel("Tempo de Verificação SMT (s)", fontsize=12)
ax1.set_title("Escalabilidade da Verificação Formal: GEMM com Tiling (até $N=60$)", fontsize=14)

ax1.set_yscale('log')
ax1.grid(True, which="both", ls="--", alpha=0.5)

# Annotation for N=60
ax1.annotate(f"N=60: Tempo inviável\n(~{time_extrapolated[-1]:.1e} s)", 
             xy=(60, time_extrapolated[-1]), xytext=(40, time_extrapolated[-1]/1000),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

ax1.legend(loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('case2_plot.png', dpi=300)
print("Salvo: case2_plot.png")
