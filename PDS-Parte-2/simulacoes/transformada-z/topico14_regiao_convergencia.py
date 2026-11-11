import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
a = 0.7
n = np.arange(-5, 15)

# Sequência 1: Causal (ROC: |z| > |a|)
x_causal = np.where(n >= 0, a**n, 0)

# Sequência 2: Anti-causal (ROC: |z| < |a|)
x_anticausal = np.where(n < 0, -(a**n), 0)

# Configuração da Figura
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Graph 1: Plano-Z com ROC Causal ---
ax1 = axes[0]
theta = np.linspace(0, 2*np.pi, 200)

# Círculos
ax1.plot(np.cos(theta), np.sin(theta), 'k--', label='Círculo Unitário (|z|=1)')
ax1.plot(a*np.cos(theta), a*np.sin(theta), 'r:', label=f'Fronteira da ROC (|z|={a})')

# Preenchimento da ROC Causal (|z| > a)
ax1.fill_between(2*np.cos(theta), 2*np.sin(theta), a*np.cos(theta),
                 color='green', alpha=0.2, label='ROC Causal (|z| > 0.7)')

ax1.scatter([a], [0], color='red', marker='x', s=100, linewidths=2, label=f'Polo z = {a}')
ax1.scatter([0], [0], color='blue', marker='o', s=80, facecolors='none', edgecolors='blue', label='Zero z = 0')

ax1.set_xlim([-1.5, 1.5])
ax1.set_ylim([-1.5, 1.5])
ax1.set_aspect('equal')
ax1.set_title("Plano-z: ROC Causal (|z| > 0.7)\n(Inclui o círculo unitário -> Estável)", fontsize=10)
ax1.set_xlabel("Re{z}")
ax1.set_ylabel("Im{z}")
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right', fontsize=8)

# --- Graph 2: Resposta Temporal Causal ---
ax2 = axes[1]
ax2.stem(n, x_causal, basefmt=" ")
ax2.set_title(r"Sinal Causal: x[n]=(0.7)nu[n]", fontsize=11)
ax2.set_xlabel("n (Amostras)")
ax2.set_ylabel("Amplitude")
ax2.grid(True, linestyle=':', alpha=0.6)

# --- Graph 3: Resposta Temporal Anti-causal ---
ax3 = axes[2]
ax3.stem(n, x_anticausal, basefmt=" ", linefmt='r-', markerfmt='ro')
ax3.set_title(r"Sinal Anti-causal: x[n]=-(0.7)nu[-n-1]", fontsize=11)
ax3.set_xlabel("n (Amostras)")
ax3.set_ylabel("Amplitude")
ax3.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()