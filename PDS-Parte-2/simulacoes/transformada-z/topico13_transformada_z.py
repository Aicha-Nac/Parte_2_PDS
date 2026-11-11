import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Configuração estética do Matplotlib
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# -------------------------------------------------------------
# 1. Definição do Sistema Discreto H(z)
# -------------------------------------------------------------
# Exemplo: Filtro IIR de 2ª ordem
# H(z) = (z^2 - 0.5z) / (z^2 - 1.2z + 0.64)
# Numerador b_k e Denominador a_k em potências decrescentes de z:
num = [1.0, -0.5, 0.0]        # z^2 - 0.5z
den = [1.0, -1.2, 0.64]       # z^2 - 1.2z + 0.64

# Cria o objeto de transferência discreta (frequência de amostragem arbitrária/normalizada)
sys_z = signal.TransferFunction(num, den, dt=1.0)

# Extrai polos e zeros
zeros = sys_z.zeros
polos = sys_z.poles

print("Polos do sistema:", polos)
print("Zeros do sistema:", zeros)

# -------------------------------------------------------------
# 2. Criação da Figura (3 Subplots)
# -------------------------------------------------------------
fig = plt.figure(figsize=(15, 5))
ax_pz = fig.add_subplot(1, 3, 1)
ax_mag = fig.add_subplot(1, 3, 2)
ax_phase = fig.add_subplot(1, 3, 3)

# --- A. Mapa de Polos e Zeros no Plano-z ---
# Plot do Círculo Unitário |z| = 1
theta = np.linspace(0, 2 * np.pi, 300)
ax_pz.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1.5, label='Círculo Unitário (|z|=1)')

# Eixos Re e Im
ax_pz.axhline(0, color='gray', linestyle=':', alpha=0.7)
ax_pz.axvline(0, color='gray', linestyle=':', alpha=0.7)

# Polos (x) e Zeros (o)
ax_pz.scatter(polos.real, polos.imag, color='red', marker='x', s=120, linewidths=2.5, label='Polos', zorder=5)
ax_pz.scatter(zeros.real, zeros.imag, color='blue', marker='o', s=100, facecolors='none', edgecolors='blue', linewidths=2, label='Zeros', zorder=5)

ax_pz.set_title("Plano-z: Polos e Zeros", fontsize=12, fontweight='bold')
ax_pz.set_xlabel("Parte Real Re{z}")
ax_pz.set_ylabel("Parte Imaginária Im{z}")
ax_pz.set_xlim([-1.5, 1.5])
ax_pz.set_ylim([-1.5, 1.5])
ax_pz.set_aspect('equal')
ax_pz.grid(True, linestyle=':', alpha=0.6)
ax_pz.legend(loc='upper right')

# --- B. Resposta em Frequência (Magnitude e Fase) ---
# w: Frequência normalizada de 0 a pi rad/amostra
w, h = signal.freqz(num, den, worN=1024)

# Magnitude em dB
mag_db = 20 * np.log10(np.abs(h))

# Fase em graus
fase_deg = np.angle(h, deg=True)

# Plot da Magnitude
ax_mag.plot(w / np.pi, mag_db, color='purple', linewidth=2)
ax_mag.set_title("Resposta em Frequência (Magnitude)", fontsize=12, fontweight='bold')
ax_mag.set_xlabel("Frequência Normalizada (×π rad/amostra)")
ax_mag.set_ylabel("Magnitude (dB)")
ax_mag.grid(True, linestyle=':', alpha=0.6)

# Plot da Fase
ax_phase.plot(w / np.pi, fase_deg, color='green', linewidth=2)
ax_phase.set_title("Resposta em Frequência (Fase)", fontsize=12, fontweight='bold')
ax_phase.set_xlabel("Frequência Normalizada (×π rad/amostra)")
ax_phase.set_ylabel("Fase (graus)")
ax_phase.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()