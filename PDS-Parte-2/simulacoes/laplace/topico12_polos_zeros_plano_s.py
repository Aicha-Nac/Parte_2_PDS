import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Configuração visual dos gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax_plano, ax_tempo) = plt.subplots(1, 2, figsize=(14, 6))

# Define os sistemas a serem simulados: H(s) = num / den
# 1. Polo em p1 = -1
sys_p1 = signal.TransferFunction([1], [1, 1])

# 2. Polo em p2 = -5 (Ganho ajustado para amplitude unitária no degrau)
sys_p2 = signal.TransferFunction([5], [1, 5])

# 3. Polo em p3 = +1 (Instável)
sys_p3 = signal.TransferFunction([1], [1, -1])

# 4. Par complexo em p4,5 = -1 ± j5  --> s^2 + 2s + 26
sys_p45 = signal.TransferFunction([26], [1, 2, 26])

sistemas = [
    {"sys": sys_p1, "label": "p1 = -1 (Lento)", "color": "blue", "marker": "o"},
    {"sys": sys_p2, "label": "p2 = -5 (Rápido)", "color": "green", "marker": "s"},
    {"sys": sys_p3, "label": "p3 = +1 (Instável)", "color": "red", "marker": "^"},
    {"sys": sys_p45, "label": "p4,5 = -1 ± j5 (Oscilatório)", "color": "purple", "marker": "d"}
]

# -------------------------------------------------------------
# 1. Plot do Plano-s (Polos e Zeros)
# -------------------------------------------------------------
ax_plano.axvline(0, color='black', linewidth=1.2, linestyle='--')
ax_plano.axhline(0, color='black', linewidth=1.2, linestyle='--')

for item in sistemas:
    sys = item["sys"]
    polos = sys.poles
    ax_plano.scatter(polos.real, polos.imag, color=item["color"],
                     s=100, label=item["label"], zorder=5)

ax_plano.set_title("1. Representação dos Polos no Plano-s", fontsize=12, fontweight='bold')
ax_plano.set_xlabel("Eixo Real σ (Re)")
ax_plano.set_ylabel("Eixo Imaginário jω (Im)")
ax_plano.set_xlim([-6, 2])
ax_plano.set_ylim([-6, 6])
ax_plano.grid(True, linestyle=':', alpha=0.7)
ax_plano.legend(loc='upper left')

# Regiões de estabilidade
ax_plano.axvspan(-6, 0, color='green', alpha=0.05, label='Região Estável (SPE)')
ax_plano.axvspan(0, 2, color='red', alpha=0.05, label='Região Instável (SPD)')

# -------------------------------------------------------------
# 2. Simulação da Resposta ao Degrau
# -------------------------------------------------------------
t = np.linspace(0, 4, 1000)  # Vetor de tempo de 0 a 4 segundos

for item in sistemas:
    sys = item["sys"]
    # Calcula a resposta ao degrau
    t_out, y_out = signal.step(sys, T=t)
    ax_tempo.plot(t_out, y_out, label=item["label"], color=item["color"], linewidth=2)

ax_tempo.set_title("2. Resposta Temporal ao Degrau", fontsize=12, fontweight='bold')
ax_tempo.set_xlabel("Tempo (s)")
ax_tempo.set_ylabel("Amplitude y(t)")
ax_tempo.set_ylim([-0.5, 3.5])  # Limite vertical para não estourar com p3=+1
ax_tempo.grid(True, linestyle=':', alpha=0.7)
ax_tempo.legend(loc='upper left')

plt.tight_layout()
plt.show()