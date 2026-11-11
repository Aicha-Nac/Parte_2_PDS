"""Tópico 2 - Transformada de Fourier em Tempo Discreto (DTFT) de um pulso retangular."""
import os
import numpy as np
import matplotlib.pyplot as plt

PASTA = "figuras"
os.makedirs(PASTA, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

# Pulso retangular de 5 amostras: x[n] = 1 para 0 <= n <= 4
L = 5
n = np.arange(L)
x = np.ones(L)

# DTFT calculada diretamente pela definição, em uma grade fina de frequências
w = np.linspace(-np.pi, np.pi, 4001)
X = np.array([np.sum(x * np.exp(-1j * wi * n)) for wi in w])

# Expressão analítica: X(e^jw) = e^{-j2w} sen(5w/2) / sen(w/2)
with np.errstate(invalid="ignore", divide="ignore"):
    razao = np.where(np.isclose(w, 0), L, np.sin(L * w / 2) / np.sin(w / 2))
X_analitica = np.exp(-1j * (L - 1) / 2 * w) * razao
print(f"Diferença máxima entre a soma e a fórmula analítica: {np.abs(X - X_analitica).max():.1e}")
print(f"|X| em w = 0: {np.abs(X[np.argmin(np.abs(w))]):.4f} (soma das amostras = {x.sum():.0f})")
for k in (1, 2):
    wz = 2 * np.pi * k / L
    print(f"|X| em w = {2 * k}π/5 = {wz:.4f} rad/amostra: {np.abs(np.sum(x * np.exp(-1j * wz * n))):.1e}")

# Transformada inversa calculada numericamente: deve devolver o pulso
for amostra in range(-2, 8):
    valor = np.trapezoid(X * np.exp(1j * w * amostra), w).real / (2 * np.pi)
    print(f"Inversa numérica, x[{amostra:2d}] = {valor:6.3f}")

fig, ax = plt.subplots(3, 1, figsize=(10, 9))
ax[0].stem(np.arange(-4, 9), np.r_[np.zeros(4), x, np.zeros(4)], linefmt="tab:blue", markerfmt="o", basefmt=" ")
ax[0].set(title="Domínio do tempo: pulso retangular x[n], L = 5", xlabel="Índice n (amostras)", ylabel="Amplitude")
ax[1].plot(w / np.pi, np.abs(X), color="tab:blue")
for k in (-2, -1, 1, 2):
    ax[1].axvline(2 * k / L, color="gray", linestyle=":", linewidth=1)
ax[1].set(title="Espectro de magnitude |X(e^{jω})| (linhas pontilhadas: zeros em ω = 2πk/5)",
          xlabel="Frequência ω/π (rad/amostra, normalizada por π)", ylabel="|X(e^{jω})|")
ax[2].plot(w / np.pi, np.angle(X), color="tab:red")
ax[2].set(title="Espectro de fase ∠X(e^{jω})", xlabel="Frequência ω/π (rad/amostra, normalizada por π)",
          ylabel="Fase (rad)", yticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax[2].set_yticklabels(["−π", "−π/2", "0", "π/2", "π"])
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "t2_dtft.png"), dpi=150)
plt.show()
