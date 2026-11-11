"""Tópico 3 - DFT de um sinal com duas senoides."""
import os
import numpy as np
import matplotlib.pyplot as plt

PASTA = "figuras"
os.makedirs(PASTA, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

fs = 1000.0
N = 200
A1, f1, fase1 = 1.0, 50.0, 0.0
A2, f2, fase2 = 0.5, 120.0, -np.pi / 3
n = np.arange(N)
x = A1 * np.cos(2 * np.pi * f1 / fs * n + fase1) + A2 * np.cos(2 * np.pi * f2 / fs * n + fase2)

X = np.fft.fft(x)
k = np.arange(N)
f = k * fs / N
metade = N // 2
amplitude = 2 * np.abs(X[:metade]) / N
amplitude[0] = np.abs(X[0]) / N
fase = np.angle(X[:metade])
significativo = amplitude > 0.01

print(f"fs = {fs:g} Hz | N = {N} | Δf = fs/N = {fs / N:g} Hz")
print(" k | f (Hz) | |X[k]| | amplitude | fase (graus)")
for kk in np.flatnonzero(significativo):
    print(f"{kk:2d} | {f[kk]:6.1f} | {np.abs(X[kk]):6.2f} | {amplitude[kk]:9.4f} | {np.degrees(fase[kk]):7.2f}")
print("Simetria X[N-k] = conj(X[k]):", np.allclose(X[1:][::-1], np.conj(X[1:])))
print(f"Maior amplitude fora dos picos: {amplitude[~significativo].max():.1e}")

fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))
ax[0, 0].plot(n / fs * 1000, x, "o-", markersize=2.5, linewidth=0.8, color="tab:blue")
ax[0, 0].set(title="Domínio do tempo: x[n] = cos(2π·50n/fs) + 0,5 cos(2π·120n/fs − π/3)",
             xlabel="Tempo nTs (ms)", ylabel="Amplitude", xlim=(0, 100))
ax[0, 1].stem(k, np.abs(X), linefmt="tab:purple", markerfmt=".", basefmt=" ")
ax[0, 1].set(title="DFT completa |X[k]|, k = 0 a N − 1 (simetria em torno de N/2)", xlabel="Índice k",
             ylabel="|X[k]|")
ax[1, 0].stem(f[:metade], amplitude, linefmt="tab:blue", markerfmt=".", basefmt=" ")
ax[1, 0].set(title="Espectro de magnitude (amplitude de cada componente)", xlabel="Frequência (Hz)",
             ylabel="Amplitude")
ax[1, 1].stem(f[:metade][significativo], np.degrees(fase[significativo]), linefmt="tab:red", markerfmt="o",
              basefmt=" ")
ax[1, 1].set(title="Espectro de fase (só onde a magnitude não é nula)", xlabel="Frequência (Hz)",
             ylabel="Fase (graus)", xlim=(0, 500), ylim=(-90, 30))
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "t3_dft.png"), dpi=150)
plt.show()
