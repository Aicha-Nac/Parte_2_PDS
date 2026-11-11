"""Tópico 5 - Resolução espectral com duas frequências próximas."""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

PASTA = "figuras"
os.makedirs(PASTA, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

fs = 1000.0
f1, f2 = 100.0, 110.0
PREENCHIMENTO = 16384

fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
print("    N | Δf (Hz) | duração (ms) | picos encontrados entre 80 e 130 Hz")
for eixo, N in zip(ax, [64, 256, 1024]):
    n = np.arange(N)
    x = np.cos(2 * np.pi * f1 / fs * n) + np.cos(2 * np.pi * f2 / fs * n)
    X = np.fft.fft(x)
    f = np.arange(N) * fs / N
    amplitude = 2 * np.abs(X) / N

    # Curva contínua (DTFT do trecho observado) obtida com preenchimento de zeros
    Xp = np.fft.fft(x, PREENCHIMENTO)
    fp = np.arange(PREENCHIMENTO) * fs / PREENCHIMENTO
    ap = 2 * np.abs(Xp) / N
    faixa = (fp > 80) & (fp < 130)
    picos, _ = find_peaks(ap[faixa], height=0.3)
    encontrados = fp[faixa][picos]
    print(f"{N:5d} | {fs / N:7.3f} | {N / fs * 1000:12.0f} | {np.round(encontrados, 1)}")

    eixo.plot(fp, ap, color="gray", linewidth=1, label="curva com preenchimento de zeros")
    eixo.stem(f, amplitude, linefmt="tab:blue", markerfmt="o", basefmt=" ", label="pontos da DFT de N pontos")
    eixo.axvline(f1, color="tab:red", linestyle=":", linewidth=1)
    eixo.axvline(f2, color="tab:red", linestyle=":", linewidth=1, label="frequências reais (100 e 110 Hz)")
    eixo.set(title=f"N = {N}: Δf = {fs / N:.2f} Hz, observação de {N / fs * 1000:.0f} ms",
             ylabel="Amplitude", xlim=(60, 150), ylim=(0, 1.3))
    eixo.legend(loc="upper right", fontsize=7)
ax[-1].set_xlabel("Frequência (Hz)")
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "t5_resolucao.png"), dpi=150)
plt.show()
