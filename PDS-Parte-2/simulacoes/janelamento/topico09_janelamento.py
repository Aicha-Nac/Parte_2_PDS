"""Tópico 9 - Vazamento espectral e janelas retangular, Hann e Hamming."""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window

PASTA = "figuras"
os.makedirs(PASTA, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

fs, N, f0 = 1000.0, 256, 100.0
PREENCHIMENTO = 16384
n = np.arange(N)
x = np.cos(2 * np.pi * f0 / fs * n)
print(f"Δf = {fs / N:.3f} Hz | f0/Δf = {f0 / (fs / N):.2f} (não é inteiro: a senoide cai entre dois bins)")

janelas = {"Retangular": np.ones(N), "Hann": get_window("hann", N), "Hamming": get_window("hamming", N)}
cores = {"Retangular": "tab:red", "Hann": "tab:blue", "Hamming": "tab:green"}


def em_db(v):
    return 20 * np.log10(np.maximum(v, 1e-12))


def lobos(w):
    """Largura do lóbulo principal (entre os primeiros nulos) e nível do maior lóbulo lateral."""
    W = np.abs(np.fft.fft(w, PREENCHIMENTO))
    W = W / W[0]
    passo = PREENCHIMENTO / N
    i = 1
    while W[i + 1] < W[i]:
        i += 1
    return 2 * i / passo, em_db(W[i:PREENCHIMENTO // 2].max())


fig, ax = plt.subplots(2, 1, figsize=(11, 8.5))
fp = np.arange(PREENCHIMENTO) * fs / PREENCHIMENTO
print("Janela     | lóbulo principal (bins) | maior lóbulo lateral (dB) | pico (Hz) | amplitude no pico | "
      "nível a 50 Hz do pico (dB)")
for nome, w in janelas.items():
    ganho = w.sum()
    Xp = np.abs(np.fft.fft(x * w, PREENCHIMENTO)) * 2 / ganho
    Xb = np.abs(np.fft.fft(x * w)) * 2 / ganho
    kb = np.argmax(Xb[:N // 2])
    largura, lateral = lobos(w)
    nivel_50 = em_db(Xp[np.argmin(np.abs(fp - (f0 + 50)))])
    print(f"{nome:10s} | {largura:23.2f} | {lateral:25.1f} | {kb * fs / N:9.2f} | {Xb[kb]:17.3f} | {nivel_50:.1f}")
    ax[0].plot(fp[:PREENCHIMENTO // 2], em_db(Xp[:PREENCHIMENTO // 2]), color=cores[nome], linewidth=1,
               label=f"{nome}: lóbulo lateral máx. {lateral:.0f} dB")
    W = np.abs(np.fft.fftshift(np.fft.fft(w, PREENCHIMENTO)))
    bins = (np.arange(PREENCHIMENTO) - PREENCHIMENTO // 2) / (PREENCHIMENTO / N)
    ax[1].plot(bins, em_db(W / W.max()), color=cores[nome], linewidth=1, label=nome)
ax[0].axvline(f0, color="gray", linestyle=":", linewidth=1)
ax[0].set(title=f"Espectro de uma senoide de {f0:g} Hz fora de um bin (N = {N}, fs = {fs:g} Hz)",
          xlabel="Frequência (Hz)", ylabel="Amplitude (dB)", xlim=(0, 250), ylim=(-110, 5))
ax[0].legend(loc="upper right", fontsize=8)
ax[1].set(title="Espectro das próprias janelas (resposta a uma senoide exatamente num bin)",
          xlabel="Distância ao centro (bins)", ylabel="Magnitude normalizada (dB)", xlim=(-12, 12), ylim=(-100, 5))
ax[1].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "t9_janelamento.png"), dpi=150)
plt.show()
