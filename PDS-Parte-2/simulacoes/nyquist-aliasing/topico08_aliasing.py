"""Tópico 8 - Aliasing de uma senoide de 70 Hz amostrada a 100 Hz."""
import os
import numpy as np
import matplotlib.pyplot as plt

PASTA = "figuras"
os.makedirs(PASTA, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def frequencia_aparente(f0, fs):
    """Distância de f0 até o múltiplo de fs mais próximo."""
    return abs(f0 - fs * np.round(f0 / fs))


def pico(x, fs):
    N = len(x)
    a = 2 * np.abs(np.fft.fft(x)[:N // 2]) / N
    k = np.argmax(a)
    return k * fs / N, a[k]


f0, fs = 70.0, 100.0
n = np.arange(100)
x = np.cos(2 * np.pi * f0 * n / fs)
t = np.linspace(0, 0.1, 5000)
fp, ap = pico(x, fs)
print(f"f0 = {f0:g} Hz, fs = {fs:g} Hz, fs/2 = {fs / 2:g} Hz")
print(f"Pico do espectro em {fp:g} Hz (amplitude {ap:.3f}); previsão |f0 - fs| = {frequencia_aparente(f0, fs):g} Hz")
print("Amostras de 70 Hz iguais às de 30 Hz:", np.allclose(x, np.cos(2 * np.pi * 30 * n / fs)))

print("f0 (Hz) | prevista (Hz) | medida no espectro (Hz)")
tabela = [30, 70, 130, 170, 230, 270]
for f in tabela:
    print(f"{f:7d} | {frequencia_aparente(f, fs):13g} | {pico(np.cos(2 * np.pi * f * n / fs), fs)[0]:g}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1.5, 1, 1.1]})
ax[0].plot(t, np.cos(2 * np.pi * 70 * t), color="tab:blue", label="x(t), 70 Hz")
ax[0].plot(t, np.cos(2 * np.pi * 30 * t), "--", color="gray", label="senoide de 30 Hz")
ax[0].stem(n[:11] / fs, x[:11], linefmt="tab:red", markerfmt="o", basefmt=" ", label="amostras, fs = 100 Hz")
ax[0].set(title="As amostras de 70 Hz também pertencem a 30 Hz", xlabel="Tempo (s)", ylabel="Amplitude")
ax[0].legend(loc="lower left", fontsize=7)
a = 2 * np.abs(np.fft.fft(x)[:50]) / 100
ax[1].stem(np.arange(50) * fs / 100, a, linefmt="tab:purple", markerfmt=".", basefmt=" ")
ax[1].axvline(70, color="tab:blue", linestyle=":", label="frequência real (70 Hz, fora da faixa)")
ax[1].set(title="Espectro da sequência: pico em 30 Hz", xlabel="Frequência (Hz)", ylabel="Amplitude",
          xlim=(0, 80))
ax[1].legend(loc="upper right", fontsize=7)
fr = np.linspace(0, 300, 3001)
ax[2].plot(fr, frequencia_aparente(fr, fs), color="tab:blue")
ax[2].plot(tabela, [frequencia_aparente(f, fs) for f in tabela], "o", color="tab:red", label="casos da tabela")
for m in range(1, 4):
    ax[2].axvline(m * fs, color="gray", linestyle=":", linewidth=0.8)
ax[2].set(title="Frequência aparente em função de f0 (fs = 100 Hz)", xlabel="Frequência real f0 (Hz)",
          ylabel="Frequência aparente (Hz)")
ax[2].legend(loc="upper right", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "t8_aliasing.png"), dpi=150)
plt.show()
