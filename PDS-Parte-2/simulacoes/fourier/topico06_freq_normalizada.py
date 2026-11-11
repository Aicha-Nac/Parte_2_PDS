"""Tópico 6 - Conversões entre hertz, ciclos por amostra e radianos por amostra."""
import os
import numpy as np
import matplotlib.pyplot as plt

PASTA = "figuras"
os.makedirs(PASTA, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def principal(w):
    """Leva uma frequência angular para o intervalo (-π, π]."""
    return np.pi - np.mod(np.pi - w, 2 * np.pi)


casos = [(50, 1000), (120, 1000), (250, 1000), (500, 1000), (440, 8000), (440, 44100), (1000, 8000), (70, 100)]
print("f (Hz) | fs (Hz) | fn (ciclos/amostra) | ω (rad/amostra) | ω/π | amostras por período | ω em (-π, π]")
for f, fs in casos:
    fn = f / fs
    w = 2 * np.pi * fn
    print(f"{f:6d} | {fs:7d} | {fn:19.4f} | {w:15.4f} | {w / np.pi:5.3f} | {fs / f:20.2f} | "
          f"{principal(w):7.4f} ({principal(w) / np.pi:+.3f}π)")

# A mesma frequência normalizada gera a mesma sequência, qualquer que seja fs
n = np.arange(0, 41)
xa = np.cos(2 * np.pi * 50 / 1000 * n)
xb = np.cos(2 * np.pi * 100 / 2000 * n)
print("50 Hz com fs = 1000 Hz e 100 Hz com fs = 2000 Hz geram a mesma sequência:", np.allclose(xa, xb))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].stem(n, xa, linefmt="tab:blue", markerfmt="o", basefmt=" ", label="f = 50 Hz, fs = 1000 Hz")
ax[0].plot(n, xb, "x", color="tab:red", markersize=8, label="f = 100 Hz, fs = 2000 Hz")
ax[0].set(title="Mesma fn = 0,05 ciclo/amostra: sequências idênticas", xlabel="Índice n (amostras)",
          ylabel="Amplitude")
ax[0].legend(loc="lower left", fontsize=8)
fs = 1000
f = np.linspace(0, 1000, 2001)
ax[1].plot(f, 2 * np.pi * f / fs / np.pi, "--", color="gray", label="ω = 2πf/fs")
ax[1].plot(f, principal(2 * np.pi * f / fs) / np.pi, color="tab:blue", label="ω equivalente em (−π, π]")
ax[1].axvline(fs / 2, color="tab:red", linestyle=":", label="fs/2 = 500 Hz")
ax[1].set(title="Frequência física e frequência angular discreta (fs = 1000 Hz)", xlabel="Frequência f (Hz)",
          ylabel="ω/π (rad/amostra, normalizada por π)")
ax[1].legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "t6_freq_normalizada.png"), dpi=150)
plt.show()
