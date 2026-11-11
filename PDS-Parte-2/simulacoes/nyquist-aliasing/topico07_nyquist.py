"""Tópico 7 - Amostragem de uma senoide acima, perto e abaixo de 2f0."""
import os
import numpy as np
import matplotlib.pyplot as plt

PASTA = "figuras"
os.makedirs(PASTA, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

f0, fase, DURACAO = 10.0, np.pi / 3, 2.0
t = np.linspace(0, 0.5, 5000)
x_t = np.cos(2 * np.pi * f0 * t + fase)


def espectro(x, fs):
    """Amplitude de cada componente (lado positivo), tratando o termo de Nyquist à parte."""
    N = len(x)
    X = np.fft.fft(x)
    metade = N // 2
    a = 2 * np.abs(X[:metade + 1]) / N
    a[0] /= 2
    if N % 2 == 0:
        a[metade] /= 2
    return np.arange(metade + 1) * fs / N, a


condicoes = [(100.0, "fs > 2f0"), (20.0, "fs = 2f0"), (15.0, "fs < 2f0")]
fig, ax = plt.subplots(3, 2, figsize=(13, 9), gridspec_kw={"width_ratios": [1.6, 1]})
print(f"f0 = {f0:g} Hz, fase = π/3, observação de {DURACAO:g} s")
for linha, (fs, nome) in enumerate(condicoes):
    n = np.arange(int(DURACAO * fs))
    x = np.cos(2 * np.pi * f0 * n / fs + fase)
    f, a = espectro(x, fs)
    pico = np.argmax(a)
    print(f"{nome:9s} | fs = {fs:5.1f} Hz | {fs / f0:4.1f} amostras por período | Δf = {fs / len(n):.2f} Hz | "
          f"pico em {f[pico]:5.2f} Hz com amplitude {a[pico]:.3f} | primeiras amostras {np.round(x[:4], 3)}")

    visiveis = n / fs <= 0.5
    eixo = ax[linha, 0]
    eixo.plot(t, x_t, color="tab:blue", label="x(t) contínuo, 10 Hz")
    eixo.stem(n[visiveis] / fs, x[visiveis], linefmt="tab:red", markerfmt="o", basefmt=" ", label="amostras")
    if fs < 2 * f0:
        eixo.plot(t, np.cos(2 * np.pi * (fs - f0) * t - fase), "--", color="gray", label="senoide de 5 Hz")
    eixo.set(title=f"{nome}: fs = {fs:g} Hz ({fs / f0:g} amostras por período)", ylabel="Amplitude",
             ylim=(-1.3, 1.3))
    eixo.legend(loc="upper right", fontsize=7)
    ax[linha, 1].stem(f, a, linefmt="tab:purple", markerfmt=".", basefmt=" ")
    ax[linha, 1].axvline(fs / 2, color="gray", linestyle=":", label=f"fs/2 = {fs / 2:g} Hz")
    ax[linha, 1].set(title=f"Espectro: pico em {f[pico]:g} Hz", ylabel="Amplitude", xlim=(0, 50), ylim=(0, 1.2))
    ax[linha, 1].legend(loc="upper right", fontsize=7)
ax[2, 0].set_xlabel("Tempo (s)")
ax[2, 1].set_xlabel("Frequência (Hz)")
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "t7_nyquist.png"), dpi=150)
plt.show()
