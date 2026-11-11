"""Mini Projeto, Parte 2 - Etapas 2 a 6: análise espectral da vibração de um motor."""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window

PASTA = "figuras"
PASTA_DADOS = "dados"
os.makedirs(PASTA, exist_ok=True)
os.makedirs(PASTA_DADOS, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

# Etapa 2 - Sinal: rotação de 30 Hz e componente eletromagnética de 120 Hz (em g)
A1, F1, FASE1 = 1.0, 30.0, 0.0
A2, F2, FASE2 = 0.4, 120.0, np.pi / 4
DURACAO = 1.0


def vibracao(t):
    return A1 * np.cos(2 * np.pi * F1 * t + FASE1) + A2 * np.cos(2 * np.pi * F2 * t + FASE2)


def espectro(x, fs, janela=None):
    """Amplitude e fase de cada componente no lado positivo do espectro."""
    N = len(x)
    w = np.ones(N) if janela is None else janela
    X = np.fft.fft(x * w)
    metade = N // 2
    amplitude = 2 * np.abs(X[:metade]) / w.sum()
    amplitude[0] /= 2
    return np.arange(metade) * fs / N, amplitude, np.angle(X[:metade])


def picos(f, a, limiar=0.05):
    k = np.flatnonzero((a > limiar) & (a >= np.roll(a, 1)) & (a >= np.roll(a, -1)))
    return [(f[i], a[i]) for i in k]


t = np.arange(0, DURACAO, 1 / 100_000)
x_t = vibracao(t)
print(f"Etapa 2: x(t) = {A1} cos(2π·{F1:g}t) + {A2} cos(2π·{F2:g}t + π/4) g, observação de {DURACAO:g} s")

# Etapa 3 - Amostragem
fs = 1000.0
n = np.arange(int(DURACAO * fs))
x = vibracao(n / fs)
print(f"Etapa 3: fmax = {F2:g} Hz | Nyquist exige fs > {2 * F2:g} Hz | fs = {fs:g} Hz | Ts = {1000 / fs:g} ms | "
      f"N = {len(n)} | fs/fmax = {fs / F2:.2f}")

fig, ax = plt.subplots(2, 1, figsize=(11, 6.5))
ax[0].plot(t, x_t, color="tab:blue")
ax[0].set(title="Etapa 2: x(t) = 1,0 cos(2π·30t) + 0,4 cos(2π·120t + π/4)", xlabel="Tempo (s)",
          ylabel="Aceleração (g)", xlim=(0, 0.1))
vis = n / fs <= 0.1
ax[1].plot(t, x_t, color="tab:blue", alpha=0.4, label="x(t)")
ax[1].stem(n[vis] / fs, x[vis], linefmt="tab:red", markerfmt=".", basefmt=" ", label="x[n], fs = 1000 Hz")
ax[1].set(title="Etapa 3: amostragem com fs = 1000 Hz (Ts = 1 ms)", xlabel="Tempo (s)", ylabel="Aceleração (g)",
          xlim=(0, 0.1))
ax[1].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "mp2_1_sinal_amostragem.png"), dpi=150)
plt.show()

# Etapa 4 - Análise espectral pela FFT
f, a, fase = espectro(x, fs)
print(f"Etapa 4: Δf = fs/N = {fs / len(n):g} Hz")
for fp, ap in picos(f, a):
    k = int(round(fp * len(n) / fs))
    print(f"   pico em {fp:6.1f} Hz | amplitude {ap:.4f} g | fase {np.degrees(fase[k]):7.2f}°")
print(f"   maior amplitude fora dos picos: {np.delete(a, [30, 120]).max():.1e} g")

fig, ax = plt.subplots(3, 1, figsize=(11, 9))
ax[0].plot(n / fs, x, color="tab:blue", linewidth=0.8)
ax[0].set(title="Domínio do tempo (1 s, N = 1000 amostras)", xlabel="Tempo (s)", ylabel="Aceleração (g)")
ax[1].stem(f, a, linefmt="tab:purple", markerfmt=".", basefmt=" ")
ax[1].set(title="Espectro de magnitude (Δf = 1 Hz)", xlabel="Frequência (Hz)", ylabel="Amplitude (g)",
          xlim=(0, 200))
sig = a > 0.05
ax[2].stem(f[sig], np.degrees(fase[sig]), linefmt="tab:red", markerfmt="o", basefmt=" ")
ax[2].set(title="Espectro de fase (só nas componentes presentes)", xlabel="Frequência (Hz)", ylabel="Fase (graus)",
          xlim=(0, 200), ylim=(-10, 60))
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "mp2_2_espectro.png"), dpi=150)
plt.show()

# Etapa 5 - Aliasing: fs abaixo de 2 x 120 = 240 Hz
fig, ax = plt.subplots(3, 2, figsize=(13, 9), gridspec_kw={"width_ratios": [1.5, 1]})
for linha, fs_teste in enumerate([1000.0, 200.0, 150.0]):
    m = np.arange(int(DURACAO * fs_teste))
    xm = vibracao(m / fs_teste)
    fm, am, fasem = espectro(xm, fs_teste)
    encontrados = picos(fm, am)
    texto = ", ".join(f"{p:g} Hz ({v:.3f} g)" for p, v in encontrados)
    print(f"Etapa 5: fs = {fs_teste:6.1f} Hz | fs/2 = {fs_teste / 2:5.1f} Hz | picos: {texto}")
    vis = m / fs_teste <= 0.1
    ax[linha, 0].plot(t, x_t, color="tab:blue", alpha=0.4, label="x(t)")
    ax[linha, 0].plot(m[vis] / fs_teste, xm[vis], "o-", color="tab:red", markersize=3, linewidth=1,
                      label="amostras ligadas")
    ax[linha, 0].set(title=f"fs = {fs_teste:g} Hz", ylabel="Aceleração (g)", xlim=(0, 0.1))
    ax[linha, 0].legend(loc="upper right", fontsize=7)
    ax[linha, 1].stem(fm, am, linefmt="tab:purple", markerfmt=".", basefmt=" ")
    ax[linha, 1].axvline(fs_teste / 2, color="gray", linestyle=":", label=f"fs/2 = {fs_teste / 2:g} Hz")
    ax[linha, 1].set(title=f"Espectro: {texto}", ylabel="Amplitude (g)", xlim=(0, 200), ylim=(0, 1.5))
    ax[linha, 1].legend(loc="upper right", fontsize=7)
ax[2, 0].set_xlabel("Tempo (s)")
ax[2, 1].set_xlabel("Frequência (Hz)")
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "mp2_3_aliasing.png"), dpi=150)
plt.show()

# Etapa 6 - Janelamento: 256 amostras não contêm um número inteiro de períodos
N6 = 256
x6 = x[:N6]
print(f"Etapa 6: N = {N6} | Δf = {fs / N6:.3f} Hz | 30 Hz = {F1 / (fs / N6):.2f} bins | "
      f"120 Hz = {F2 / (fs / N6):.2f} bins")
PREENCHIMENTO = 16384
fp = np.arange(PREENCHIMENTO // 2) * fs / PREENCHIMENTO
cores = {"Retangular": "tab:red", "Hann": "tab:blue", "Hamming": "tab:green"}
fig, ax = plt.subplots(figsize=(11, 5))
for nome in cores:
    w = np.ones(N6) if nome == "Retangular" else get_window(nome.lower(), N6)
    Xp = np.abs(np.fft.fft(x6 * w, PREENCHIMENTO))[:PREENCHIMENTO // 2] * 2 / w.sum()
    fb, ab, _ = espectro(x6, fs, w)
    k30 = np.argmax(ab[(fb > 20) & (fb < 40)]) + np.flatnonzero(fb > 20)[0]
    k120 = np.argmax(ab[(fb > 110) & (fb < 130)]) + np.flatnonzero(fb > 110)[0]
    nivel75 = 20 * np.log10(Xp[np.argmin(np.abs(fp - 75))])
    nivel200 = 20 * np.log10(Xp[np.argmin(np.abs(fp - 200))])
    print(f"   {nome:10s} | 30 Hz: bin {fb[k30]:6.2f} Hz, {ab[k30]:.3f} g | 120 Hz: bin {fb[k120]:6.2f} Hz, "
          f"{ab[k120]:.3f} g | nível em 75 Hz {nivel75:6.1f} dB | em 200 Hz {nivel200:6.1f} dB")
    ax.plot(fp, 20 * np.log10(np.maximum(Xp, 1e-12)), color=cores[nome], linewidth=1, label=nome)
ax.axvline(F1, color="gray", linestyle=":")
ax.axvline(F2, color="gray", linestyle=":")
ax.set(title="Etapa 6: espectro com N = 256 (30 Hz e 120 Hz fora dos bins)", xlabel="Frequência (Hz)",
       ylabel="Amplitude (dB re 1 g)", xlim=(0, 300), ylim=(-100, 5))
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "mp2_4_janelamento.png"), dpi=150)
plt.show()

np.savetxt(os.path.join(PASTA_DADOS, "espectro_fs1000.csv"), np.column_stack([f, a, np.degrees(fase)]),
           delimiter=",", header="f_Hz,amplitude_g,fase_graus", comments="", fmt="%.6f")
