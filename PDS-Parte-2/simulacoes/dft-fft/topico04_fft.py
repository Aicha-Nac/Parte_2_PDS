"""Tópico 4 - DFT direta comparada com a FFT."""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

PASTA = "figuras"
os.makedirs(PASTA, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def dft_direta(x):
    """DFT pela definição: N somas de N termos cada, ou seja, da ordem de N² operações."""
    N = len(x)
    n = np.arange(N)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        X[k] = np.sum(x * np.exp(-2j * np.pi * k * n / N))
    return X


def medir(funcao, x, repeticoes):
    """Menor tempo entre várias execuções, para reduzir a influência de outros processos."""
    tempos = []
    for _ in range(repeticoes):
        inicio = time.perf_counter()
        funcao(x)
        tempos.append(time.perf_counter() - inicio)
    return min(tempos)


gerador = np.random.default_rng(1)
x = gerador.standard_normal(64)
erro = np.abs(dft_direta(x) - np.fft.fft(x)).max()
print(f"N = 64 | maior diferença entre a DFT direta e numpy.fft.fft: {erro:.1e}")

tamanhos = [64, 128, 256, 512, 1024, 2048, 4096]
t_dft, t_fft = [], []
print("    N | DFT direta (ms) | FFT (ms) | razão | N² | N log2 N")
for N in tamanhos:
    x = gerador.standard_normal(N)
    t_dft.append(medir(dft_direta, x, 3 if N <= 1024 else 1))
    t_fft.append(medir(np.fft.fft, x, 200))
    print(f"{N:5d} | {t_dft[-1] * 1e3:15.2f} | {t_fft[-1] * 1e3:8.4f} | {t_dft[-1] / t_fft[-1]:6.0f} | "
          f"{N * N:8d} | {N * np.log2(N):8.0f}")

N = np.array(tamanhos)
fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(N, np.array(t_dft) * 1e3, "o-", color="tab:red", label="DFT direta (implementação própria)")
ax.loglog(N, np.array(t_fft) * 1e3, "s-", color="tab:blue", label="FFT (numpy.fft.fft)")
ax.loglog(N, t_dft[0] * 1e3 * (N / N[0]) ** 2, "--", color="tab:red", alpha=0.5, label="referência ∝ N²")
ax.loglog(N, t_fft[0] * 1e3 * (N * np.log2(N)) / (N[0] * np.log2(N[0])), "--", color="tab:blue", alpha=0.5,
          label="referência ∝ N log2 N")
ax.set(title="Tempo de execução em função de N (escala logarítmica)", xlabel="Número de amostras N",
       ylabel="Tempo (ms)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "t4_fft_tempo.png"), dpi=150)
plt.show()
