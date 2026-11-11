import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz

# ---------- Parâmetros ----------
polos = [0.8, 0.5, 0.9, 1.1, -0.8]
N = 30
n = np.arange(N)
impulso = np.zeros(N); impulso[0] = 1


# ---------- Função auxiliar: legenda "Figura N" abaixo da imagem ----------
def legenda(fig, texto, altura=0.06):
    """Reserva espaço na parte inferior e escreve a legenda centralizada."""
    fig.tight_layout(rect=[0, altura, 1, 1])
    fig.text(0.5, 0.015, texto, ha='center', va='bottom',
             fontsize=11, fontweight='bold')


# ---------- Resposta ao impulso (equação de diferenças) ----------
def resposta_impulso(p, N=N):
    # y[n] = p*y[n-1] + x[n]  ->  b=[1], a=[1, -p]
    return lfilter([1.0], [1.0, -p], impulso)


# ---------- Tabela de resultados ----------
print(f"{'polo':>6} | {'h[10]':>10} | {'h[29]':>10} | {'tau':>8} | {'soma|h|':>10} | estabilidade")
for p in polos:
    h = resposta_impulso(p)
    tau = -1 / np.log(abs(p)) if abs(p) != 1 else np.inf
    estab = "ESTÁVEL" if abs(p) < 1 else ("MARGINAL" if abs(p) == 1 else "INSTÁVEL")
    print(f"{p:6.2f} | {h[10]:10.4f} | {h[29]:10.4f} | {tau:8.2f} | {np.sum(np.abs(h)):10.2f} | {estab}")

theta = np.linspace(0, 2 * np.pi, 400)

# ---------- Figura 1: sistema base (p = 0.8) ----------
fig1, ax = plt.subplots(1, 2, figsize=(11, 4.6))
h = resposta_impulso(0.8)
ax[0].stem(n, h)
ax[0].set(title="Resposta ao impulso, p = 0,8", xlabel="n", ylabel="h[n]")
ax[0].grid(alpha=.3)

ax[1].plot(np.cos(theta), np.sin(theta), 'k--', lw=1)
ax[1].plot(0.8, 0, 'x', ms=12, mew=2, color='r', label='polo (0,8)')
ax[1].plot(0, 0, 'o', ms=9, mfc='none', mew=2, color='b', label='zero (0)')
ax[1].set(title="Plano-z", xlabel="Re(z)", ylabel="Im(z)", aspect='equal')
ax[1].legend(); ax[1].grid(alpha=.3)
legenda(fig1, "Figura 1")
fig1.savefig("fig1_base.png", dpi=150)

# ---------- Figura 2: comparação de polos ----------
fig2, axs = plt.subplots(2, 2, figsize=(11, 7.6))
for ax, p in zip(axs.ravel(), [0.5, 0.9, 1.1, -0.8]):
    h = resposta_impulso(p)
    ax.stem(n, h)
    ax.set(title=f"p = {p}", xlabel="n", ylabel="h[n]")
    ax.grid(alpha=.3)
legenda(fig2, "Figura 2", altura=0.05)
fig2.savefig("fig2_comparacao.png", dpi=150)

# ---------- Figura 3: plano-z com todos os polos ----------
fig3 = plt.figure(figsize=(5.5, 6))
plt.plot(np.cos(theta), np.sin(theta), 'k--', lw=1, label='círculo unitário')
cores = {0.5: 'g', 0.8: 'b', 0.9: 'orange', 1.1: 'r', -0.8: 'purple'}
for p, c in cores.items():
    plt.plot(p, 0, 'x', ms=12, mew=2, color=c, label=f'p = {p}')
plt.plot(0, 0, 'ko', mfc='none', ms=8, label='zero (0)')
plt.title("Polos no plano-z"); plt.xlabel("Re(z)"); plt.ylabel("Im(z)")
plt.gca().set_aspect('equal'); plt.grid(alpha=.3); plt.legend(fontsize=8)
legenda(fig3, "Figura 3", altura=0.07)
fig3.savefig("fig3_planoz.png", dpi=150)

# ---------- Figura 4: resposta em frequência (somente estáveis) ----------
fig4 = plt.figure(figsize=(7, 4.6))
for p in [0.5, 0.8, 0.9, -0.8]:
    w, Hf = freqz([1], [1, -p], worN=512)
    plt.plot(w / np.pi, np.abs(Hf), label=f'p = {p}')
plt.title("|H(e^{jω})|"); plt.xlabel("ω/π"); plt.ylabel("Magnitude")
plt.grid(alpha=.3); plt.legend()
legenda(fig4, "Figura 4", altura=0.08)
fig4.savefig("fig4_freq.png", dpi=150)

plt.show()