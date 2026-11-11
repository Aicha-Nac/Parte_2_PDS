import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import lfilter

# ---------- Parâmetros ----------
N = 40
n = np.arange(N)
impulso = np.zeros(N); impulso[0] = 1
theta = np.linspace(0, 2 * np.pi, 400)


def legenda(fig, texto, altura=0.06):
    """Escreve 'Figura N' centralizado abaixo da imagem."""
    fig.tight_layout(rect=[0, altura, 1, 1])
    fig.text(0.5, 0.015, texto, ha='center', va='bottom',
             fontsize=11, fontweight='bold')


def classifica(r):
    if r < 1 - 1e-12:
        return "ESTÁVEL"
    if abs(r - 1) <= 1e-12:
        return "MARGINAL"
    return "INSTÁVEL"


# ---------- Sistemas ----------
# Polo real p:                 H(z) = 1 / (1 - p z^-1)
# Par complexo r e^{±jθ}:      H(z) = 1 / (1 - 2 r cosθ z^-1 + r² z^-2)
def sistema_real(p):
    return [1.0], [1.0, -p]


def sistema_complexo(r, th):
    return [1.0], [1.0, -2 * r * np.cos(th), r ** 2]


def resposta(b, a):
    return lfilter(b, a, impulso)


reais = [0.5, 0.9, -0.5, 1.0, 1.1]
th0 = np.pi / 4
complexos = [0.8, 1.0, 1.1]

# ---------- Tabela de resultados ----------
print(f"{'polo(s)':>22} | {'|p|':>5} | {'max|h|':>12} | {'max|h| final':>14} | classificação")
for p in reais:
    h = resposta(*sistema_real(p))
    print(f"{'p = ' + str(p):>22} | {abs(p):5.2f} | {np.max(np.abs(h)):12.4g} | "
          f"{np.max(np.abs(h[-10:])):14.4g} | {classifica(abs(p))}")
for r in complexos:
    h = resposta(*sistema_complexo(r, th0))
    print(f"{'r = ' + str(r) + ', θ = π/4':>22} | {r:5.2f} | {np.max(np.abs(h)):12.4g} | "
          f"{np.max(np.abs(h[-10:])):14.4g} | {classifica(r)}")

# ---------- Figura 1: círculo unitário e configurações de polos ----------
fig1, ax = plt.subplots(figsize=(6.4, 6.6))
ax.add_patch(Circle((0, 0), 1, color='tab:green', alpha=0.12, label='região estável'))
ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1.2, label='círculo unitário')
cor = {"ESTÁVEL": 'tab:green', "MARGINAL": 'tab:orange', "INSTÁVEL": 'tab:red'}
usados = set()
for p in reais:
    c = classifica(abs(p))
    ax.plot(p, 0, 'x', ms=11, mew=2.2, color=cor[c],
            label=c.capitalize() if c not in usados else None)
    usados.add(c)
for r in complexos:
    c = classifica(r)
    for s in (+1, -1):
        ax.plot(r * np.cos(th0), s * r * np.sin(th0), 'x', ms=11, mew=2.2, color=cor[c],
                label=c.capitalize() if c not in usados else None)
        usados.add(c)
ax.axhline(0, color='gray', lw=.5); ax.axvline(0, color='gray', lw=.5)
ax.set(title="Polos no plano-z", xlabel="Re(z)", ylabel="Im(z)", aspect='equal',
       xlim=(-1.4, 1.4), ylim=(-1.4, 1.4))
ax.grid(alpha=.3); ax.legend(loc='lower left', fontsize=8)
legenda(fig1, "Figura 1", altura=0.05)
fig1.savefig("fig1_planoz_estabilidade.png", dpi=150)

# ---------- Figura 2: polos reais e resposta temporal ----------
fig2, axs = plt.subplots(2, 3, figsize=(13, 7.4))
for ax, p in zip(axs.ravel(), reais):
    h = resposta(*sistema_real(p))
    ax.stem(n, h)
    ax.set(title=f"p = {p}  ({classifica(abs(p)).lower()})", xlabel="n", ylabel="h[n]")
    ax.grid(alpha=.3)
axs.ravel()[-1].axis('off')
legenda(fig2, "Figura 2", altura=0.05)
fig2.savefig("fig2_polos_reais.png", dpi=150)

# ---------- Figura 3: polos complexos conjugados ----------
fig3, axs = plt.subplots(2, 3, figsize=(13, 7.4),
                         gridspec_kw={'height_ratios': [1, 1.1]})
for j, r in enumerate(complexos):
    c = classifica(r)
    a0, a1 = axs[0, j], axs[1, j]
    a0.add_patch(Circle((0, 0), 1, color='tab:green', alpha=0.12))
    a0.plot(np.cos(theta), np.sin(theta), 'k--', lw=1)
    for s in (+1, -1):
        a0.plot(r * np.cos(th0), s * r * np.sin(th0), 'x', ms=11, mew=2.2, color=cor[c])
    a0.set(title=f"r = {r}, θ = π/4", xlabel="Re(z)", ylabel="Im(z)", aspect='equal',
           xlim=(-1.4, 1.4), ylim=(-1.4, 1.4))
    a0.grid(alpha=.3)

    h = resposta(*sistema_complexo(r, th0))
    env = r ** n / np.sin(th0)       # envoltória teórica
    a1.stem(n, h)
    a1.plot(n, env, 'r--', lw=1, label=r'±rn/sinθ')
    a1.plot(n, -env, 'r--', lw=1)
    a1.set(title=c.lower(), xlabel="n", ylabel="h[n]")
    a1.grid(alpha=.3); a1.legend(fontsize=8)
legenda(fig3, "Figura 3", altura=0.05)
fig3.savefig("fig3_polos_complexos.png", dpi=150)

# ---------- Figura 4: |h[N-1]| em função do módulo do polo ----------
raios = np.linspace(0.3, 1.2, 200)
saida = [abs(resposta(*sistema_real(r))[-1]) for r in raios]
fig4, ax = plt.subplots(figsize=(7, 4.8))
ax.semilogy(raios, saida, lw=2)
ax.axvline(1, color='k', ls='--', lw=1, label='|p| = 1 (fronteira)')
ax.axvspan(raios[0], 1, color='tab:green', alpha=.1)
ax.axvspan(1, raios[-1], color='tab:red', alpha=.1)
ax.set(title=f"|h[{N-1}]| em função de |p|", xlabel="|p|", ylabel=f"|h[{N-1}]|")
ax.grid(alpha=.3, which='both'); ax.legend()
legenda(fig4, "Figura 4", altura=0.08)
fig4.savefig("fig4_modulo_vs_resposta.png", dpi=150)

plt.show()