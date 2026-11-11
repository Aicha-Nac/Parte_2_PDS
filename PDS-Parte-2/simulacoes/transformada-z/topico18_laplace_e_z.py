import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------- Parâmetros ----------
# Polos no plano-s escolhidos (estáveis, marginais e instáveis)
polos_s = np.array([-2, -0.5, 1,
                    -1 + 5j, -1 - 5j,
                    0 + 4j, 0 - 4j,
                    0.5 + 5j, 0.5 - 5j], dtype=complex)
Ts_lista = [0.05, 0.1, 0.3, 0.7]
Ts0 = 0.1
theta = np.linspace(0, 2 * np.pi, 400)


def legenda(fig, texto, altura=0.06):
    """Escreve 'Figura N' centralizado abaixo da imagem."""
    fig.tight_layout(rect=[0, altura, 1, 1])
    fig.text(0.5, 0.015, texto, ha='center', va='bottom',
             fontsize=11, fontweight='bold')


def s_para_z(s, Ts):
    """Mapeamento z = e^{s Ts}."""
    return np.exp(s * Ts)


def classe_s(s, tol=1e-12):
    if s.real < -tol:
        return 'estável'
    if abs(s.real) <= tol:
        return 'marginal'
    return 'instável'


cor = {'estável': 'tab:green', 'marginal': 'tab:orange', 'instável': 'tab:red'}

# ---------- Exemplo do enunciado ----------
z_ex = s_para_z(-2, Ts0)
print(f"Exemplo: s = -2, Ts = {Ts0} -> z = e^(-0.2) = {z_ex.real:.4f}, |z| = {abs(z_ex):.4f}")

# ---------- Tabela de resultados ----------
print(f"\n{'s':>12} | {'Ts':>5} | {'z':>20} | {'|z|':>7} | {'ang(z)':>8} | s-plano -> z-plano")
for Ts in Ts_lista:
    for s in polos_s:
        z = s_para_z(s, Ts)
        st = classe_s(s)
        ok = 'dentro' if abs(z) < 1 - 1e-12 else ('sobre' if abs(abs(z) - 1) <= 1e-12 else 'fora')
        print(f"{str(np.round(s, 2)):>12} | {Ts:5.2f} | {str(np.round(z, 4)):>20} | "
              f"{abs(z):7.4f} | {np.angle(z):8.3f} | {st} -> {ok} do círculo")
    print('-' * 85)

# Frequência de Nyquist para cada Ts (fronteira do aliasing)
for Ts in Ts_lista:
    print(f"Ts = {Ts:4.2f} s -> faixa primária |Im(s)| < π/Ts = {np.pi / Ts:6.2f} rad/s")

# Verificação do mapeamento inverso e do aliasing
s_teste = -1 + 5j
Ts = 0.7
z_teste = s_para_z(s_teste, Ts)
s_recuperado = np.log(z_teste) / Ts
print(f"\nAliasing: s = {s_teste}, Ts = {Ts}: ln(z)/Ts = {np.round(s_recuperado, 4)} "
      f"(ωTs = {5 * Ts:.2f} > π)")

# Comparação com aproximação de Euler (z = 1 + s Ts) para polo estável rápido
s_rapido, Ts_grande = -2.0, 1.2
print(f"Exato: z = {s_para_z(s_rapido, Ts_grande).real:.4f}   "
      f"Euler progressivo: z = {1 + s_rapido * Ts_grande:.4f} (|z|>1 -> instável)")

# ---------- Figura 1: plano-s e plano-z (Ts = 0.1) ----------
fig1, axs = plt.subplots(1, 2, figsize=(12.5, 5.8))
axs[0].axvspan(-8, 0, color='tab:green', alpha=0.10, label='semiplano esquerdo')
axs[0].axvline(0, color='k', lw=1, ls='--')
usados = set()
for s in polos_s:
    c = classe_s(s)
    axs[0].plot(s.real, s.imag, 'x', ms=11, mew=2.2, color=cor[c],
                label=c.capitalize() if c not in usados else None)
    usados.add(c)
axs[0].set(title="Plano-s", xlabel=r"σ", ylabel=r"jω",
           xlim=(-4, 2), ylim=(-7, 7))
axs[0].grid(alpha=.3); axs[0].legend(loc='lower left', fontsize=8)

axs[1].add_patch(Circle((0, 0), 1, color='tab:green', alpha=0.10))
axs[1].plot(np.cos(theta), np.sin(theta), 'k--', lw=1.2, label='círculo unitário')
usados = set()
for s in polos_s:
    c = classe_s(s)
    z = s_para_z(s, Ts0)
    axs[1].plot(z.real, z.imag, 'x', ms=11, mew=2.2, color=cor[c],
                label=c.capitalize() if c not in usados else None)
    usados.add(c)
axs[1].set(title=f"Plano-z (Ts = {Ts0} s)", xlabel="Re(z)", ylabel="Im(z)",
           aspect='equal', xlim=(-1.6, 1.6), ylim=(-1.6, 1.6))
axs[1].grid(alpha=.3); axs[1].legend(loc='lower left', fontsize=8)
legenda(fig1, "Figura 1", altura=0.06)
fig1.savefig("fig1_planos_s_z.png", dpi=150)

# ---------- Figura 2: efeito de Ts no mapeamento ----------
fig2, axs = plt.subplots(2, 2, figsize=(11.5, 10.6))
for ax, Ts in zip(axs.ravel(), Ts_lista):
    ax.add_patch(Circle((0, 0), 1, color='tab:green', alpha=0.10))
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1.1)
    for s in polos_s:
        z = s_para_z(s, Ts)
        ax.plot(z.real, z.imag, 'x', ms=10, mew=2.2, color=cor[classe_s(s)])
    ax.set(title=rf"Ts = {Ts} s  (π/Ts = {np.pi / Ts:.1f} rad/s)",
           xlabel="Re(z)", ylabel="Im(z)", aspect='equal',
           xlim=(-1.6, 2.2), ylim=(-1.6, 1.6))
    ax.grid(alpha=.3)
legenda(fig2, "Figura 2", altura=0.05)
fig2.savefig("fig2_efeito_Ts.png", dpi=150)

# ---------- Figura 3: mapeamento de retas do plano-s ----------
Ts = Ts0
w_max = np.pi / Ts                       # faixa primária: |ω| < π/Ts
w = np.linspace(-w_max, w_max, 400)
fig3, axs = plt.subplots(1, 2, figsize=(13, 6.2))
sigmas = [-15, -10, -5, -2, 0, 5]
cores_s = plt.cm.viridis(np.linspace(0, 0.9, len(sigmas)))
for sg, c in zip(sigmas, cores_s):       # retas verticais -> circunferências
    axs[0].plot([sg, sg], [-w_max, w_max], color=c, lw=1.8, label=rf"σ = {sg}")
    z = s_para_z(sg + 1j * w, Ts)
    axs[1].plot(z.real, z.imag, color=c, lw=1.8)
for wk in np.linspace(-w_max, w_max, 9)[1:-1]:   # retas horizontais -> raios
    axs[0].plot([-15, 5], [wk, wk], color='gray', lw=0.7, alpha=.7)
    sg = np.linspace(-15, 5, 200)
    z = s_para_z(sg + 1j * wk, Ts)
    axs[1].plot(z.real, z.imag, color='gray', lw=0.7, alpha=.7)
axs[0].axhline(w_max, color='k', ls=':', lw=1); axs[0].axhline(-w_max, color='k', ls=':', lw=1)
axs[0].axvline(0, color='k', ls='--', lw=1)
axs[0].set(title=rf"Plano-s: faixa primária |ω|<π/Ts = {w_max:.1f} rad/s",
           xlabel=r"σ", ylabel=r"ω (rad/s)")
axs[0].grid(alpha=.3); axs[0].legend(fontsize=8, loc='lower left')
axs[1].plot(np.cos(theta), np.sin(theta), 'k--', lw=1.5)
axs[1].set(title=rf"Plano-z: z=esTs, Ts = {Ts} s", xlabel="Re(z)", ylabel="Im(z)",
           aspect='equal', xlim=(-2, 2), ylim=(-2, 2))
axs[1].grid(alpha=.3)
legenda(fig3, "Figura 3", altura=0.06)
fig3.savefig("fig3_mapeamento_retas.png", dpi=150)

# ---------- Figura 4: sinais associados (amostragem e aliasing) ----------
fig4, axs = plt.subplots(1, 3, figsize=(15, 4.9))
n = np.arange(0, 30)

# (a) s = -2, Ts = 0.1: e^{-2t} amostrado = z^n
t = np.linspace(0, 3, 600)
axs[0].plot(t, np.exp(-2 * t), lw=1.8, label=r"e-2t")
axs[0].stem(n * 0.1, z_ex.real ** n, linefmt='C1-', markerfmt='C1o', basefmt=' ',
            label=r"zn, z=e-0.2")
axs[0].set(title=r"(a) s=-2, Ts=0,1", xlabel="t (s)")
axs[0].grid(alpha=.3); axs[0].legend(fontsize=8)

# (b) s = -1+5j, Ts = 0.1: parte real
s_c = -1 + 5j
Ts_b = 0.1
t = np.linspace(0, 3, 1500)
n_b = np.arange(0, 31)
z_c = s_para_z(s_c, Ts_b)
axs[1].plot(t, np.real(np.exp(s_c * t)), lw=1.5, label=r"Re{est}")
axs[1].stem(n_b * Ts_b, np.real(z_c ** n_b), linefmt='C1-', markerfmt='C1o', basefmt=' ',
            label=r"Re{zn}")
axs[1].set(title=rf"(b) s=-1+5j, Ts=Tsb (sem aliasing)", xlabel="t (s)")
axs[1].grid(alpha=.3); axs[1].legend(fontsize=8)

# (c) mesmo polo, Ts = 0.7: aliasing (ωTs = 3.5 > π)
Ts_c = 0.7
n_c = np.arange(0, 5)
t = np.linspace(0, 3.2, 1500)
s_alias = s_c - 1j * 2 * np.pi / Ts_c
z_c2 = s_para_z(s_c, Ts_c)
axs[2].plot(t, np.real(np.exp(s_c * t)), lw=1.5, label=r"Re{est}, ω=5")
axs[2].plot(t, np.real(np.exp(s_alias * t)), lw=1.5, ls='--',
            label=rf"alias: ω=5-2*np.pi/Tsc:.2f")
axs[2].stem(n_c * Ts_c, np.real(z_c2 ** n_c), linefmt='C2-', markerfmt='C2o', basefmt=' ',
            label=r"Re{zn}")
axs[2].set(title=rf"(c) Ts=Tsc: aliasing (ωTs = {5 * Ts_c:.1f} > π)", xlabel="t (s)")
axs[2].grid(alpha=.3); axs[2].legend(fontsize=8)
legenda(fig4, "Figura 4", altura=0.08)
fig4.savefig("fig4_sinais_aliasing.png", dpi=150)

plt.show()