# tabela atualizada

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

# Aumenta o tamanho padronizado das fontes globalmente
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11
})

t = np.linspace(0, 10, 2000)
def step(num, den, t=t):
    return signal.step(signal.TransferFunction(num, den), T=t)[1]

# 1) tau (K=1) - Figsize aumentado para (10, 6) e dpi=300
fig, ax = plt.subplots(figsize=(10, 6))
for tau in [0.5, 1, 2, 4]:
    ax.plot(t, step([1],[tau,1]), label=f"τ={tau}", linewidth=2)
    ax.axvline(tau, color="gray", lw=0.8, ls=":")
ax.axhline(0.632, color="k", lw=0.8, ls="--")
ax.set(title="1ª ordem: efeito de τ (K=1)", xlabel="t (s)", ylabel="y(t)")
ax.legend()
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig("fig1_tau.png", dpi=300)

# 2) K (tau=1)
fig, ax = plt.subplots(figsize=(10, 6))
for K in [0.5, 1, 2, 3]:
    ax.plot(t, step([K],[1,1]), label=f"K={K}", linewidth=2)
ax.set(title="1ª ordem: efeito de K (τ=1)", xlabel="t (s)", ylabel="y(t)")
ax.legend()
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig("fig2_K.png", dpi=300)

# 3) zeta (wn=2)
fig, ax = plt.subplots(figsize=(10, 6))
wn = 2
for z in [0.1, 0.3, 0.7, 1.0, 2.0]:
    ax.plot(t, step([wn**2],[1, 2*z*wn, wn**2]), label=f"ζ={z}", linewidth=2)
ax.axhline(1, color="k", lw=0.8)
ax.set(title="2ª ordem: efeito de ζ (ωn=2 rad/s)", xlabel="t (s)", ylabel="y(t)")
ax.legend()
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig("fig3_zeta.png", dpi=300)

# 4) wn (zeta=0.3)
fig, ax = plt.subplots(figsize=(10, 6))
z = 0.3
for w in [1, 2, 4, 8]:
    ax.plot(t, step([w**2],[1, 2*z*w, w**2]), label=f"ωn={w}", linewidth=2)
ax.axhline(1, color="k", lw=0.8)
ax.set(title="2ª ordem: efeito de ωn (ζ=0,3)", xlabel="t (s)", ylabel="y(t)")
ax.legend()
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig("fig4_wn.png", dpi=300)

# 5) mapa de polos
fig, ax = plt.subplots(figsize=(8, 8))
for z in [0.1, 0.3, 0.7, 1.0, 2.0]:
    p = np.roots([1, 2*z*wn, wn**2])
    ax.plot(p.real, p.imag, "x", ms=12, mew=2, label=f"ζ={z}")
ax.axhline(0, color="k", lw=.8)
ax.axvline(0, color="k", lw=.8)
ax.set(title="Polos (ωn=2)", xlabel="Re(s)", ylabel="Im(s)")
ax.legend()
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig("fig5_polos.png", dpi=300)