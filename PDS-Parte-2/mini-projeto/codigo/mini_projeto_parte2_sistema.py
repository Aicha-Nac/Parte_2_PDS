"""Mini Projeto, Parte 2 - Etapas 7 a 9: base de fixação do motor (segunda ordem) nos planos s e z."""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

PASTA = "figuras"
os.makedirs(PASTA, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

# Etapa 7 - Modelo contínuo: base de fixação com frequência natural igual à rotação (30 Hz)
fn, zeta = 30.0, 0.1
wn = 2 * np.pi * fn
num_s = [wn ** 2]
den_s = [1, 2 * zeta * wn, wn ** 2]
H_s = signal.TransferFunction(num_s, den_s)
polos_s = np.roots(den_s)
zeros_s = np.roots(num_s)
wd = wn * np.sqrt(1 - zeta ** 2)
print("Etapa 7: H(s) = ωn² / (s² + 2ζωn s + ωn²)")
print(f"   fn = {fn:g} Hz | ωn = {wn:.2f} rad/s | ζ = {zeta:g} | 2ζωn = {2 * zeta * wn:.3f} | ωn² = {wn ** 2:.1f}")
print(f"   polos: {np.round(polos_s, 2)} | zeros: nenhum")
print(f"   frequência amortecida = {wd / (2 * np.pi):.2f} Hz | constante de tempo 1/(ζωn) = {1 / (zeta * wn) * 1000:.1f} ms | "
      f"tempo de acomodação 4/(ζωn) = {4 / (zeta * wn):.3f} s")

t = np.linspace(0, 1.0, 20001)
_, y_degrau = signal.step(H_s, T=t)
_, y_impulso = signal.impulse(H_s, T=t)
print(f"   degrau: pico {y_degrau.max():.3f} em t = {t[np.argmax(y_degrau)] * 1000:.1f} ms | valor final {y_degrau[-1]:.3f}")
for f in (30.0, 120.0):
    _, resp = signal.freqresp(H_s, w=[2 * np.pi * f])
    print(f"   |H(j2π·{f:g})| = {abs(resp[0]):.4f} ({abs(resp[0]) * 100:.1f}% transmitido)")

# Resposta do suporte à vibração do motor das Etapas 2 a 6
x_motor = 1.0 * np.cos(2 * np.pi * 30 * t) + 0.4 * np.cos(2 * np.pi * 120 * t + np.pi / 4)
_, y_motor, _ = signal.lsim(H_s, U=x_motor, T=t)
regime = t >= 0.6
print(f"   vibração do motor: pico de entrada {np.abs(x_motor[regime]).max():.3f} g | "
      f"pico transmitido em regime {np.abs(y_motor[regime]).max():.3f} g")

fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
ax[0, 0].axvspan(-600, 0, color="tab:green", alpha=0.08, label="semiplano esquerdo (estável)")
ax[0, 0].plot(polos_s.real, polos_s.imag, "x", color="tab:red", markersize=11, mew=2, label="polos")
ax[0, 0].axhline(0, color="black", linewidth=0.6)
ax[0, 0].axvline(0, color="black", linewidth=0.6)
ax[0, 0].set(title="Plano-s: polos em −18,85 ± j187,55 (sem zeros)", xlabel="Re{s} = σ (1/s)",
             ylabel="Im{s} = ω (rad/s)", xlim=(-60, 20), ylim=(-260, 260))
ax[0, 0].legend(loc="lower left", fontsize=8)
ax[0, 1].plot(t, y_degrau, color="tab:blue", label="resposta ao degrau")
ax[0, 1].plot(t, np.exp(-zeta * wn * t) + 1, "--", color="gray", linewidth=0.8, label="envoltória 1 ± e^(−ζωn t)")
ax[0, 1].plot(t, 1 - np.exp(-zeta * wn * t), "--", color="gray", linewidth=0.8)
ax[0, 1].set(title="Resposta ao degrau unitário", xlabel="Tempo (s)", ylabel="Saída (adimensional)", xlim=(0, 0.3))
ax[0, 1].legend(loc="upper right", fontsize=8)
ax[1, 0].plot(t, y_impulso, color="tab:purple")
ax[1, 0].set(title="Resposta ao impulso h(t)", xlabel="Tempo (s)", ylabel="h(t) (1/s)", xlim=(0, 0.3))
ax[1, 1].plot(t, x_motor, color="tab:red", alpha=0.5, linewidth=0.8, label="vibração do motor (entrada)")
ax[1, 1].plot(t, y_motor, color="tab:blue", linewidth=1.2, label="vibração da base (saída)")
ax[1, 1].set(title="Resposta da base à vibração do motor (ressonância em 30 Hz)", xlabel="Tempo (s)", ylabel="Aceleração (g)",
             xlim=(0.6, 0.7))
ax[1, 1].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "mp2_5_modelo_continuo.png"), dpi=150)
plt.show()

# Etapa 8 - Modelo discreto com o mesmo Ts da aquisição (1 ms), por segurador de ordem zero (ZOH)
Ts = 1e-3
num_z, den_z, _ = signal.cont2discrete((num_s, den_s), Ts, method="zoh")
num_z = np.trim_zeros(np.ravel(num_z), "f")
polos_z = np.roots(den_z)
zeros_z = np.roots(num_z)
print(f"Etapa 8: Ts = {Ts * 1000:g} ms | B(z) = {np.round(num_z, 6)} | A(z) = {np.round(den_z, 6)}")
print(f"   polos: {np.round(polos_z, 5)} | |p| = {np.abs(polos_z).round(5)} | ângulo = {np.angle(polos_z).round(5)} rad")
print(f"   zero: {np.round(zeros_z, 5)} | estável: {bool(np.all(np.abs(polos_z) < 1))}")

n = np.arange(0, 600)
H_z = signal.dlti(num_z, den_z, dt=Ts)
_, (h_n,) = signal.dimpulse(H_z, n=len(n))
_, (y_dz,) = signal.dstep(H_z, n=len(n))
_, y_cont = signal.step(H_s, T=n * Ts)
print(f"   h[n]: máx |h| = {np.abs(h_n).max():.4f} | |h[599]| = {abs(h_n[-1, 0]):.2e} | soma |h| = {np.abs(h_n).sum():.3f}")
print(f"   degrau discreto x contínuo nos instantes nTs: diferença máxima = {np.abs(y_dz[:, 0] - y_cont).max():.1e}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.3, 1.3]})
circ = np.exp(1j * np.linspace(0, 2 * np.pi, 400))
ax[0].plot(circ.real, circ.imag, color="gray", linewidth=1, label="círculo unitário")
ax[0].plot(polos_z.real, polos_z.imag, "x", color="tab:red", markersize=10, mew=2, label="polos")
ax[0].plot(zeros_z.real, zeros_z.imag, "o", mfc="none", color="tab:blue", markersize=9, mew=2, label="zero")
ax[0].set(title="Plano-z (Ts = 1 ms)", xlabel="Re{z}", ylabel="Im{z}", xlim=(-1.3, 1.3), ylim=(-1.3, 1.3))
ax[0].set_aspect("equal")
ax[0].legend(loc="lower left", fontsize=7)
ins = ax[0].inset_axes([0.03, 0.6, 0.37, 0.37])
ins.plot(circ.real, circ.imag, color="gray", linewidth=1)
ins.plot(polos_z.real, polos_z.imag, "x", color="tab:red", markersize=8, mew=2)
ins.set(xlim=(0.90, 1.02), ylim=(-0.25, 0.25), title="detalhe perto de z = 1", xticks=[], yticks=[])
ins.title.set_fontsize(7)
ax[1].stem(n[:250], h_n[:250, 0], linefmt="tab:purple", markerfmt=" ", basefmt=" ")
ax[1].set(title="Resposta ao impulso h[n] do modelo discreto", xlabel="Índice n (Ts = 1 ms)", ylabel="h[n]")
ax[2].plot(n * Ts, y_cont, color="tab:blue", label="contínuo, y(nTs)")
ax[2].plot(n * Ts, y_dz[:, 0], "--", color="tab:red", label="discreto, y[n]")
ax[2].set(title="Degrau: modelo contínuo e discreto", xlabel="Tempo (s)", ylabel="Saída", xlim=(0, 0.3))
ax[2].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "mp2_6_modelo_discreto.png"), dpi=150)
plt.show()

# Etapa 9 - Relação entre os planos: z = e^{s Ts}
print("Etapa 9: mapeamento z = e^(s Ts) do polo s = -18,85 + j187,55")
p = polos_s[np.argmax(polos_s.imag)]
casos_Ts = [1e-3, 5e-3, 10e-3, 20e-3]
for T in casos_Ts:
    z = np.exp(p * T)
    print(f"   Ts = {T * 1000:4.0f} ms | z = {z.real:+.4f} {z.imag:+.4f}j | |z| = e^(σTs) = {abs(z):.4f} | "
          f"ângulo = ωTs = {np.angle(z):+.4f} rad ({np.degrees(np.angle(z)):+.1f}°) | ωTs/π = {p.imag * T / np.pi:.3f}")
z_exato = np.exp(polos_s * Ts)
print(f"   e^(s Ts) com Ts = 1 ms: {np.round(np.sort_complex(z_exato), 6)}")
print(f"   polos do modelo ZOH:     {np.round(np.sort_complex(polos_z), 6)}")
print(f"   diferença máxima: {np.abs(np.sort_complex(z_exato) - np.sort_complex(polos_z)).max():.1e}")
print(f"   zero criado pela discretização ZOH (o contínuo não tem zeros): {zeros_z[0].real:.4f}")
print(f"   Ts máximo sem aliasing do polo: π/ωd = {np.pi / wd * 1000:.1f} ms")

fig, ax = plt.subplots(1, 2, figsize=(13, 7))
ax[0].axvspan(-40, 0, color="tab:green", alpha=0.1, label="semiplano esquerdo, Re{s} < 0")
ax[0].axvline(0, color="black", linewidth=1, label="eixo jω (fronteira)")
ax[0].plot(polos_s.real, polos_s.imag, "x", color="tab:red", markersize=11, mew=2, label="polos da base")
ax[0].plot([10, 10], [187.55, -187.55], "x", color="tab:orange", markersize=9, mew=2, label="polo instável de referência (σ = +10)")
ax[0].set(title="Plano-s", xlabel="σ (1/s)", ylabel="ω (rad/s)", xlim=(-40, 25), ylim=(-260, 260))
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
ax[1].fill(circ.real, circ.imag, color="tab:green", alpha=0.1, label="interior do círculo, |z| < 1")
ax[1].plot(circ.real, circ.imag, color="black", linewidth=1, label="círculo unitário (imagem do eixo jω)")
marc = {1e-3: "o", 5e-3: "s", 10e-3: "^", 20e-3: "D"}
for T in casos_Ts:
    zz = np.exp(polos_s * T)
    ax[1].plot(zz.real, zz.imag, marc[T], color="tab:red", mfc="none", markersize=8, mew=1.5,
               label=f"polos da base, Ts = {T * 1000:g} ms")
zi = np.exp(np.array([10 + 187.55j, 10 - 187.55j]) * 5e-3)
ax[1].plot(zi.real, zi.imag, "x", color="tab:orange", markersize=9, mew=2, label="polo instável, Ts = 5 ms")
ax[1].set(title="Plano-z: z = e^(s·Ts)", xlabel="Re{z}", ylabel="Im{z}", xlim=(-1.4, 1.4), ylim=(-1.4, 1.4))
ax[1].set_aspect("equal")
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "mp2_7_planos_s_z.png"), dpi=150)
plt.show()
