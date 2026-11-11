import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import freqz, tf2zpk

# ---------- Sistema escolhido ----------
# H(z) = (1 - z^-2) / (1 - 0.9 z^-1 + 0.81 z^-2)
#      = (z^2 - 1) / (z^2 - 0.9 z + 0.81)
# Zeros: z = +1 e z = -1      Polos: z = 0.9 e^{±jπ/3}
b = np.array([1.0, 0.0, -1.0])
a = np.array([1.0, -0.9, 0.81])
zeros, polos, ganho = tf2zpk(b, a)

theta = np.linspace(0, 2 * np.pi, 400)


def legenda(fig, texto, altura=0.06):
    """Escreve 'Figura N' centralizado abaixo da imagem."""
    fig.tight_layout(rect=[0, altura, 1, 1])
    fig.text(0.5, 0.015, texto, ha='center', va='bottom',
             fontsize=11, fontweight='bold')


def H_z(z):
    """Avalia H(z) diretamente em qualquer ponto do plano-z."""
    return np.polyval(b, z) / np.polyval(a, z)


# ---------- Resposta em frequência: duas formas equivalentes ----------
w, H_freqz = freqz(b, a, worN=1024)          # DTFT via scipy
H_circ = H_z(np.exp(1j * w))                 # H(z) em z = e^{jω}
erro = np.max(np.abs(H_freqz - H_circ))

mag = np.abs(H_circ)
i_pico = np.argmax(mag)

print("Zeros:", np.round(zeros, 4))
print("Polos:", np.round(polos, 4), " |p| =", np.round(np.abs(polos), 4))
print(f"Erro máx. |freqz - H(e^jw)| = {erro:.2e}")
print(f"Pico: |H| = {mag[i_pico]:.3f} em ω = {w[i_pico]/np.pi:.3f}π rad/amostra")
print(f"|H(e^j0)| = {abs(H_z(1.0)):.3e}   |H(e^jπ)| = {abs(H_z(-1.0)):.3e}")
for k in (0.2, 0.5, 1.0 / 3, 0.8):
    print(f"  |H| em ω = {k:.3f}π: {abs(H_z(np.exp(1j*k*np.pi))):.3f}")

# ---------- Figura 1: polos, zeros e círculo unitário ----------
fig1, ax = plt.subplots(figsize=(6.4, 6.6))
ax.add_patch(Circle((0, 0), 1, color='tab:blue', alpha=0.07))
ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1.2, label='círculo unitário |z| = 1')
ax.plot(zeros.real, zeros.imag, 'o', ms=10, mfc='none', mew=2, color='tab:blue', label='zeros')
ax.plot(polos.real, polos.imag, 'x', ms=11, mew=2.2, color='tab:red', label='polos')
w0 = np.pi / 5                                # ponto de exemplo e^{jω0}
ax.plot(np.cos(w0), np.sin(w0), 's', color='tab:green', ms=7, label=r'z=ejω0')
ax.plot([0, np.cos(w0)], [0, np.sin(w0)], color='tab:green', lw=1)
ax.annotate(r'ω0', (0.25, 0.07), color='tab:green')
ax.axhline(0, color='gray', lw=.5); ax.axvline(0, color='gray', lw=.5)
ax.set(title="Polos e zeros de H(z)", xlabel="Re(z)", ylabel="Im(z)", aspect='equal',
       xlim=(-1.4, 1.4), ylim=(-1.4, 1.4))
ax.grid(alpha=.3); ax.legend(loc='lower left', fontsize=8)
legenda(fig1, "Figura 1", altura=0.05)
fig1.savefig("fig1_polos_zeros.png", dpi=150)

# ---------- Figura 2: magnitude e fase sobre o círculo unitário ----------
fig2, axs = plt.subplots(1, 2, figsize=(12, 4.8))
axs[0].plot(w / np.pi, mag, lw=2)
axs[0].axvline(np.angle(polos[0]) / np.pi, color='tab:red', ls=':', label='ângulo dos polos')
axs[0].plot([0, 1], [0, 0], 'o', mfc='none', mew=2, color='tab:blue', label='zeros (ω = 0 e π)')
axs[0].set(title=r"|H(ejω)|", xlabel=r"ω/π", ylabel="Magnitude")
axs[0].grid(alpha=.3); axs[0].legend(fontsize=8)
axs[1].plot(w / np.pi, np.unwrap(np.angle(H_circ)) / np.pi, lw=2, color='tab:purple')
axs[1].set(title=r"Fase de H(ejω)", xlabel=r"ω/π", ylabel="Fase (×π rad)")
axs[1].grid(alpha=.3)
legenda(fig2, "Figura 2", altura=0.07)
fig2.savefig("fig2_resposta_freq.png", dpi=150)

# ---------- Figura 3: |H(z)| em todo o plano-z e o corte no círculo ----------
x = np.linspace(-1.5, 1.5, 500)
X, Y = np.meshgrid(x, x)
Z = X + 1j * Y
with np.errstate(divide='ignore', invalid='ignore'):
    mapa = 20 * np.log10(np.abs(H_z(Z)))
mapa = np.clip(mapa, -40, 30)

fig3, axs = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={'width_ratios': [1.1, 1]})
im = axs[0].pcolormesh(X, Y, mapa, shading='auto', cmap='viridis')
axs[0].plot(np.cos(theta), np.sin(theta), 'w--', lw=1.5)
axs[0].plot(zeros.real, zeros.imag, 'o', mfc='none', mew=2, ms=9, color='cyan')
axs[0].plot(polos.real, polos.imag, 'x', mew=2.2, ms=10, color='red')
axs[0].set(title=r"20log10|H(z)| no plano-z", xlabel="Re(z)", ylabel="Im(z)", aspect='equal')
fig3.colorbar(im, ax=axs[0], label="dB", fraction=0.046, pad=0.04)

axs[1].plot(w / np.pi, 20 * np.log10(np.maximum(mag, 1e-6)), lw=2)
axs[1].set(title=r"Corte ao longo de |z|=1", xlabel=r"ω/π",
           ylabel="Magnitude (dB)", ylim=(-40, 30))
axs[1].grid(alpha=.3)
legenda(fig3, "Figura 3", altura=0.06)
fig3.savefig("fig3_mapa_plano_z.png", dpi=150)

# ---------- Figura 4: interpretação geométrica por distâncias ----------
# |H(e^jω)| = |K| · Π|e^jω - z_k| / Π|e^jω - p_k|
ej = np.exp(1j * w)
dist_z = np.abs(ej[:, None] - zeros[None, :])
dist_p = np.abs(ej[:, None] - polos[None, :])
mag_geo = np.abs(ganho) * np.prod(dist_z, axis=1) / np.prod(dist_p, axis=1)
print(f"Erro máx. |H| geométrico vs. direto = {np.max(np.abs(mag_geo - mag)):.2e}")

fig4, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for k in range(len(zeros)):
    axs[0].plot(w / np.pi, dist_z[:, k], label=f"distância ao zero {np.round(zeros[k], 2)}")
for k in range(0, len(polos), 2):             # um polo do par conjugado basta
    axs[0].plot(w / np.pi, dist_p[:, k], '--', lw=2, color='tab:red',
                label=f"distância ao polo {np.round(polos[k], 2)}")
axs[0].set(title="Distâncias de $e^{j\\omega}$ aos zeros e polos",
           xlabel=r"ω/π", ylabel="Distância")
axs[0].grid(alpha=.3); axs[0].legend(fontsize=7)
axs[1].plot(w / np.pi, mag, lw=3, alpha=.5, label=r"|H(ejω)| direto")
axs[1].plot(w / np.pi, mag_geo, 'k--', lw=1.2, label="produto de distâncias")
axs[1].set(title="Verificação geométrica", xlabel=r"ω/π", ylabel="Magnitude")
axs[1].grid(alpha=.3); axs[1].legend(fontsize=8)
legenda(fig4, "Figura 4", altura=0.07)
fig4.savefig("fig4_distancias.png", dpi=150)

plt.show()