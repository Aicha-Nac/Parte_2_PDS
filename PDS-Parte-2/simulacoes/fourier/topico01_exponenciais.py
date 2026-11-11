"""Tópico 1 - Frequência e exponenciais complexas."""
import os
import numpy as np
import matplotlib.pyplot as plt

PASTA = "figuras"
os.makedirs(PASTA, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

n = np.arange(0, 40)

# Cada linha da figura varia um único parâmetro de x[n] = A cos(w0 n + fase)
casos = [
    ("Frequência", [(1.0, np.pi / 16, 0.0), (1.0, np.pi / 8, 0.0), (1.0, np.pi / 4, 0.0)]),
    ("Fase", [(1.0, np.pi / 8, 0.0), (1.0, np.pi / 8, np.pi / 4), (1.0, np.pi / 8, np.pi / 2)]),
    ("Amplitude", [(0.5, np.pi / 8, 0.0), (1.0, np.pi / 8, 0.0), (2.0, np.pi / 8, 0.0)]),
]
rotulo_w = {np.pi / 16: "π/16", np.pi / 8: "π/8", np.pi / 4: "π/4"}
rotulo_f = {0.0: "0", np.pi / 4: "π/4", np.pi / 2: "π/2"}

fig, ax = plt.subplots(3, 3, figsize=(13, 8), sharex=True)
for linha, (nome, parametros) in enumerate(casos):
    for coluna, (A, w0, fase) in enumerate(parametros):
        x = A * np.cos(w0 * n + fase)
        eixo = ax[linha, coluna]
        eixo.stem(n, x, linefmt="tab:blue", markerfmt="o", basefmt=" ")
        eixo.set(title=f"A = {A:g}, ω0 = {rotulo_w[w0]}, φ = {rotulo_f[fase]}", ylim=(-2.3, 2.3))
        if coluna == 0:
            eixo.set_ylabel(f"{nome}\nAmplitude")
        print(f"{nome:10s} | A = {A:3.1f} | ω0 = {rotulo_w[w0]:5s} | φ = {rotulo_f[fase]:4s} | "
              f"período = {2 * np.pi / w0:4.0f} amostras | x[0] = {x[0]:6.3f} | máx = {x.max():5.2f}")
for eixo in ax[2]:
    eixo.set_xlabel("Índice n (amostras)")
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "t1_senoides.png"), dpi=150)
plt.show()

# Exponencial complexa e^{j w0 n} e a soma com e^{-j w0 n}
w0 = np.pi / 8
m = np.arange(0, 33)
z_pos = np.exp(1j * w0 * m)
z_neg = np.exp(-1j * w0 * m)
media = (z_pos + z_neg) / 2
print("Parte real de e^{jw0n} igual a cos(w0n):", np.allclose(z_pos.real, np.cos(w0 * m)))
print("Parte imaginária de e^{jw0n} igual a sen(w0n):", np.allclose(z_pos.imag, np.sin(w0 * m)))
print(f"Maior parte imaginária de (e^jw0n + e^-jw0n)/2: {np.abs(media.imag).max():.1e}")
print("(e^jw0n + e^-jw0n)/2 igual a cos(w0n):", np.allclose(media.real, np.cos(w0 * m)))

fig = plt.figure(figsize=(13, 4.2))
a1 = fig.add_subplot(1, 3, 1)
a1.stem(m, z_pos.real, linefmt="tab:blue", markerfmt="o", basefmt=" ", label="Re: cos(πn/8)")
a1.stem(m + 0.25, z_pos.imag, linefmt="tab:red", markerfmt="s", basefmt=" ", label="Im: sen(πn/8)")
a1.set(title="Partes real e imaginária de e^{jπn/8}", xlabel="Índice n (amostras)", ylabel="Amplitude")
a1.legend(loc="lower left", fontsize=8)
a2 = fig.add_subplot(1, 3, 2)
circulo = np.exp(1j * np.linspace(0, 2 * np.pi, 400))
a2.plot(circulo.real, circulo.imag, color="gray", linewidth=0.8)
a2.plot(z_pos.real[:6], z_pos.imag[:6], "o-", color="tab:blue", label="e^{+jω0n}, sentido anti-horário")
a2.plot(z_neg.real[:6], z_neg.imag[:6], "s-", color="tab:red", label="e^{−jω0n}, sentido horário")
a2.set(title="n = 0 a 5 no plano complexo", xlabel="Parte real", ylabel="Parte imaginária",
       xlim=(-1.3, 1.3), ylim=(-1.3, 1.3))
a2.set_aspect("equal")
a2.legend(loc="lower left", fontsize=7)
a3 = fig.add_subplot(1, 3, 3)
a3.stem(m, media.real, linefmt="tab:green", markerfmt="o", basefmt=" ", label="Re{(e^{jω0n} + e^{−jω0n})/2}")
a3.plot(m, media.imag, "k--", label="Im{(e^{jω0n} + e^{−jω0n})/2}")
a3.set(title="A soma das duas exponenciais é real", xlabel="Índice n (amostras)", ylabel="Amplitude")
a3.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "t1_exponencial_complexa.png"), dpi=150)
plt.show()
