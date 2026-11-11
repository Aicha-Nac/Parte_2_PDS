import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Parâmetros de tempo
t = np.linspace(0, 3, 1000)

# Sinal 1: x1(t) = e^(-3t) u(t) -> H1(s) = 1 / (s + 3)
num1 = [1]
den1 = [1, 3]
sys1 = signal.TransferFunction(num1, den1)
t1, y1 = signal.impulse(sys1, T=t)

# Sinal 2: h2(t) -> H2(s) = 25 / (s^2 + 6s + 25)
num2 = [25]
den2 = [1, 6, 25]
sys2 = signal.TransferFunction(num2, den2)
t2, y2 = signal.impulse(sys2, T=t)

# Plotagem dos Gráficos
plt.figure(figsize=(12, 5))

# Gráfico 1: Exemplo 1
plt.subplot(1, 2, 1)
plt.plot(t1, y1, 'b-', linewidth=2, label=r'h1(t)=e-3t')
plt.title('Exemplo 1: Polo Real s=-3')
plt.xlabel('Tempo [s]')
plt.ylabel('Resposta no Tempo')
plt.grid(True)
plt.legend()

# Gráfico 2: Exemplo 2
plt.subplot(1, 2, 2)
plt.plot(t2, y2, 'r-', linewidth=2, label=r'h2(t)=6.25e-3tsin(4t)')
plt.title('Exemplo 2: Polos Complexos Conjugados s=-3±j4')
plt.xlabel('Tempo [s]')
plt.ylabel('Resposta no Tempo')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()