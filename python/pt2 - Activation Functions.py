import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.figsize'] = (8.5, 5)
plt.xticks([-10, 0, 10])
plt.yticks([0, 1])
plt.xlim(-10, 10)
plt.title('Sigmoid')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

plt.axhline(0, color='gray', lw=1)
plt.plot(x, sigmoid(x), label=r'$\frac{1}{1 + e^{-x}}$')
plt.legend(fontsize=19)
plt.show()

plt.xticks([-10, 0, 10])
plt.yticks([-1, 0, 1])
plt.title('tanh')
plt.xlim(-10, 10)

def tanh(x):
    return np.tanh(x)

plt.axhline(0, color='gray', lw=1)
plt.plot(x, tanh(x))
plt.show()

plt.xticks([-10, 0, 10])
plt.yticks([0, 10])
plt.title('ReLU')
plt.xlim(-10, 10)

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-10, 10, 1001)
plt.axhline(0, color='gray', lw=1)
plt.plot(x, relu(x))
plt.show()

plt.xticks([-10, 0, 10])
plt.yticks([0, 1])
plt.title('Gradients')
plt.xlim(-10, 10)

plt.plot(x, np.gradient(sigmoid(x), 0.005), label=r'$\frac{d}{dx}signmoid$')
plt.plot(x, np.gradient(tanh(x), 0.02), label=r'$\frac{d}{dx}tanh$')
plt.plot(x, np.gradient(relu(x), 0.02), label=r'$\frac{d}{dx}ReLU$')
plt.legend()
plt.show()

