import numpy as np
from scipy.fftpack import fft

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import matplotlib.pyplot as plt

N = 6000
T = 1.0 / 1600.0

x = np.linspace(0.0, N*T, N)
y = (
    20 * np.sin(50.0 * 2.0*np.pi*x) +
    10 * np.sin(80.0 * 2.0*np.pi*x) +
    40 * np.sin(315.0 * 2.0*np.pi*x) +
    np.random.uniform(-50., 50., size=len(x))
)

plt.figure(figsize=(15,8))
plt.scatter(x, y)
plt.grid()
plt.show()

yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

plt.figure(figsize=(15,8))
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

