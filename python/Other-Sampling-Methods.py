get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

cauchy = lambda x: 1./np.pi/(1+x**2)

xvals = np.linspace(-100, 100, 10000)
plt.plot(xvals, cauchy(xvals), label='$Cauchy(0,1)$')
plt.legend(loc='upper right', shadow=True)
plt.show()

y = np.random.uniform(0, 1, 1000)
x = np.tan(np.pi * (y - 0.5))
plt.hist(x, normed=1, bins=1000, label='bins')
plt.plot(xvals, cauchy(xvals), label='$Cauchy(0,1)$')
plt.xlim(-100, 100)
plt.legend(loc='upper right', shadow=True)
plt.show()

exponential = lambda x: np.exp(-x)

xvals = np.linspace(0, 10, 100)
plt.plot(xvals, exponential(xvals), label='$Exp(1)$')
plt.legend(loc='upper right', shadow=True)
plt.xlim(0, 10)
plt.show()

y = np.random.uniform(0, 1, 100000)
x = -np.log((1 - y))
plt.hist(x, normed=1, bins=100, label='bins')
plt.plot(xvals, exponential(xvals), label='$Exp(1)$')
plt.xlim(0, 10)
plt.legend(loc='upper right', shadow=True)
plt.show()

gumbel = lambda x: np.exp(-(x + np.exp(-x)))

xvals = np.linspace(0, 10, 100)
plt.plot(xvals, gumbel(xvals), label='$Gumbel(0,1)$')
plt.legend(loc='upper right', shadow=True)
plt.xlim(0, 10)
plt.show()

y = np.random.uniform(0, 1, 100000)
x = -np.log(-np.log(y))
plt.hist(x, normed=1, bins=100, label='bins')
plt.plot(xvals, gumbel(xvals), label='$Gumbel(0,1)$')
plt.xlim(0, 10)
plt.legend(loc='upper right', shadow=True)
plt.show()



