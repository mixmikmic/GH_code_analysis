import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.2)
n = x.size
s = 1e-9

m = np.square(x) * 0.25

a = np.repeat(x, n).reshape(n, n)
k = np.exp(-0.5*np.square(a - a.transpose())) + s*np.identity(n)

r = np.random.multivariate_normal(m, k, 1)
y = np.reshape(r, n)

plt.plot(x,y)
plt.show()



