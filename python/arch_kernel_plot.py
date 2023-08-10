get_ipython().magic('matplotlib inline')

from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

alpha0 = 0.5
alpha1 = 0.8

def arch_sk(x, y):
    v = np.sqrt(alpha0 + alpha1 * x**2)
    return norm.pdf(y / v) / v

left, right, bottom, top = -1.4, 1.4, -1.4, 1.4

x_grid = np.linspace(left, right, 60)
y_grid = np.linspace(bottom, top, 60)
X, Y = np.meshgrid(x_grid, y_grid)

Z = arch_sk(X, Y)

fig, ax = plt.subplots(figsize=(10, 8))

#ax.set_ylim(bottom, top)
#ax.set_xlim(left, right)

#ax.set_xticks((-0.5, 0, 0.5))
#ax.set_yticks((-0.5, 0, 0.5))

ax.contourf(X, Y, Z, 10, alpha=0.5, cmap=cm.jet)
cs = ax.contour(X, Y, Z, 10, colors='k', lw=0.5, alpha=0.5, antialias=True)
plt.clabel(cs, inline=1, fontsize=12)

ax.plot([bottom, top], [bottom, top], 'k-', lw=2, alpha=0.6)

ax.set_xlabel("$s$", fontsize=16)
ax.set_ylabel("$s'$", fontsize=16)

plt.show()



