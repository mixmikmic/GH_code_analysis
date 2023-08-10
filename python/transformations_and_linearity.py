get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

N = 200
alpha = 0.4
xmin = -2.5
xmax = 2.5

xgrid = np.linspace(xmin, xmax, N)
U = np.random.randn(N) * 0.5

def f(x):
    return alpha * x + x**3 

Y = f(xgrid) + U

fig, ax = plt.subplots()
ax.scatter(xgrid, Y)
ax.set_xlabel(r'$x$', fontsize=15)
ax.set_ylabel(r'$y$', fontsize=15)
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xgrid, xgrid**3, Y) 

ax.set_xlabel(r'$x$', fontsize=15)
ax.set_ylabel(r'$x^3$', fontsize=15)
ax.set_zlabel(r'$y$', fontsize=15)

ax.set_xticks((-2,  2))
ax.set_yticks((-15, 15))
ax.set_zticks((-15, -5, 5, 15))


xg = np.linspace(xmin, xmax, 2)
x2, y2 = np.meshgrid(xg, xg**3)
z2 = alpha * x2 + y2
ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, color='gray',
        linewidth=1, antialiased=True, alpha=0.3)
ax.view_init(elev=22.0, azim=-150)

plt.show()



