get_ipython().magic('matplotlib notebook')
import seaborn
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn, randint, uniform, choice

theta = np.arange(0, .99, 0.01)
N = len(theta)
n1, z1 = 8,6
n2, z2 = 7, 2
a1, b1 = 2,2
a2, b2 = 1,1

post = np.zeros((N, N))

for c, t1 in enumerate(theta):
    for r, t2 in enumerate(theta):
        post[r, c] = t1**(z1+a1-1) * (1-t1)**(n1-z1+b1-1) * t2**(z2+a2-1) * (1-t2)**(n2-z2+b2-1)
post /= sum(sum(post))

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(theta, theta)
ax.plot_surface(X, Y, post, rstride=2, cstride=2, cmap=cm.viridis, lw=0, antialiased=False);
plt.show()

plt.figure()
plt.contour(X, Y, post);
plt.show()

