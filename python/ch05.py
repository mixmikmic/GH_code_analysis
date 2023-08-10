# declare libraries used by this notebook
import math

import numpy as np
from numpy.linalg import inv
from numpy.linalg import det

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')

# point inside a 3d box

x = np.arange(31) * 5 / 30
y = np.arange(31) * 5 / 30
z = np.arange(31) * 5 / 30

# randomly choose a point in 3-space from a normal distribution (mean=2.5, sd=0.5)
rbar1 = np.random.normal(2.5, 0.5, 3)

# build 3d grid of probability density function
rbar2 = np.array([(2.5, 2.5, 2.5)])
sd = 0.5
C = sd**2 * np.eye(3)
CI = inv(C)
DC = det(C)
norm = (2 * math.pi)**(3/2) * DC**0.5

# calculate PDF probabilities
PP = np.zeros((31,31,31))
for i in range(0,31):
    for j in range(0,31):
        for k in range(0,31):
            r = np.array([(i * 5/30, j * 5/30, k * 5/30)])
            PP[i,j,k] = math.exp((-0.5 * (r-rbar2))[0].dot(CI.dot((r-rbar2).reshape(3,1))))/norm

# threshold PDF probability (PP) volume as a triangulated isosurface
verts, faces = measure.marching_cubes(PP, np.max(PP)/10, spacing=(5/30, 5/30, 5/30))
            
# build figure
plt.subplots(1, 2, figsize=(15, 7))

# randomly chosen point graph
ax = plt.subplot(121, projection='3d')
ax.scatter(rbar1[0], rbar1[1], rbar1[2])
plt.plot(x,y,z, color='black')
ax.set_xlim3d(0, 5)
ax.set_ylim3d(0, 5)
ax.set_zlim3d(0, 5)
ax.view_init(10, 225)
plt.title('Realization of a random variable')

# PDF as a spherical cloud graph
ax = plt.subplot(122, projection='3d')
plt.plot(x,y,z, color='black')
ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], cmap='Spectral', lw=0)
ax.set_xlim3d(0, 5)
ax.set_ylim3d(0, 5)
ax.set_zlim3d(0, 5)
ax.view_init(10, 225)
plt.title('Probability density function as spherical cloud')

# random data
N = 100
d = np.random.normal(2.5, 1.5, N)

# grid
m = 1.5 + np.arange(51) * 3.5 / 50
s = 1 + np.arange(51) / 50
m, s = np.meshgrid(m,s)

L = np.zeros((51,51))
for i in range(0,51):
    for j in range(0,51):
        L[i, j] = -100 * math.log(2*math.pi)/2 - 100 * math.log(s[i,j]) - 0.5 * sum(np.square(d-m[i,j])) / s[i,j]**2

imax = np.unravel_index(np.argmax(L), L.shape)
lmax = np.max(L)
        
# build figure
plt.subplots(1, 1, figsize=(10, 7))

# maximum likelihood surface graph
ax = plt.subplot(111, projection='3d')
surf = ax.plot_surface(m, s, L, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.text(m[imax], s[imax], lmax+50, 'Maximum Likelihood', None)
ax.scatter(m[imax], s[imax], lmax+35, color='black', s=100)
ax.set_xlim3d(0, 5)
ax.set_ylim3d(0, 5)
ax.set_zlim3d(-500, -100)
ax.view_init(35, 345)
ax.set_xlabel(r'$m_1$')
ax.set_ylabel(r'$\sigma$')
ax.set_zlabel(r'$L(m_1,\sigma$')
plt.title('Likelihood surface for random variables')

