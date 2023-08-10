name = '2017-06-02-matplotlib-contourf-subplots'
title = 'Filled contour plots and colormap normalization'
tags = 'matplotlib'
author = 'Maria Zamyatina'

from nb_tools import connect_notebook_to_post
from IPython.core.display import HTML, Image

html = connect_notebook_to_post(name, title, tags, author)

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Array 1
delta1 = 0.025
x1 = np.arange(-3.0, 3.0, delta1)
y1 = np.arange(-2.0, 2.0, delta1)
X1, Y1 = np.meshgrid(x1, y1)
Z1_1 = mlab.bivariate_normal(X1, Y1, 1.0, 1.0, 0.0, 0.0)
Z2_1 = mlab.bivariate_normal(X1, Y1, 1.5, 0.5, 1, 1)
Z1 = 10.0 * (Z2_1 - Z1_1)
# Array 2
delta2 = 0.05
x2 = np.arange(-6.0, 6.0, delta2)
y2 = np.arange(-4.0, 4.0, delta2)
X2, Y2 = np.meshgrid(x2, y2)
Z1_2 = mlab.bivariate_normal(X2, Y2, 1.0, 1.0, 0.0, 0.0)
Z2_2 = mlab.bivariate_normal(X2, Y2, 1.5, 0.5, 1, 1)
Z2 = 30.0 * (Z2_2 - Z1_2)

print(Z1.shape, Z2.shape)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
ax[0].contourf(X1, Y1, Z1)
ax[1].contourf(X2, Y2, Z2)
ax[2].contourf(X2, Y2, Z1 - Z2)
ax[0].set_title('Z1')
ax[1].set_title('Z2')
ax[2].set_title('diff')
plt.ioff()

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
p1 = ax[0].contourf(X1, Y1, Z1)
p2 = ax[1].contourf(X2, Y2, Z2)
p3 = ax[2].contourf(X2, Y2, Z1 - Z2)

fig.colorbar(p1, ax=ax[0])
fig.colorbar(p2, ax=ax[1])
fig.colorbar(p3, ax=ax[2])

ax[0].set_title('Z1')
ax[1].set_title('Z2')
ax[2].set_title('diff')
plt.ioff()

print(Z1.min(), Z1.max(), Z2.min(), Z2.max())

Z_range = np.arange( round(min(Z1.min(), Z2.min()))-1, round(max(Z1.max(), Z2.max()))+2, 1)
Z_range

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
p1 = ax[0].contourf(X1, Y1, Z1, levels=Z_range)
p2 = ax[1].contourf(X2, Y2, Z2, levels=Z_range)
p3 = ax[2].contourf(X2, Y2, Z1 - Z2)

fig.colorbar(p1, ax=ax[0])
fig.colorbar(p2, ax=ax[1])
fig.colorbar(p3, ax=ax[2])

ax[0].set_title('Z1')
ax[1].set_title('Z2')
ax[2].set_title('diff')
plt.ioff()

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
p1 = ax[0].contourf(X1, Y1, Z1, levels=Z_range)
p2 = ax[1].contourf(X2, Y2, Z2, levels=Z_range)
p3 = ax[2].contourf(X2, Y2, Z1 - Z2)

fig.colorbar(p3, ax=ax[2])

cax = fig.add_axes([0.18, 0., 0.4, 0.03])
fig.colorbar(p1, cax=cax, orientation='horizontal')

ax[0].set_title('Z1')
ax[1].set_title('Z2')
ax[2].set_title('diff')
plt.ioff()

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
p1 = ax[0].contourf(X1, Y1, Z1, levels=Z_range)
p2 = ax[1].contourf(X2, Y2, Z2, levels=Z_range)
p3 = ax[2].contourf(X2, Y2, Z1 - Z2, cmap='RdBu_r')

fig.colorbar(p3, ax=ax[2])

cax = fig.add_axes([0.18, 0., 0.4, 0.03])
fig.colorbar(p1, cax=cax, orientation='horizontal')

ax[0].set_title('Z1')
ax[1].set_title('Z2')
ax[2].set_title('diff')
plt.ioff()

import matplotlib.colors as colors

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
p1 = ax[0].contourf(X1, Y1, Z1, levels=Z_range)
p2 = ax[1].contourf(X2, Y2, Z2, levels=Z_range)
p3 = ax[2].contourf(X2, Y2, Z1 - Z2, norm=MidpointNormalize(midpoint=0.), cmap='RdBu_r')

fig.colorbar(p3, ax=ax[2])

cax = fig.add_axes([0.18, 0., 0.4, 0.03])
fig.colorbar(p1, cax=cax, orientation='horizontal')

ax[0].set_title('Z1')
ax[1].set_title('Z2')
ax[2].set_title('diff')
plt.ioff()

HTML(html)

