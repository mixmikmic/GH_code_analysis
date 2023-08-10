rep = 7
tmp = 2.0
for i in range(1,rep,1):
    tmp = tmp / 10.0

print('The value is {}'.format(tmp))
    
for i in range(1,rep,1):
    tmp = tmp * 10.0

print('The value is {}'.format(tmp))

rep = 64
tmp = 10.0

for i in range(1,rep,1):
    tmp = tmp * 10.0

print('The value is {}'.format(tmp))
    
for i in range(1,rep,1):
    tmp = tmp / 10.0

print('The value is {}'.format(tmp))

# Use Math Library
import math
rep = 32.0
orig = 3.0
tmp = orig

tmp = tmp / math.pow(orig,rep)
print('The value is {}'.format(tmp))

tmp = tmp * math.pow(orig,rep)
print('The value is {}'.format(tmp))

# Use Math Library
import math
rep = 32.0
orig = 10.0
tmp = orig

tmp = tmp * math.pow(orig,rep)
print('The value is {}'.format(tmp))
    
tmp = tmp / math.pow(orig,rep)
print('The value is {}'.format(tmp))

# From http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#surface-plots
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

n_angles = 72 #36
n_radii = 8

# An array of radii
# Does not include radius r=0, this is to eliminate duplicate points
radii = np.linspace(0.125, 1.0, n_radii)

# An array of angles
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

# Repeat all angles for each radius
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

# Convert polar (radii, angles) coords to cartesian (x, y) coords
# (0, 0) is added here. There are no duplicate points in the (x, y) plane
x = np.append(0, (radii*np.cos(angles)).flatten())
y = np.append(0, (radii*np.sin(angles)).flatten())

# Pringle surface
# z = np.sin(-x*y)
# Self defined surface
z = (x*x - y*y)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)

plt.show()

# Write a Gredient based algorithm here:

#Implement Lagrange function algorithm


