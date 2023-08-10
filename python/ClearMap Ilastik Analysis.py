## After running through the same previous steps used in generating a pixel classifier, we obtain
## a numpy array showing the likelihoods of a pixel being in a given label.
get_ipython().magic('matplotlib inline')

import os
import numpy as np

good_probability = np.load("goodprobability.npy");
bad_probability = np.load("badderprobabilities.npy");

print good_probability.shape
print bad_probability.shape

## Plot "good" probability density

from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

xs = [];
ys = [];
zs = [];
the_fourth_dimension = [];

for i in range(10):
    print i
    for j in range(250):
        for k in range(250):
            xs = np.append(xs, i);
            ys = np.append(ys, j);
            zs = np.append(zs, k);
            the_fourth_dimension = np.append(the_fourth_dimension, good_probability[i, j, k]);

print "subset complete"

## Generate 5000 random points

import random

randX = [];
randY = [];
randZ = [];

for i in range(5000):
    randX = np.append(randX, random.randrange(0, 250, 1))
    randY = np.append(randY, random.randrange(0, 250, 1))
    randZ = np.append(randZ, random.randrange(0, 250, 1))

outputColors = [];
    
for j in range(5000):
    outputColors = np.append(outputColors, good_probability[randX[j], randY[j], randZ[j]])

## Plot 5000 of the random points.

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(111,projection='3d')

colors = cm.viridis_r(outputColors/max(outputColors))

colmap = cm.ScalarMappable(cmap=cm.viridis_r)
colmap.set_array(outputColors)

yg = ax.scatter(randX, randY, randZ, c=colors, marker='.')
cb = fig.colorbar(colmap)

ax.set_xlabel('X location')
ax.set_ylabel('Y location')
ax.set_zlabel('Z location')

plt.show()

## Plot the pixel likelihood histogram, showing the distribution of likelihoods.

n, bins, patches = plt.hist(outputColors, 5000, normed=1, facecolor='green', alpha=0.75)

plt.xlabel('Pixel Probability')
plt.ylabel('Frequency')
plt.title(r'$\mathrm{Histogram\ of\ Pixel\ Probabilities\ for\ 5000\ values:}\ x = [0, 250],\ y = [0, 250], z = [0, 250]$')
plt.grid(True)
plt.axis([0, 1, 0, 250])

plt.show()

print bad_probability.shape

## Generate 5000 random points for the bad example

randXBad = [];
randYBad = [];
randZBad = [];

for i in range(5000):
    randXBad = np.append(randXBad, random.randrange(0, 5, 1))
    randYBad = np.append(randYBad, random.randrange(0, 250, 1))
    randZBad = np.append(randZBad, random.randrange(0, 250, 1))

outputColorsBad = [];
    
for j in range(5000):
    outputColorsBad = np.append(outputColorsBad, bad_probability[randXBad[j], randYBad[j], randZBad[j]])

## Plot 5000 of the "bad" random points.

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(111,projection='3d')

colors = cm.viridis_r(outputColorsBad/max(outputColorsBad))

colmap = cm.ScalarMappable(cmap=cm.viridis_r)
colmap.set_array(outputColorsBad)

yg = ax.scatter(randXBad, randYBad, randZBad, c=colors, marker='.')
cb = fig.colorbar(colmap)

ax.set_xlabel('X location')
ax.set_ylabel('Y location')
ax.set_zlabel('Z location')

plt.show()

## Plot the pixel likelihood histogram, showing the distribution of likelihoods.

n, bins, patches = plt.hist(outputColorsBad, 500, normed=1, facecolor='green', alpha=0.75)

plt.xlabel('Pixel Probability')
plt.ylabel('Frequency')
plt.title(r'$\mathrm{Histogram\ of\ Pixel\ Probabilities\ for\ 500\ values}$')
plt.grid(True)

plt.show()

