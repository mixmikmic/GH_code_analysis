import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
from IPython import display
import pylab as pl
get_ipython().magic('matplotlib inline')

# prepare data and assign labels according to the targetConcept
d = 3
theta = float(d) / 2
targetConcept = np.array([0,0,1])
# generate data
data = np.array([[0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1]])
# shuffle the ordering of data points
np.random.shuffle(data)
labels = np.any(data*np.tile(targetConcept,[8,1]),1)
# initialize w
w = np.ones([3,1])

# show the points and labels
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[labels,0], data[labels,1], data[labels,2], c='w', marker='o')
ax.scatter(data[~labels,0], data[~labels,1], data[~labels,2], c='w', marker='s')
# show initial surface
xx, yy = np.meshgrid(range(0,2), range(0,2))
zz = (-w[0] * xx - w[1] * yy + theta) * 1. /w[2]
#plt3d = plt.figure().gca(projection='3d')
ax.plot_surface(xx, yy, zz,linewidth=0,alpha=1)
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_zticks([0,1])

# consider each training example and update the decision surface
for i in range(8):
    p = np.dot(data[i,:],w) > theta
    if(p != labels[i]):
        # a mistake - update w
        if(labels[i]):
            w[data[i,:] == 1] = 2*w[data[i,:] == 1]
        else:
            w[data[i,:] == 1] = 0

# show the points and labels
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[labels,0], data[labels,1], data[labels,2], c='w', marker='o')
ax.scatter(data[~labels,0], data[~labels,1], data[~labels,2], c='w', marker='s')
# show the learnt surface
xx, yy = np.meshgrid(range(0,2), range(0,2))
zz = (-w[0] * xx - w[1] * yy + theta) * 1. /w[2]
#plt3d = plt.figure().gca(projection='3d')
ax.plot_surface(xx, yy, zz,linewidth=0,alpha=1)
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_zticks([0,1])

