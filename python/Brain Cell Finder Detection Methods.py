## Following https://github.com/log0/build-your-own-meanshift/blob/master/Meanshift%20In%202D.ipynb
import math
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from sklearn.datasets import *
get_ipython().magic('matplotlib inline')
pylab.rcParams['figure.figsize'] = 16, 12

## Generate points in 3D space, with std. dev 1.5 ("bad" example)
from mpl_toolkits.mplot3d import Axes3D
original_X, X_shapes = make_blobs(100, 3, centers=6, cluster_std=1.5)
print(original_X.shape)

## Show 2D cross section.
plt.plot(original_X[:,0], original_X[:,1], 'bo', markersize = 10)

## Plot points in 3D space.

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(111,projection='3d')

yg = ax.scatter(original_X[:,0], original_X[:,1], original_X[:,2], marker='.')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

## Use Euclidean Distance for evaluating nearest neighbor.  Use Gaussian Kernel for the kernel density estimates.

def euclid_distance(x, xi):
    return np.sqrt(np.sum((x - xi)**2))

def neighbourhood_points(X, x_centroid, distance = 5):
    eligible_X = []
    for x in X:
        distance_between = euclid_distance(x, x_centroid)
        # print('Evaluating: [%s vs %s] yield dist=%.2f' % (x, x_centroid, distance_between))
        if distance_between <= distance:
            eligible_X.append(x)
    return eligible_X

def gaussian_kernel(distance, bandwidth):
    val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((distance / bandwidth))**2)
    return val

from sklearn import cluster
sklearn.cluster.estimate_bandwidth(original_X, quantile=0.3, n_samples=None, random_state=0)

## Define window and kernel bandwidth.  The kernel bandwidth is estimated using
## sklearn.cluster.estimate_bandwidth

look_distance = 5  # How far to look for neighbours.
kernel_bandwidth = 7  # Kernel parameter.

X = np.copy(original_X)
# print('Initial X: ', X)

past_X = []
n_iterations = 5
for it in range(n_iterations):
    # print('Iteration [%d]' % (it))    

    for i, x in enumerate(X):
        ### Step 1. For each datapoint x ∈ X, find the neighbouring points N(x) of x.
        neighbours = neighbourhood_points(X, x, look_distance)
        # print('[%s] has neighbours [%d]' % (x, len(neighbours)))
        
        ### Step 2. For each datapoint x ∈ X, calculate the mean shift m(x).
        numerator = 0
        denominator = 0
        for neighbour in neighbours:
            distance = euclid_distance(neighbour, x)
            weight = gaussian_kernel(distance, kernel_bandwidth)
            numerator += (weight * neighbour)
            denominator += weight
        
        new_x = numerator / denominator
        
        ### Step 3. For each datapoint x ∈ X, update x ← m(x).
        X[i] = new_x
    
    # print('New X: ', X)
    past_X.append(np.copy(X))

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(111,projection='3d')

yg = ax.scatter(past_X[4][:,0], past_X[4][:,1], past_X[4][:,2], 'bo', marker='.')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

## Generate points in 3D, with low standard deviation ("good" example)
original_X_good, X_shapes = make_blobs(100, 3, centers=6, cluster_std=0.3)
plt.plot(original_X_good[:,0], original_X_good[:,1], 'bo', markersize = 10)

sklearn.cluster.estimate_bandwidth(original_X_good, quantile=0.3, n_samples=None, random_state=0)

## Plot points in 3D space.

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(111,projection='3d')

yg = ax.scatter(original_X_good[:,0], original_X_good[:,1], original_X_good[:,2], marker='.')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

## Define window and kernel bandwidth.  The kernel bandwidth is estimated using
## sklearn.cluster.estimate_bandwidth

look_distance = 5  # How far to look for neighbours.
kernel_bandwidth = 8  # Kernel parameter.

X = np.copy(original_X_good)
# print('Initial X: ', X)

past_X = []
n_iterations = 5
for it in range(n_iterations):
    # print('Iteration [%d]' % (it))    

    for i, x in enumerate(X):
        ### Step 1. For each datapoint x ∈ X, find the neighbouring points N(x) of x.
        neighbours = neighbourhood_points(X, x, look_distance)
        # print('[%s] has neighbours [%d]' % (x, len(neighbours)))
        
        ### Step 2. For each datapoint x ∈ X, calculate the mean shift m(x).
        numerator = 0
        denominator = 0
        for neighbour in neighbours:
            distance = euclid_distance(neighbour, x)
            weight = gaussian_kernel(distance, kernel_bandwidth)
            numerator += (weight * neighbour)
            denominator += weight
        
        new_x = numerator / denominator
        
        ### Step 3. For each datapoint x ∈ X, update x ← m(x).
        X[i] = new_x
    
    # print('New X: ', X)
    past_X.append(np.copy(X))

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(111,projection='3d')

yg = ax.scatter(past_X[4][:,0], past_X[4][:,1], past_X[4][:,2], 'bo', marker='.')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

