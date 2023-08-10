import sys
print(sys.version)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import time

import pandas as pd
import seaborn as sns

from scipy.spatial.distance import pdist

import sys
sys.path.append('../code/')

from rbf_kernel import RBFKernel
from scipy.spatial.distance import squareform

#X = np.array([[5., 1, 2], [0., 1, 2], [1, 4, 6], [0, 2, 4], [1., 1, 1]])
X = np.array([[5., 1], [9., 1], [1, 4]])

k = RBFKernel(X)

k.sigma

k.sigma is None

k.transform_vector(X[0,:])

class chickenchicken:
    def __init__(self, X, kernel=RBFKernel):
        self.kernel = kernel(X)
        

c = chickenchicken(X)

c.kernel

k.transform(X)

k.sigma = 0.0001

k.transform(X)

X

np.concatenate((X, X))

def so(X, sigma):
    # pdist to calculate the squared Euclidean distances for every pair of points
    # in the 100x2 dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Variance of the Euclidean distance between all pairs of data points.
    variance = np.var(sq_dists)

    # squareform to converts the pairwise distances into a symmetric 100x100 matrix
    mat_sq_dists = squareform(sq_dists)

    # Compute the 100x100 kernel matrix
    K = np.exp(-1/(2*sigma**2) * mat_sq_dists)
    
    return K

np.multiply(X, X)

so(X, 4)

X - X[0,:]

np.linalg.norm(X - X[0,:], axis=1)

np.linalg.norm(X - X[0,:], axis=1)/(-2.)/1**2

np.exp(np.linalg.norm(X - X[0,:], axis=1)/(-2.)/1**2)

np.sum(X, axis=0)

pdist(X)



squareform(pdist(X))



