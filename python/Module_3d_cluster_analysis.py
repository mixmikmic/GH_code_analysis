import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (10, 6)

import ipywidgets as widgets
from ipywidgets import interact

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples = 1500
random_state = 170

X, y = make_blobs(n_samples=n_samples, random_state=random_state, 
                  cluster_std = 2, centers = 4)

plt.scatter(X[:,0],X[:,1], cmap=plt.cm.viridis)

plt.scatter(X[:,0],X[:,1], c = y, cmap=plt.cm.viridis)

get_ipython().magic('pinfo KMeans')

def cluster(iters, std):
    
    # Generate blobs
    X, y = make_blobs(n_samples=n_samples, random_state=random_state, 
                  cluster_std = std, centers = 4)
    
    # Run kmeans clustering
    kmeans = KMeans(init='random', n_clusters=4, max_iter= iters, 
                    random_state=random_state, n_init=1)
    kmeans.fit(X)

    # Mesh creation to plot the decision boundary.
    step = .02  
    x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, step), np.arange(x1_min, x1_max, step))

    # Obtain labels for each point in mesh.
    Z = kmeans.predict(np.c_[xx0.ravel(), xx1.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx0.shape)
    cs = plt.contourf(xx0, xx1, Z, cmap=plt.cm.viridis, alpha = 0.5)
    
    # Plot the original plots
    plt.scatter(X[:,0],X[:,1], c=y, cmap=plt.cm.magma)

iters_list = widgets.IntSlider(min=1, max=10, step=1, value=1)
std_list = widgets.FloatSlider(min=0.5, max=2.5, step=0.5, value=2)

interact(cluster, iters=iters_list, std=std_list)















