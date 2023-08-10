# Import all required libraries
from __future__ import division # For python 2.*

import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

np.random.seed(0)
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(1)
X,Y = ml.datagen.data_GMM(500, 3, get_Z=True) # Random data distribution 

plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()

n_clusters = 3
Z, mu, ssd = ml.cluster.kmeans(X, K=n_clusters, init='k++', max_iter=100)

mu

fig, ax = plt.subplots(1, 2, figsize=(20, 8))
# Plotting the original data
ax[0].scatter(X[:, 0], X[:, 1], c=Y)

# Plotting the clustered data
ax[1].scatter(X[:, 0], X[:, 1], c=Z) # Plotting the data
ax[1].scatter(mu[:, 0], mu[:, 1], s=500, marker='x', facecolor='black', lw=8) # Plotting the centroids
ax[1].scatter(mu[:, 0], mu[:, 1], s=30000, alpha=.45, c=np.unique(Z)) # Lazy way of plotting the clusters area :)

plt.show()

cluster_KNN = ml.knn.knnClassify(mu, np.arange(n_clusters), 1)
c = cluster_KNN.predict(X)

fig, ax = plt.subplots(1, 2, figsize=(20, 8))
# Plotting the clustered data
ax[0].scatter(X[:, 0], X[:, 1], c=Z) # Plotting the data
ax[0].scatter(mu[:, 0], mu[:, 1], s=500, marker='x', facecolor='black', lw=8) # Plotting the centroids
ax[0].scatter(mu[:, 0], mu[:, 1], s=30000, alpha=.45, c=np.unique(Z)) # Lazy way of plotting the clusters area :)

ax[1].scatter(X[:, 0], X[:, 1], c=c) # Plotting the data

plt.show()

X = np.genfromtxt("data/faces.txt", delimiter=None) # load face dataset 

X[0]

X.shape

f, ax = plt.subplots(3, 5, figsize=(17, 13))
ax = ax.flatten()

# Plotting a random 15 faces
for j in range(15):
    i = np.random.randint(X.shape[0])
    img = np.reshape(X[i,:],(24,24))  # reshape flattened data into a 24*24 patch
    
    # We've seen the imshow method in the previous discussion :)
    ax[j].imshow( img.T , cmap="gray")
    
plt.show()

n_clusters = 10
Zi, mui, ssdi = ml.cluster.kmeans(X, K=n_clusters, init='k++')

f, ax = plt.subplots(2, 5, figsize=(17, 8))
ax = ax.flatten()
for i in range(min(len(ax), n_clusters)):
    img = np.reshape(mui[i,:] ,(24, 24))
    ax[i].imshow(img.T , cmap="gray")
    
plt.show()



