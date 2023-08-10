get_ipython().run_line_magic('matplotlib', 'inline')

#Import Libraries

from sklearn import datasets, cluster
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Random seed fixes the random 
# values the clustering starts 
# with, resulting in same centroid
np.random.seed(2)

# load data
iris = datasets.load_iris()

X_iris = iris.data
X_iris

# do the clustering
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X_iris)
labels = k_means.labels_

# plot the clusters in color
fig = plt.figure(1, figsize=(8, 8))
# specifies a figure in which the graph will be plotted
plt.clf()
ax = Axes3D(fig)
plt.cla()

ax.scatter(X_iris[:, 3], X_iris[:, 0], X_iris[:, 2], c=labels.astype(np.float))

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')

plt.show()

