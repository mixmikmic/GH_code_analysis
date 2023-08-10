import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets

# Initialize initial three cluster centers randomly
centers = [[1, 1], [-1, -1], [1, -1]]
# Load iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Initialize k-means
k_means = KMeans(n_clusters=3) 
# Fit the k-means by passing it the input
k_means.fit(X)
# Retrieve labels (cluster ids) for all the inputs
labels = k_means.labels_

# Verify the output
print (len(labels))
print (labels[0])

# Plotting the clusters in 3-D using Axes3D library
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
# 'elev' stores the elevation angle in the z plane 
# 'azim' stores the azimuth angle in the x,y plane
ax = Axes3D(fig, elev=50, azim=135)
plt.cla()#clear axes
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float)) # Plotting just three columns, because we can't plot 4!
ax.w_xaxis.set_ticklabels([]) # Removes markings along the axes
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
plt.show()

# Change number of clusters to 5 and repeat
k_means = KMeans(n_clusters=5) 
k_means.fit(X)
labels = k_means.labels_
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
plt.show()

