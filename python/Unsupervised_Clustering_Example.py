get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1,2],
              [1.5,1.8],
              [5,8],
              [8,8],
              [1,1.06],
              [9,11]]
            )

plt.scatter(X[:,0], X[:,1], s=50)
plt.title("Data Points (n=6)")
plt.show()

clf = KMeans(n_clusters=2, init='k-means++')
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

centroids

labels

colors = ['r.','g.']

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=15)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=50, linewidth=4)
plt.title('Data Points and the 2 Centroids')
plt.show()

