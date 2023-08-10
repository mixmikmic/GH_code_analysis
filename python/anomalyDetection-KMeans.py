get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from sklearn import cluster

X_svd = np.loadtxt('../dataset/X_svd.txt', delimiter=',')
X_svd_3 = np.loadtxt('../dataset/X_svd_3.txt', delimiter=',')

k_means = cluster.MiniBatchKMeans(n_clusters=20)
k_means.fit_transform(X_svd_3)
labels = k_means.labels_
centroids = k_means.cluster_centers_

fig1 = plt.figure(figsize=(18, 18))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(X_svd_3[:,0], X_svd_3[:,1], X_svd_3[:,2], s=25, alpha=0.5, c=k_means.labels_, edgecolor='w')

