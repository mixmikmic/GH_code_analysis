import numpy as np
import matplotlib.pyplot as plt
import sys
if "../" not in sys.path:
    sys.path.append("../")
#import src.reduction
from tsap.cluster import Cluster
import matplotlib
get_ipython().magic('matplotlib inline')

# read SP500 data
SP500 = np.genfromtxt('../data/SP500array.csv', delimiter=',')
SP500 = SP500.T
nStock = len(SP500[:,0])
nTime = len(SP500[0,:])

# preprocessing, standardize data
X = np.copy(SP500)
for i in range(nStock):
    X[i,:] = (X[i,:] - np.mean(X[i,:]))/np.std(X[i,:])
    
from sklearn import manifold
DR= manifold.TSNE(n_components=2, random_state=0)
Y = DR.fit_transform(X)

model = Cluster(X)
print("The dimension of the S&P500 dataset is ("+str(model._nsample)+","+str(model._nfeature)+")" )

# run K-means

import time

start = time.time()
centroid, labels, clusters = model.kMeans(nClusters = 3)
end = time.time()

print("K-means takes "+str(end-start)+" seconds")
print(labels)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(Y[:, 0], Y[:, 1], c=labels)
plt.title('S&P 500 Stock Clustering, K-Means', fontsize=18)
plt.show()

# hierarchical clustering
start = time.time()
centroid, labels, clusters = model.H_clustering(nClusters = 3)
end = time.time()
print("Hierarchical clustering takes "+str(end-start)+" seconds")

plt.figure()
plt.scatter(Y[:, 0], Y[:, 1], c=labels)
plt.title('S&P 500 Stock Clustering, Hierarchical Clustering', fontsize=18)
plt.show()


# Spectral Clustering
start = time.time()
labels, clusters, X_embed = model.Spectral(nClusters = 3, cluster_metric = 'euclidean', sigma = 0.05 )
end = time.time()
print("Spectral clustering takes "+str(end-start)+" seconds")

plt.figure()
plt.scatter(Y[:, 0], Y[:, 1], c=labels)
plt.title('S&P 500 Stock Clustering, Spectral Clustering', fontsize=18)
plt.show()


from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, AffinityPropagation

start = time.time()
fit_KM = KMeans(n_clusters=3, init='random').fit(X)
end = time.time()
print("K-means takes "+str(end-start)+" seconds")
labels_km = fit_KM.labels_


start = time.time()
fit_SP = SpectralClustering(n_clusters = 3).fit(X)
end = time.time()
print("Spectral Clustering takes "+str(end-start)+" seconds")
labels_sp = fit_SP.labels_

start = time.time()
fit_AC = AgglomerativeClustering(n_clusters=3).fit(X)
end = time.time()
print("Agglomerative Clustering takes "+str(end-start)+" seconds")
labels_ac = fit_AC.labels_

start = time.time()
fit_AP = AffinityPropagation(damping=0.5).fit(X)
end = time.time()
print("Affinity Propagation takes "+str(end-start)+" seconds")
labels_ap = fit_AP.labels_

plt.figure()
plt.scatter(Y[:, 0], Y[:, 1], c=labels_km)
plt.title('S&P 500 Stock Clustering, K-means Clustering-Sklearn', fontsize=18)
plt.show()

plt.figure()
plt.scatter(Y[:, 0], Y[:, 1], c=labels_sp)
plt.title('S&P 500 Stock Clustering, Spectral Clustering-Sklearn', fontsize=18)
plt.show()

plt.figure()
plt.scatter(Y[:, 0], Y[:, 1], c=labels_ac)
plt.title('S&P 500 Stock Clustering, Agglomerative Clustering-Sklearn', fontsize=18)
plt.show()

plt.figure()
plt.scatter(Y[:, 0], Y[:, 1], c=labels_ap)
plt.title('S&P 500 Stock Clustering, Affinity Propagation-Sklearn', fontsize=18)
plt.show()



