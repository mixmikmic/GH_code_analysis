import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=1)
plt.scatter(X[:, 0], X[:, 1], c=y)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=8).fit(X)
plt.scatter(X[:, 0], X[:, 1], c=km.labels_)

from sklearn.metrics.cluster import silhouette_score
silhouette_score(X, km.labels_)

km = KMeans(n_clusters=2).fit(X)
silhouette_score(X, km.labels_)

scores = []

for n_clusters in range(2, 10):
    km = KMeans(n_clusters=n_clusters).fit(X)
    scores.append(silhouette_score(X, km.labels_))

plt.plot(range(2, 10), scores)

X, y = make_blobs(random_state=101, centers=5)
plt.scatter(X[:, 0], X[:, 1], c=y)

scores = []

for n_clusters in range(2, 10):
    km = KMeans(n_clusters=n_clusters).fit(X)
    scores.append(silhouette_score(X, km.labels_))

plt.plot(range(2, 10), scores)

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=500, noise=.1, random_state=0)
plt.scatter(X[:, 0], X[:, 1], c=y)

from sklearn.cluster import SpectralClustering
sc = SpectralClustering(gamma=1, n_clusters=2).fit(X)
plt.scatter(X[:, 0], X[:, 1], c=sc.labels_)

scores = []

for gamma in np.logspace(-3, 3, 7):
    sc = SpectralClustering(n_clusters=2, gamma=gamma).fit(X)
    scores.append(silhouette_score(X, sc.labels_))
    
plt.plot(scores)
plt.xticks(range(len(scores)), np.logspace(-3, 3, 7));

sc = SpectralClustering(gamma=50, n_clusters=2).fit(X)
plt.scatter(X[:, 0], X[:, 1], c=sc.labels_)

from sklearn.metrics import adjusted_rand_score

scores_ari = []

for gamma in np.logspace(-3, 3, 7):
    sc = SpectralClustering(n_clusters=2, gamma=gamma).fit(X)
    scores_ari.append(adjusted_rand_score(sc.labels_, y))

plt.plot(scores_ari, label="adjusted rand score")
plt.plot(scores, label="silhouette score")
plt.xticks(range(len(scores)), np.logspace(-3, 3, 7))
plt.legend(loc="best")

sc = SpectralClustering(gamma=100, n_clusters=2).fit(X)
plt.scatter(X[:, 0], X[:, 1], c=sc.labels_)



