get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

digits = load_digits()
X = digits.data
y = digits.target

print y[np.argsort(y)]

plt.imshow(X[np.argsort(y),], aspect=0.02, cmap=plt.cm.gray_r)
plt.show()

pca = PCA(n_components=64).fit(X)
X_reduced =pca.transform(X)

print X_reduced.shape

plt.plot(pca.explained_variance_ratio_.cumsum())
plt.hlines(0.8, 0, 64)

plt.imshow(X_reduced[np.argsort(y),], aspect=0.02, cmap=plt.cm.gray_r)

plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y, s=50, alpha=0.7)

from sklearn.cluster import KMeans

from sklearn.datasets.samples_generator import make_classification
from sklearn.datasets import make_circles, make_blobs

KMeans()

X, y = make_blobs(n_samples=100, n_features=2, centers=[[0,0], [5,5]], random_state=1,  )
kmeans = KMeans(n_clusters=2)
y_pred = kmeans.fit_predict(X)

same = y == np.array([0 for x in range(len(y))])
plt.scatter(X[same, 0], X[same, 1], c="lightgrey", s=350, alpha=1.0, edgecolor='None')
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=100, alpha=0.75, edgecolor='none')

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=6, n_clusters_per_class=1)
kmeans = KMeans(n_clusters=2)
y_pred = kmeans.fit_predict(X)

same = y == np.array([0 for x in range(len(y))])
plt.scatter(X[same, 0], X[same, 1], c="lightgrey", s=350, alpha=1.0, edgecolor='None')
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=100, alpha=0.75, edgecolor='none')

X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

kmeans = KMeans(n_clusters=2)
y_pred = kmeans.fit_predict(X)

same = y == np.array([0 for x in range(len(y))])
plt.scatter(X[same, 0], X[same, 1], c="lightgrey", s=350, alpha=1.0, edgecolor='None')
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=100, alpha=0.75, edgecolor='none')

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=6, n_clusters_per_class=1)
kmeans = KMeans(n_clusters=4)
y_pred = kmeans.fit_predict(X)

same = y == np.array([0 for x in range(len(y))])
plt.scatter(X[same, 0], X[same, 1], c="lightgrey", s=350, alpha=1.0, edgecolor='None')
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=100, alpha=0.75, edgecolor='none')



