get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# create data points with 3 cluters

from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=300, centers=3,n_features=2,
                  random_state=0, cluster_std=0.50)
plt.scatter(X[:, 0], X[:, 1], s=50);

from sklearn.cluster import KMeans
km = KMeans(3)  # 3 clusters
km.fit(X)
y_kmeans = km.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=25, cmap='rainbow')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=250,marker='*')

distortion = []
for i in range(10,0,-1):
    km = KMeans(n_clusters=i)
    km.fit(X)
    distortion.append(km.inertia_)
plt.plot(range(10,0,-1),distortion, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

from sklearn.cluster import KMeans
km = KMeans(3)  # 3 clusters
km.fit(X)
y_km = km.predict(X)

from matplotlib import cm
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
s_vals = silhouette_samples(X, y_km, metric='euclidean')

y_ax_lower, y_ax_upper = 0,0
yticks = []

for i,c in enumerate(cluster_labels):
    c_s_vals = s_vals[y_km==c]
    c_s_vals.sort()
    y_ax_upper += len(c_s_vals)
    plt.barh(range(y_ax_lower,y_ax_upper),
            c_s_vals,
            height = 1, label="cluster-"+str(c))
    yticks.append((y_ax_lower+y_ax_upper)/2)
    y_ax_lower += len(c_s_vals)
plt.legend()

km = KMeans(2)  # 2 clusters
km.fit(X)
y_km = km.predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
s_vals = silhouette_samples(X, y_km, metric='euclidean')

y_ax_lower, y_ax_upper = 0,0
yticks = []

for i,c in enumerate(cluster_labels):
    c_s_vals = s_vals[y_km==c]
    c_s_vals.sort()
    y_ax_upper += len(c_s_vals)
    plt.barh(range(y_ax_lower,y_ax_upper),
            c_s_vals,
            height = 1, label="cluster-"+str(c))
    yticks.append((y_ax_lower+y_ax_upper)/2)
    y_ax_lower += len(c_s_vals)
plt.legend()

from sklearn.datasets import load_digits
digits = load_digits()

est = KMeans(n_clusters=10)
clusters = est.fit_predict(digits.data)
est.cluster_centers_.shape

fig = plt.figure(figsize=(8, 3))
for i in range(10):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.imshow(est.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

from scipy.stats import mode

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

from sklearn.decomposition import PCA

X = PCA(2).fit_transform(digits.data)

kwargs = dict(cmap = plt.cm.get_cmap('rainbow', 10),
              edgecolor='none', alpha=0.6)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=labels, **kwargs)
ax[0].set_title('learned cluster labels')

ax[1].scatter(X[:, 0], X[:, 1], c=digits.target, **kwargs)
ax[1].set_title('true labels');

from sklearn.metrics import accuracy_score
accuracy_score(digits.target, labels)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(digits.target, labels))

plt.imshow(confusion_matrix(digits.target, labels),
           cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.grid(False)
plt.ylabel('true')
plt.xlabel('predicted');

