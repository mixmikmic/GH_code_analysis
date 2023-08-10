import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import homogeneity_score

from scipy.cluster.hierarchy import linkage, dendrogram

np.set_printoptions(suppress=True, precision=5)


get_ipython().magic('matplotlib inline')

X, y = make_blobs(n_samples = 150, n_features=2, 
                  centers=3, cluster_std=0.5, shuffle=True, random_state=0)

plt.scatter(X[:, 0], X[:, 1], c = "steelblue", marker = "o", s = 50)
plt.xlabel("X1")
plt.ylabel("X2")

km = KMeans(n_clusters=3, init="random", n_init = 10, 
            max_iter = 300, tol = 1e-04, random_state=0)
y_km = km.fit_predict(X)

def show_cluster(X, y, estimator = None, ignore_noise = True):
    levels = set(y)
    
    if ignore_noise and -1 in levels:
        levels.remove(-1)
    
    colors = sns.color_palette("husl", len(levels))
    centroids = None 
    if estimator is not None and hasattr(estimator, "cluster_centers_"):
        centroids = estimator.cluster_centers_  

    for k in levels:
        data = X[y == k, :]
        plt.scatter(data[:, 0], data[:, 1], color = colors[k], s = 50, label = "Cluster %s" % k)

    if not centroids is None:
        plt.scatter(centroids[:, 0], centroids[:, 1], color = "black", marker = "*", s = 150)

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(loc = "lower left")
    
show_cluster(X, y_km, km)

km.cluster_centers_

#Sum of distances of samples to their closest cluster center.
print("Distortion (Within Cluster SSE): %.2f" % km.inertia_)

#Sum of distances of samples to their closest cluster center.
homogeneity_score(y, y_km)

X, y = make_blobs(n_samples = 150, n_features=2, centers=3, 
                  cluster_std=1.0, shuffle=True, random_state=0)
km = KMeans(n_clusters=3, init="random", n_init = 10, 
            max_iter = 300, tol = 1e-04, random_state=0)
y_km = km.fit_predict(X)
print("Homogeneity score: ", homogeneity_score(y, y_km), "Inertia: ", km.inertia_)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
show_cluster(X, y, km)
plt.title("True Clusters")
plt.subplot(1, 2, 2)
show_cluster(X, y_km, km)
plt.title("Estimated clusters")

def find_elbow(X, n = 10):
    distortions = []
    for i in range(1, n):
        km = KMeans(n_clusters=i, max_iter=300, n_init=10, random_state=0, init="k-means++")
        km.fit(X)
        distortions.append(km.inertia_)
    plt.plot(range(1, n), distortions)
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Distortion")

find_elbow(X)

plt.figure(figsize = (15, 10))
row_clusters = linkage(X, method="complete", metric="euclidean")
f = dendrogram(row_clusters)

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.09, random_state=0)

plt.scatter(X[:, 0], X[:, 1], c = "steelblue", marker = "o", s = 50)
plt.xlabel("X1")
plt.ylabel("X2")

km = KMeans(n_clusters=2, init="random", n_init = 10, max_iter = 300, tol = 1e-04, random_state=0)
y_km = km.fit_predict(X)
#show_cluster(km, X, y_km)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
show_cluster(X, y)
plt.title("True Clusters")
plt.subplot(1, 2, 2)
show_cluster(X, y_km, km)
plt.title("Estimated clusters")

homogeneity_score(y, y_km)

dbscan = DBSCAN(eps=0.2, min_samples=5, metric="euclidean")
y_db = dbscan.fit_predict(X)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
show_cluster(X, y, dbscan)
plt.title("True Clusters")
plt.subplot(1, 2, 2)
show_cluster(X, y_db, dbscan)
plt.title("Estimated clusters")

labels = set(y_db)
if -1 in labels: #Noise
    labels.remove(-1)
print("No of clusters: ", len(labels))

homogeneity_score(y, y_db)

movies = pd.read_csv("data/ml-latest-small/movies.csv", index_col="movieId")
movies.head()

movies.sample(10)

movies = movies[~movies["genres"].str.contains("\(no genres listed\)")]
movies.sample(10)

genres = set()
movies["genres"].apply(lambda g: genres.update(g.split(r"|")))
genres = list(genres)
genres.sort()
print(genres, len(genres))

def to_vector(g):
    indices = np.array([genres.index(v) for v in g.split(r"|")])
    l = np.zeros(len(genres))
    l[indices] = 1
    return l

genres_idx = movies["genres"].apply(to_vector)
genres_idx.head(10)

X = np.array(genres_idx.tolist())
print("X.shape: ", X.shape)

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

plt.figure(figsize = (15, 10))
row_clusters = linkage(X_std, method="complete", metric="euclidean")
f = dendrogram(row_clusters, p = 5, truncate_mode="level")

from sklearn.decomposition import KernelPCA, PCA

pca = PCA(random_state=0)
X_pca = pca.fit_transform(X_std)

ratios = pca.explained_variance_ratio_
plt.bar(range(len(ratios)), ratios)
plt.step(range(len(ratios)), np.cumsum(ratios), 
         label = "Cumsum of Explained variance ratio")
plt.title("Explained variance")
plt.ylabel("Explained Variance Ratio")
plt.xlabel("Number of PCA components")

pca = PCA(random_state=0, n_components=2)
X_pca = pca.fit_transform(X_std)

plt.figure(figsize = (15, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("PCA1")
plt.ylabel("PCA2")

find_elbow(X_std, 40)

knn = KMeans(n_clusters=8, max_iter=300, random_state=0)
y_pred = knn.fit_predict(X_std)

def distance(p1, p2):
    p1, p2 = p1.flatten(), p2.flatten()
    return np.sqrt(np.sum((p1 - p2) ** 2))

distances = []
for i in range(X_std.shape[0]):
    p1 = X_std[i, :]
    cluster = knn.labels_[i]
    center = knn.cluster_centers_[cluster]
    distances.append(distance(p1, center))

movies["distance"] = np.array(distances)
movies.sort_values("distance", ascending=False)[:10]

movies[y_pred == 3].sample(10)



