from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150,
                 n_features=2,
                 centers=3,
                 cluster_std=0.5,
                 shuffle=True,
                 random_state=0)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.scatter(X[:,0],
            X[:,1],
            c='white',
            marker='o',
            s=50)
plt.grid()
plt.show()
## 150 random generated points roughly grouped into three regions

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,
           init='random',
           n_init=10,
           max_iter=300,
           tol=1e-04,
           random_state=0)
y_km = km.fit_predict(X)

## visualize clusters and centroids
plt.scatter(X[y_km==0,0],
            X[y_km==0,1],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1')
plt.scatter(X[y_km==1,0],
            X[y_km==1,1],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2')
plt.scatter(X[y_km==2,0],
            X[y_km==2,1],
            s=50,
            c='lightblue',
            marker='v',
            label='cluster 3')
plt.scatter(km.cluster_centers_[:,0],
            km.cluster_centers_[:,1],
            s=250,
            marker='*',
            c='red',
            label='centroids')
plt.legend()
plt.grid()
plt.show()

## within cluster SSE (distortion) for comparing performance of 
## different k-means clusterings; accesible via inertia_ attribute 
## from k-means model
print('Distortion: %.2f' % km.inertia_)

## ploting the distortions to illustrate elbow method meaning
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
               init='k-means++',
               n_init=10,
               max_iter=300,
               random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
## can tell the elbow is located at k=3
## providing evidence that k=3 is good choice for this dataset

## crreat plot of the silhouette coefficients
km = KMeans(n_clusters=3,
           init='k-means++',
           n_init=10,
           max_iter=300,
           tol=1e-04,
           random_state=0)
y_km = km.fit_predict(X)

import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km,
                                    metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), 
             c_silhouette_vals, 
             height=1.0, 
             edgecolor='none', 
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
            color="red", 
            linestyle="--") 
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()
## shows good clustering!

## looking at relatively bad clusting now, here with two centroids
km = KMeans(n_clusters=2, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

plt.scatter(X[y_km==0,0], 
            X[y_km==0,1], 
            s=50, c='lightgreen', 
            marker='s', 
            label='cluster 1')
plt.scatter(X[y_km==1,0], 
            X[y_km==1,1], 
            s=50, 
            c='orange', 
            marker='o', 
            label='cluster 2')
plt.scatter(km.cluster_centers_[:,0], 
            km.cluster_centers_[:,1], 
            s=250, 
            marker='*', 
            c='red', 
            label='centroids')
plt.legend()
plt.grid()
plt.show()

## now looking at a silhouette plot of to evaluate these results
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, 
                                     y_km, 
                                     metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), 
             c_silhouette_vals, 
             height=1.0, 
             edgecolor='none', 
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()
## visibly different lengths and widths, evidence for suboptimal
## clustering

## agglomerative clustering
## generate some random data to work with
## rows -> observations (IDs 0-4) 
## cols -> different features(X,Y,Z)
import pandas as pd
import numpy as np
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0','ID_1','ID_2','ID_3','ID_4']
X = np.random.random_sample([5,3])*10
df = pd.DataFrame(X, columns=variables, index=labels)
df

## computing the distance matrix
## use pdist from scipy's spatial.distance
from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
                       columns=labels, index=labels)
row_dist

## apply complete linkage agglomeration to clusters
## using linkage from scipy's cluster.hierarchy (returns linkage matrix)
from scipy.cluster.hierarchy import linkage
help(linkage)

## INCORRECT approach - using squareform distance matrix
from scipy.cluster.hierarchy import linkage
# incorrect way
# row_clusters = linkage(row_dist, method='complete',
#                       metric='euclidead')
## CORRECT approaches
# use condensed distance matrix from pdist
row_clusters = linkage(pdist(df, metric='euclidean'),
                      method='complete')
## OR
# use input sample matrix
row_clusters = linkage(df.values, method='complete',
                      metric='euclidean')

## look at cluster results
pd.DataFrame(row_clusters,
            columns=['row label 1',
                    'row label 2',
                    'distance',
                    'no. of items in clust.'],
            index=['cluster %d' % (i+1) for i in
                  range(row_clusters.shape[0])])
## first and second columns denote most dissimilar memebers in cluster
## thrid col distance between, third members in each cluster

## see ^^^ results in dendrogram
from scipy.cluster.hierarchy import dendrogram
# make dendrogram black (part 1/2)
from scipy.cluster.hierarchy import set_link_color_palette
set_link_color_palette(['black'])
row_dendr = dendrogram(row_clusters, 
                      labels=labels,
#                       make dendrogram black (part 2/2)
                      color_threshold=np.inf
                      )
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()

# Step 1: create new 'figure' object define x,y axis position, 
#         width and height of dendrogram via 'add_axes'. 
#         rotate dendrogram 90degrees counter-clockwise:

fig = plt.figure(figsize=(8,8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='left')

# Step 2: we reorder our data in initial DataFrame according 
#         to clustering labels accessed from the dendrogram object 
#         (which is essentially a dict) via leaves key

df_rowclust = df.ix[row_dendr['leaves'][::-1]]

# Step 3: construct heat map from reodered DataFrame and 
#         position it next to dedrogram:

axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest',
                 cmap='hot_r')

# Step 4: lastly we modify aesthetics of heat map removing axis 
#         ticks and hiding axis spines. Also add color bar and 
#         assign the feature and sample names to the x and y tick 
#         labels, respectively:

axd.set_xticks([])
axd.set_yticks([])

for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))

# plt.savefig('./figures/heatmap.png', dpi=300)
plt.show()

## Applying agglomerative clustering in scikit-learn
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2,
                            affinity='euclidean',
                            linkage='complete')
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)

## illustrative example with new half-moon-shaped structure dataset
## to compare k-means clsustering, hierarchical clustering and DBSCAN
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200,
                 noise=0.05,
                 random_state=0)
plt.scatter(X[:,0], X[:,1])
plt.show()

## k-means and complete linkage
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
km = KMeans(n_clusters=2,
           random_state=0)
y_km = km.fit_predict(X)
ax1.scatter(X[y_km==0, 0],
            X[y_km==0, 1],
            c='lightblue',
            marker='o',
            s=40,
            label='cluster 1')
ax1.scatter(X[y_km==1, 0],
            X[y_km==1, 1],
            c='red',
            marker='s',
            s=40,
            label='cluster 2')
ax1.set_title('K-means clustering')

ac = AgglomerativeClustering(n_clusters=2,
                             affinity='euclidean',
                             linkage='complete')
y_ac = ac.fit_predict(X)
ax2.scatter(X[y_km==0, 0],
           X[y_km==0, 1],
           c='lightblue',
           marker='o',
           s=40,
           label='cluster 1')
ax2.scatter(X[y_km==1, 0],
           X[y_km==1, 1],
           c='red',
           marker='s',
           s=40,
           label='cluster2')
ax2.set_title('Agglomerative clustering')
plt.legend()
plt.show()

## DBSCAN
### see if can find 2 half moons, using density-based approach
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2,
            min_samples=5,
            metric='euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db==0, 0],
           X[y_db==0, 1],
           c='lightblue',
           marker='o',
           s=40,
           label='cluster 1')
plt.scatter(X[y_db==1, 0],
           X[y_db==1, 1],
           c='red',
           marker='s',
           s=40,
           label='cluster2')
plt.legend()
plt.show()



