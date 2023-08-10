from IPython.core.display import display, HTML
import nbimporter
import sklearn.cluster as cls
import numpy as np
from sklearn.decomposition import PCA

from sklearn.datasets.samples_generator import make_blobs

seeds = [[1, 0], [0, 1], [1, 1], [-1, -1], [-1, 0], [0, -1], [-1, 1], [1, -1]]
M, _ = make_blobs(n_samples=300, centers=seeds, cluster_std=0.3)

def visualize(matrix, axes, labels=None, p1=None, p2=None):
    if p1 is None: p1 = 0
    if p2 is None: p2 = 1
    pca = PCA(n_components=min([p2+1,matrix.shape[1]]))
    pca.fit(matrix)
    m = pca.transform(matrix)
    if labels is None:
        axes.scatter(m[:,[p1]], m[:,[p2]], alpha=0.4)
    else:
        axes.scatter(m[:,[p1]], m[:,[p2]], alpha=0.4, c=labels)

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
import numpy as np

Z = linkage(M, 'complete')
for r in Z[:20]:
    print [round(x, 2) for x in r]

from scipy.spatial.distance import pdist
import time

H, T = {}, {}
methods = ['single', 'complete', 'average', 'ward']
for method in methods:
    start = time.time()
    H[method] = linkage(M, method)
    T[method] = time.time() - start

for method, L in H.items():
    c, coph = cophenet(L, pdist(M))
    print method, c, T[method]

print H[method][0]

cols = 2
fig, axes = plt.subplots(nrows=int(np.ceil(float(len(H))/cols)), ncols=cols, figsize=(14, 10))
cl = 0
for method, L in H.items():
    axes[cl/cols, cl%cols].set_title(method)
    dendrogram(L,leaf_rotation=90.,leaf_font_size=0,ax=axes[cl/cols, cl%cols],
              color_threshold=100)
    cl += 1
plt.tight_layout()
plt.show()

cols = 2
fig, axes = plt.subplots(nrows=int(np.ceil(float(len(H))/cols)), ncols=cols, figsize=(14, 6))
cl = 0
for method, L in H.items():
    axes[cl/cols, cl%cols].set_title(method)
    dendrogram(L,leaf_rotation=90.,leaf_font_size=0,ax=axes[cl/cols, cl%cols],
              truncate_mode='lastp', p=12, show_leaf_counts=True, color_threshold=100)
    cl += 1
plt.tight_layout()
plt.show()

max_d = 40
cols = 2
fig, axes = plt.subplots(nrows=int(np.ceil(float(len(H))/cols)), ncols=cols, figsize=(14, 6))
cl = 0
for method, L in H.items():
    axes[cl/cols, cl%cols].set_title(method)
    dendrogram(L,leaf_rotation=90.,leaf_font_size=0,ax=axes[cl/cols, cl%cols],
              truncate_mode='lastp', p=12, show_leaf_counts=True, color_threshold=max_d)
    axes[cl/cols, cl%cols].axhline(y=max_d, c='k')
    cl += 1
plt.tight_layout()
plt.show()

cols = 2
fig, axes = plt.subplots(nrows=int(np.ceil(float(len(H))/cols)), ncols=cols, figsize=(14, 6))
cl = 0
for method, L in H.items():
    axes[cl/cols, cl%cols].set_title(method)
    axes[cl/cols, cl%cols].plot(L[:,2], linewidth=2.5)
    cl += 1
plt.tight_layout()
plt.show()

from scipy.cluster.hierarchy import inconsistent, maxinconsts

depth = len(H['ward'])
incons = inconsistent(H['ward'], depth)
print incons #avg, std, count, inconsistency

limits = 100
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(14, 4))
last = H['ward'][:, 2]
last_reverse = last[::-1]
idxs = np.arange(1, len(last) + 1)
axes.plot(idxs[:limits], last_reverse[:limits])

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_reverse = acceleration[::-1]
axes.plot(idxs[:limits], acceleration_reverse[:limits])
plt.show()
k = acceleration_reverse.argmax()  # if idx 0 is the max of this we want 2 clusters
print "clusters:", k, last_reverse[k]

max_d = last_reverse[k]
cols = 2
fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=(14, 6))
dendrogram(H['ward'],leaf_rotation=90.,leaf_font_size=0,ax=axes[0],
          truncate_mode='lastp', p=12, show_leaf_counts=True, color_threshold=last_reverse[k])
axes[0].axhline(y=last_reverse[k], c='k')
dendrogram(H['ward'],leaf_rotation=90.,leaf_font_size=0,ax=axes[1],
          show_leaf_counts=True, color_threshold=last_reverse[k])
axes[1].axhline(y=last_reverse[k], c='k')
plt.tight_layout()
plt.show()

from scipy.cluster.hierarchy import fcluster

cl = H['ward']
max_d = last_reverse[k]

cl_distance = fcluster(cl, max_d, criterion='distance')
cl_k = fcluster(cl, k, criterion='maxclust')
cl_incons = fcluster(cl, 5, depth=10)

cut_offs = [None, cl_distance, cl_k, cl_incons]
titles = ['Data', 'Max distance', 'Number of clusters', 'Inconsistency']
cols = 2
rows = int(np.ceil(float(len(cut_offs))/cols))
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 10))
for i, labels in enumerate(cut_offs):
    visualize(M, axes[i/cols,i%cols], labels=labels)
    axes[i/cols,i%cols].set_title(titles[i])
plt.tight_layout()
plt.show()



