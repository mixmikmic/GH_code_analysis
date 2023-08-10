get_ipython().run_line_magic('run', '"Lith Affinity Prop PCA.ipynb"')

# The above runs notebook setup as well

import scipy
from scipy.cluster import hierarchy

# from sklearn.manifold import MDS

# dis_sim = 1 - word_matrix.values
# mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
# pos = mds.fit_transform(dis_sim)

# threshold = 0.1
# linkage = hierarchy.linkage(pos, method="centroid")
# clusters = hierarchy.fcluster(linkage, threshold, criterion="distance")
# dendro = hierarchy.dendrogram(linkage, labels=names, orientation='right')
#                               # truncate_mode='level', p=10) # last two settings tell it to only show the last 'p'

from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=5)  # , whiten=True
pos = pca.fit_transform(word_matrix.values)

xs, ys = pos[:, 0], pos[:, 1]
names = lith_desc.tolist()

threshold = 0.1
linkage = hierarchy.linkage(pos, method="centroid")
clusters = hierarchy.fcluster(linkage, threshold, criterion="distance")
dendro = hierarchy.dendrogram(linkage, labels=names, orientation='right',
                              truncate_mode='level', p=3) # last two settings tell it to only show the last 'p'

plt.figure(figsize=(10, 8))
plt.scatter(xs, ys, c=clusters, cmap='Set1')  # plot points with cluster dependent colors
plt.show()

len(np.unique(clusters))



