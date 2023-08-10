get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_s_curve
X, y = make_s_curve(n_samples=1000)

from mpl_toolkits.mplot3d import Axes3D
ax = plt.axes(projection='3d')

ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y)
ax.view_init(10, -60)

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)

from sklearn.manifold import Isomap

iso = Isomap(n_neighbors=20)
iso.fit(X)
X_iso = iso.transform(X)
plt.scatter(X_iso[:, 0], X_iso[:, 1], c=y)

from sklearn.manifold import TSNE

tsne = TSNE()
tsne.fit(X)

X_tsne = tsne.transform(X)

X_tsne = tsne.embedding_

X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)

from sklearn.datasets import load_digits
digits = load_digits(n_class=5)
X, y = digits.data, digits.target

X_tsne = TSNE().fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)

