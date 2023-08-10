import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

c1 = np.array([1, 0])
c2 = np.array([-1, 0])

centers = np.vstack((c1, c2))


raw_dataset, labels = make_blobs(n_samples=100, centers=centers, cluster_std=0.1)
dataset = StandardScaler().fit_transform(raw_dataset)

plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='coolwarm')
plt.axvline(x=0)
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=1)
reduced_dataset = pca.fit_transform(dataset)
inverse_transformed_dataset = pca.inverse_transform(reduced_dataset)

plt.scatter(inverse_transformed_dataset[:, 0], inverse_transformed_dataset[:, 1], c=labels, cmap='coolwarm')
plt.show()

