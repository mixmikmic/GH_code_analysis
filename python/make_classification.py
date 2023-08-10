from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=4400,
                           n_features=2,
                           n_redundant=0,
                           n_informative=2, 
                           n_clusters_per_class=2, 
                           class_sep=3.0)

X[:,0].shape

plt.scatter(X[:,0],X[:,1],c=y)

get_ipython().run_line_magic('whos', '')



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.decomposition import PCA

X, y = make_classification(n_samples=200,
                           n_features=4, 
                           n_informative=2, 
                           n_redundant=0, 
                           n_clusters_per_class=1, 
                           class_sep=1,
                           random_state=10)

pca = PCA(2)
X_pc = pca.fit_transform(X)

fig = plt.figure(figsize=(12,4))
fig.add_subplot(2,4,1)
plt.scatter(X[:,0],X[:,1], c=y)
fig.add_subplot(2,4,2)
plt.scatter(X[:,0],X[:,2], c=y)
fig.add_subplot(2,4,3)
plt.scatter(X[:,0],X[:,3], c=y)
fig.add_subplot(2,4,5)
plt.scatter(X[:,1],X[:,2], c=y)
fig.add_subplot(2,4,6)
plt.scatter(X[:,1],X[:,3], c=y)
fig.add_subplot(2,4,7)
plt.scatter(X[:,2],X[:,3], c=y)
fig.add_subplot(2,4,8)
plt.scatter(X_pc[:,0],X_pc[:,1], c=y)

pca.explained_variance_

pca.components_

