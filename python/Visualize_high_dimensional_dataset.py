from sklearn import datasets

# import the IRIS data
iris = datasets.load_iris()
iris_data = iris.data
Y = iris.target

print('There are %d features'%(iris_data.shape[1]))
print('There are %d classes'%(len(set(Y))))

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.style.use('seaborn-poster')

# Let's do a simple PCA and plot the first two components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(iris_data)

# plot the first two components
plt.figure(figsize = (10, 8))
plt.scatter(X_pca[:, 0], X_pca[:,1], c = Y, s = 80, linewidths=0)
plt.xlabel('First component')
plt.ylabel('Second component')

from sklearn.manifold import TSNE

X_tsne = TSNE(learning_rate=100).fit_transform(iris_data)

# plot the first two components
plt.figure(figsize = (10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:,1], c = Y, s = 80, linewidths=0)
plt.xlabel('First dimension')
plt.ylabel('Second dimension')

import pandas as pd
import numpy as np

# let's first put the data into a dataframe
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= [x[:-5] for x in iris['feature_names']] + ['target'])
df.head()

from pandas.plotting import scatter_matrix

scatter_matrix(df[df.columns[[0, 1, 2, 3]]], diagonal = 'density')

from pandas.plotting import parallel_coordinates

parallel_coordinates(df, 'target')

from pandas.plotting import andrews_curves

andrews_curves(df, 'target')

from pandas.plotting import radviz

radviz(df, 'target')

