get_ipython().magic('matplotlib notebook')

# numbers
import numpy as np
import pandas as pd

# stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# utils
import os, re
from pprint import pprint

## No learning, no testing.
#testing_df = pd.read_csv('data/optdigits/optdigits.tes',header=None)
#X_testing,  y_testing  = testing_df.loc[:,0:63],  testing_df.loc[:,64]

training_df = pd.read_csv('data/optdigits/optdigits.tra',header=None)
X_training, y_training = training_df.loc[:,0:63], training_df.loc[:,64]

print X_training.shape
print y_training.shape

def get_normed_mean_cov(X):
    X_std = StandardScaler().fit_transform(X)
    X_mean = np.mean(X_std, axis=0)
    
    ## Automatic:
    #X_cov = np.cov(X_std.T)
    
    # Manual:
    X_cov = (X_std - X_mean).T.dot((X_std - X_mean)) / (X_std.shape[0]-1)
    
    return X_std, X_mean, X_cov

X_std, X_mean, X_cov = get_normed_mean_cov(X_training)

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

n_samples = 300
similarities = euclidean_distances(X_std[:n_samples])

#mds = manifold.MDS(n_components=2, max_iter=1000, eps=1e-3,
#                   n_jobs=1)
#pos = mds.fit(similarities).embedding_

def get_cmap(n):
    #colorz = plt.cm.cool
    colorz = plt.get_cmap('Set1')
    return[ colorz(float(i)/n) for i in range(n)]

colorz = get_cmap(10)
colors = [colorz[yy] for yy in y_training]

#fig = plt.figure(figsize=(4,4))
#plt.scatter(pos[:,0],pos[:,1],c=colors)
#plt.show()

mds2 = manifold.MDS(n_components=2, max_iter=1000, eps=1e-3,
                   dissimilarity="precomputed", n_jobs=1)
pos2 = mds2.fit(similarities).embedding_

print pos2.shape

fig = plt.figure(figsize=(4,4))
plt.scatter(pos2[:,0],pos2[:,1],c=colors)
plt.show()

n_samples = 400
similarities = euclidean_distances(X_std[:n_samples])

mds3 = manifold.MDS(n_components=3, max_iter=1000, eps=1e-4,
                   dissimilarity="precomputed", n_jobs=1)
pos3 = mds3.fit(similarities).embedding_

print pos3.shape

def get_cmap(n):
    #colorz = plt.cm.cool
    colorz = plt.get_cmap('Set1')
    return[ colorz(float(i)/n) for i in range(n)]

colorz = get_cmap(10)
colors = [colorz[yy] for yy in y_training[:n_samples]]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(5,5))
ax1 = fig.add_subplot(111, projection='3d')

ax1.scatter(pos3[:,0],pos3[:,1],pos3[:,2],c=colors)
    
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

plt.show()

n_samples = 1000
similarities = euclidean_distances(X_std[:n_samples])

print X_std.shape

# XXXXXXXXXXXXXXXXXXXXXXXXX
# This takes a while.
# XXXXXXXXXXXXXXXXXXXXXXXXX
mds4 = manifold.MDS(n_components=4, max_iter=500, eps=1e-3,
                   dissimilarity="precomputed", n_jobs=1)
fit4 = mds4.fit(similarities)
pos4 = fit4.embedding_

def get_cmap(n):
    #colorz = plt.cm.cool
    colorz = plt.get_cmap('Set1')
    return[ colorz(float(i)/n) for i in range(n)]

colorz = get_cmap(10)
colors = [colorz[yy] for yy in y_training[:n_samples]]

fig = plt.figure(figsize=(14,6))
ax1, ax2 = [fig.add_subplot(120 + i + 1) for i in range(2)]

ax1.scatter( pos4[:,0], pos4[:,1] , c=colors )
ax1.set_title('MDS Components 0 and 1\nSubspace Projection')

ax2.scatter( pos4[:,2], pos4[:,3] , c=colors )
ax2.set_title('MDS Components 2 and 3\nSubspace Projection')

plt.show()

pairplot_df = pd.DataFrame(pos4, columns=['MDS Component '+str(j) for j in range(pos4.shape[1])])
pairplot_df.reindex(pairplot_df.columns.sort_values(ascending=True))
z_columns = pairplot_df.columns

pairplot_df['Cluster'] = y_training
pairplot_df = pairplot_df.sort_values('Cluster',ascending=True)
sns.pairplot(pairplot_df, hue='Cluster', 
             vars=z_columns, # don't plot the category/system response
             palette='Set1')

plt.show()

#######################################################
#
# This plot shows the subspace projections into the 
# dimensions resulting from Multidimensional Scaling (MDS)
#
#######################################################

