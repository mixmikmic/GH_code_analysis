import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_decision_regions import *
from sklearn.datasets import make_moons, make_circles

# Moon sample
X_m, y_m = make_moons(n_samples=100, random_state=123)
# Circle sample
X_c, y_c = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

fig = plt.figure(figsize=(12,5))
fig.add_subplot(1,2,1)
plt.scatter(X_m[y_m==0, 0], X_m[y_m==0, 1], color='red',  marker='^', alpha=0.5)
plt.scatter(X_m[y_m==1, 0], X_m[y_m==1, 1], color='blue', marker='o', alpha=0.5)
fig.add_subplot(1,2,2)
plt.scatter(X_c[y_c==0, 0], X_c[y_c==0, 1], color='red',  marker='^', alpha=0.5)
plt.scatter(X_c[y_c==1, 0], X_c[y_c==1, 1], color='blue', marker='o', alpha=0.5)
plt.show()

# Plotting function
def show(X, y):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    # 2D
    ax[0].scatter(X[y==0, 0], X[y==0, 1], color='red',  marker='^', alpha=0.5)
    ax[0].scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
    # 1D
    ax[1].scatter(X[y==0, 0], np.zeros((len(X[y==0]),1))+0.02, color='red',  marker='^', alpha=0.5)
    ax[1].scatter(X[y==1, 0], np.zeros((len(X[y==1]),1))-0.02, color='blue', marker='^', alpha=0.5)
    # plotting
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_xlabel('PC1')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    plt.show()

from sklearn.decomposition import PCA

scikit_pca = PCA(n_components=2)
X_mspca = scikit_pca.fit_transform(X_m)
X_cspca = scikit_pca.fit_transform(X_c)
show(X_mspca, y_m)
show(X_cspca, y_c)

from rbf_kernel_pca import *

X_mkpca = rbf_kernel_pca(X_m, gamma=15, n_components=2) # return transfered_X, eigenvalue 
X_ckpca = rbf_kernel_pca(X_c, gamma=15, n_components=2)
show(X_mkpca[0], y_m)
show(X_ckpca[0], y_c)

X_mkpca[1]

def project_x(x_new, X, gamma, eigenvectors, eiganvalues):
    pair_dist = np.array([np.sum(x_new-row)**2 for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(eigenvectors/eiganvalues)

eigenvectors, eiganvalues = rbf_kernel_pca(X_m, gamma=15, n_components=1)
x_new = X_m[25]
x_new

x_proj = eigenvectors[25]
x_proj

eigenvectors[20]

x_reproj = project_x(x_new, X_m, gamma=15, eigenvectors=eigenvectors, eiganvalues=eiganvalues)
x_reproj

plt.figure(figsize=(8,6))
plt.scatter(eigenvectors[y_m==0, 0], np.zeros((50)), color='red',  marker='^', alpha=0.5)
plt.scatter(eigenvectors[y_m==1, 0], np.zeros((50)), color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj,   0, color='black', label='original projection of point X[25]', marker='^', s=100)
plt.scatter(x_reproj, 0, color='green', label='remapped point X[25]', marker='x', s=500)
plt.legend(scatterpoints=1)
plt.show()

from sklearn.decomposition import KernelPCA

scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X_m)

plt.figure(figsize=(8,6))
plt.scatter(X_skernpca[y_m==0, 0], X_skernpca[y_m==0, 1], color='red',  marker='^', alpha=0.5)
plt.scatter(X_skernpca[y_m==1, 0], X_skernpca[y_m==1, 1], color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()



