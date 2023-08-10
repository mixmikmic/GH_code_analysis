from sys import path

problem_dir = '../generators/'  
path.append(problem_dir)
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import datasets
import numpy as np

centers = [(-1, -1), (5, 5)]
X, y = datasets.make_blobs(n_samples=100, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=42)

print('Data = Gaussian Isotropic blobs in 2D(clusters).')
print('X: %d data points in %dD.' % X.shape)
print('y: cluster of each data point.')

def blob_plot(ax, x, y, title, **params):
    ax.scatter(x, y, **params)
    ax.set_title(title)
    ax.axhline(0, color='k')
    ax.axvline(0, color='k')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
def kde_plot(ax, x1b, x2b, xx, yy, f, title, **params):
    ax.set_xlim(x1b[0], x1b[1])
    ax.set_ylim(x2b[0], x2b[1])
    cfset = ax.contourf(xx, yy, f, cmap='Blues')
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

import matplotlib.pyplot as plt
import pylab

fig, ax = plt.subplots()
blob_plot(ax, X[:, 0], X[:, 1], title='Real data', c=y, cmap=pylab.cm.spring, edgecolors='k')

from sklearn.preprocessing import LabelEncoder 
from copula_generator import matrix_to_rank
     
X_rank = matrix_to_rank(X)

fig, ax = plt.subplots()
blob_plot(ax, X_rank[:, 0], X_rank[:, 1], title='Rank - Real data', c=y, cmap=pylab.cm.spring, edgecolors='k')

from scipy.stats import norm
from copula_generator import rank_matrix_to_inverse

X_inverse = rank_matrix_to_inverse(X_rank)

fig, ax = plt.subplots()
blob_plot(ax, X_inverse[:, 0], X_inverse[:, 1], title='Inverse cdf - Real data', c=y, cmap=pylab.cm.spring, edgecolors='k')

import scipy.stats as st
from sklearn.neighbors import KernelDensity

x1 = X_inverse[:, 0]
x2 = X_inverse[:, 1]

x1b = (x1.min(), x1.max())
x2b = (x2.min(), x2.max())

#  Kernel density parameter bandwidth between 0 and 1.
bandwidth = 0.1

# KDE using Scipy.
kernel_sc = st.gaussian_kde(np.vstack([x1, x2]), bw_method=bandwidth)

# KDE using Sklearn.
kernel_sk = KernelDensity(bandwidth=bandwidth).fit(np.vstack([x1, x2]).T)

# Grid of points to plot the 2D distribution.
xx, yy = np.mgrid[x1min:x1max:100j, x2min:x2max:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
# Estimate the grid of points using the density estimations.
f_sc = np.reshape(kernel_sc(positions).T, xx.shape)
f_sk = np.reshape(np.exp(kernel_sk.score_samples(positions.T)), xx.shape)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
kde_plot(ax[0], x1b, x2b, xx, yy, f_sc, 'Kernel Density Estimation - Scipy \n Bandwidth = {:0.2f}'.format(bandwidth))
kde_plot(ax[1], x1b, x2b, xx, yy, f_sk, 'Kernel Density Estimation - Sklearn \n Bandwidth = {:0.2f}'.format(bandwidth))

X_artif_sc = kernel_sc.resample(100)
X_artif_sk = kernel_sk.sample(100)

fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
blob_plot(ax[0, 0], X_inverse[:, 0], X_inverse[:, 1], title='Inverse cdf - Real data',             c=y, cmap=pylab.cm.spring, edgecolors='k')
blob_plot(ax[0, 1], X_artif_sc[0, :], X_artif_sc[1, :], title='Inverse cdf - Artificial data - Scipy',             c='r')
blob_plot(ax[1, 1], X_artif_sk[:, 0], X_artif_sk[:, 1], title='Inverse cdf - Artificial data - Sklearn',             c='b')
ax[1, 0].axis('off')
plt.tight_layout()

from copula_generator import marginal_retrofit

X_retrofit_sc = marginal_retrofit(X_artif_sc.T, X)
X_retrofit_sk = marginal_retrofit(X_artif_sk, X)

fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
custom_plot(ax[0, 0], X[:, 0], X[:, 1], title='Real data',             c=y, cmap=pylab.cm.spring, edgecolors='k')
custom_plot(ax[0, 1], X_retrofit_sc[:, 0], X_retrofit_sk[:, 1], title='Marginal Retrofit - Artificial data - Scipy',             c='r')
ax[1, 0].axis('off')
custom_plot(ax[1, 1], X_retrofit_sk[:, 0], X_retrofit_sk[:, 1], title='Marginal Retrofit - Artificial data - Sklearn',             c='b')
plt.tight_layout()

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

# Grid search to find optimal bandwidth.
grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.001, 10.0, 1000)},
                    cv=20)
grid.fit(np.vstack([x1, x2]).T)
print(grid.best_params_)

from sys import path

generator_dir = '../generators/'
utils_dir = '../data_manager/'
path.append(problem_dir)
path.append(utils_dir)

from metric import *

print('Covariance discrepancy: ', cov_discrepancy(X, X_artif_sc.T))
print('Correlation discrepancy: ', corr_discrepancy(X, X_artif_sc.T))
print('Relief divergence: ', relief_divergence(X, X_artif_sc.T))
print('KS_test: ', ks_test(X, X_artif_sc.T))
print('NN discrepancy: ', nn_discrepancy(X, X_artif_sc.T))
print('BAC metric: ', bac_metric(X, X_artif_sc.T))



