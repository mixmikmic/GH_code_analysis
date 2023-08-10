import numpy as np, pandas as pd, GPy, seaborn as sns
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import tsne

n = 200
m = 40
np.random.seed(1)

x = np.random.uniform(-1, 1, n)
c = np.digitize(x, np.linspace(-1,1,12))-1
cols = np.asarray(sns.color_palette('spectral_r',12))[c]

plt.scatter(x, np.zeros(n), c=cols, cmap='spectral_r', lw=0)

r = np.random.normal(0,1,[m,1])
M = np.eye(m)-2*(r.dot(r.T)/r.T.dot(r))

X = np.c_[x, np.zeros((n,m-1))].dot(M)

plt.scatter(*X[:,1:3].T, c=x, cmap='spectral', lw=0)

from sklearn.decomposition import PCA
Xpca = PCA(2).fit_transform(X)
plt.scatter(*Xpca.T, c=x, cmap='spectral', lw=0)

fig, axes = plt.subplots(2,3,tight_layout=True,figsize=(15,10))
axit = axes.flat
for lr in range(6):
    ax = next(axit)
    Xtsne = tsne.bh_sne(X.copy())
    ax.scatter(*Xtsne.T, c=cols, cmap='spectral', lw=0)
    ax.set_title('restart: ${}$'.format(lr))

from sklearn.manifold import TSNE

fig, axes = plt.subplots(2,3,tight_layout=True,figsize=(15,10))
axit = axes.flat

for perplexity in [30,60]:
    for lr in [200, 500, 1000]:
        ax = next(axit)
        Xtsne = TSNE(perplexity=perplexity, learning_rate=lr, init='pca').fit_transform(X.copy())
        ax.scatter(*Xtsne.T, c=cols, cmap='spectral', lw=0)
        ax.set_title('perp=${}$, learn-rate=${}$'.format(perplexity, lr))

from mpl_toolkits.axes_grid.inset_locator import inset_axes

fig, axes = plt.subplots(2,3,tight_layout=True,figsize=(15,10))
axit = axes.flat
for lr in range(6):
    ax = next(axit)
    m = GPy.models.GPLVM(X.copy(), 2)
    m.optimize(messages=1, gtol=0, clear_after_finish=True)
    msi = m.get_most_significant_input_dimensions()[:2]
    
    is_ = m.kern.input_sensitivity().copy()
    is_ /= is_.max()
    
    XBGPLVM = m.X[:,msi] * is_[np.array(msi)]
    #m.kern.plot_ARD(ax=ax)
    ax.scatter(*XBGPLVM.T, c=cols, cmap='spectral', lw=0)
    ax.set_title('restart: ${}$'.format(lr))
    ax.set_xlabel('dimension ${}$'.format(msi[0]))
    ax.set_ylabel('dimension ${}$'.format(msi[1]))
    
    a = inset_axes(ax,
                    width="30%", # width = 30% of parent_bbox
                    height='20%', # height : 1 inch
                    loc=1)
    sns.barplot(np.array(msi), is_[np.array(msi)], label='input-sens', ax=a)
    a.set_title('sensitivity')
    a.set_xlabel('dimension')

from mpl_toolkits.axes_grid.inset_locator import inset_axes


fig, axes = plt.subplots(2,3,tight_layout=True,figsize=(15,10))
axit = axes.flat
for lr in range(6):
    ax = next(axit)
    m = GPy.models.BayesianGPLVM(X, 5, num_inducing=25)
    m.optimize(messages=1, gtol=0, clear_after_finish=True)
    msi = m.get_most_significant_input_dimensions()[:2]
    
    is_ = m.kern.input_sensitivity()
    is_ /= is_.max()
    
    XBGPLVM = m.X.mean[:,msi] * is_[np.array(msi)]
    #m.kern.plot_ARD(ax=ax)
    ax.scatter(*XBGPLVM.T, c=cols, cmap='spectral', lw=0)
    ax.set_title('restart: ${}$'.format(lr))
    ax.set_xlabel('dimension ${}$'.format(msi[0]))
    ax.set_ylabel('dimension ${}$'.format(msi[1]))
    
    a = inset_axes(ax,
                    width="30%", # width = 30% of parent_bbox
                    height='20%', # height : 1 inch
                    loc=1)
    sns.barplot(range(m.input_dim), is_, label='input-sens')
    a.set_title('sensitivity')



