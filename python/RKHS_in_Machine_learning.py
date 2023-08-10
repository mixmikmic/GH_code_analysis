from __future__ import division, print_function, absolute_import
from IPython.display import SVG, display, Image

import numpy as np, scipy as sp, pylab as pl, matplotlib.pyplot as plt, scipy.stats as stats, sklearn, sklearn.datasets
from scipy.spatial.distance import squareform, pdist, cdist

import distributions as dist #commit 480cf98 of https://github.com/ingmarschuster/distributions

display(Image(filename="monomials.jpg", width=700))

data = np.vstack([stats.multivariate_normal(np.array([-2,2]), np.eye(2)*1.5).rvs(100),
                  stats.multivariate_normal(np.ones(2)*2, np.eye(2)*1.5).rvs(100)])
distr_idx = np.r_[[0]*100, [1]*100]

for (idx, c, marker) in [(0,'r', (0,3,0)), (1, "b", "x")]:
    pl.scatter(*data[distr_idx==idx,:].T, c=c, alpha=0.4, marker=marker)
    pl.arrow(0, 0, *data[distr_idx==idx,:].mean(0), head_width=0.2, head_length=0.2, fc=c, ec=c)
pl.show()

class Kernel(object):
    def mean_emb(self, samps):
        return lambda Y: self.k(samps, Y).sum()/len(samps)
    
    def mean_emb_len(self, samps):
        return self.k(samps, samps).sum()/len(samps**2)
    
    def k(self, X, Y):
        raise NotImplementedError()

class FeatMapKernel(Kernel):
    def __init__(self, feat_map):
        self.features = feat_map
        
    def features_mean(self, samps):
        return self.features(samps).mean(0)
    
    def mean_emb_len(self, samps):
        featue_space_mean = self.features_mean(samps)
        return featue_space_mean.dot(featue_space_mean)
    
    def mean_emb(self, samps):
        featue_space_mean = self.features(samps).mean(0)
        return lambda Y: self.features(Y).dot(featue_space_mean)
    
    def k(self, X, Y):
        gram = self.features(X).dot(self.features(Y).T)
        return gram

class LinearKernel(FeatMapKernel):
    def __init__(self):
        FeatMapKernel.__init__(self, lambda x: x)

class GaussianKernel(Kernel):
    def __init__(self, sigma):
        self.width = sigma
    
    def k(self, X, Y=None):
        assert(len(np.shape(X))==2)
        
        # if X=Y, use more efficient pdist call which exploits symmetry
        if Y is None:
            sq_dists = squareform(pdist(X, 'sqeuclidean'))
        else:
            assert(len(np.shape(Y))==2)
            assert(np.shape(X)[1]==np.shape(Y)[1])
            sq_dists = cdist(X, Y, 'sqeuclidean')
    
        K = exp(-0.5 * (sq_dists) / self.width ** 2)
        return K

class StudentKernel(Kernel):
    def __init__(self, s2, df):
        self.dens = dist.mvt(0,s2,df)
        
    def k(self, X,Y=None):
        if Y is None:
            sq_dists = squareform(pdist(X, 'sqeuclidean'))
        else:
            assert(len(np.shape(Y))==2)
            assert(np.shape(X)[1]==np.shape(Y)[1])
            sq_dists = cdist(X, Y, 'sqeuclidean')
        dists = np.sqrt(sq_dists)
        return exp(self.dens.logpdf(dists.flatten())).reshape(dists.shape)


def kernel_mean_inner_prod_classification(samps1, samps2, kernel):
    mean1 = kernel.mean_emb(samps1)
    norm_mean1 = kernel.mean_emb_len(samps1)
    mean2 = kernel.mean_emb(samps2)
    norm_mean2 = kernel.mean_emb_len(samps2)
    
    def sim(test):
        return (mean1(test) - mean2(test))
    
    def decision(test):
        if sim(test) >= 0:
            return 1
        else:
            return 0
    
    return sim, decision


def apply_to_mg(func, *mg):
    #apply a function to points on a meshgrid
    x = np.vstack([e.flat for e in mg]).T
    return np.array([func(i.reshape((1,2))) for i in x]).reshape(mg[0].shape)

def plot_with_contour(samps, data_idx, cont_func, method_name, delta = 0.025, pl = pl):
    x = np.arange(samps.T[0].min()-delta, samps.T[1].max()+delta, delta)
    y = np.arange(samps.T[1].min()-delta, samps.T[1].max()+delta, delta)
    X, Y = np.meshgrid(x, y)
    Z = apply_to_mg(cont_func, X,Y)
    Z = Z.reshape(X.shape)


    # contour labels can be placed manually by providing list of positions
    # (in data coordinate). See ginput_manual_clabel.py for interactive
    # placement.
    fig = pl.figure()
    pl.pcolormesh(X, Y, Z > 0, cmap=pl.cm.Pastel2)
    pl.contour(X, Y, Z, colors=['k', 'k', 'k'],
              linestyles=['--', '-', '--'],
              levels=[-.5, 0, .5])
    pl.title('Decision for '+method_name)
    #plt.clabel(CS, inline=1, fontsize=10)
    for (idx, c, marker) in [(0,'r', (0,3,0)), (1, "b", "x")]:
        pl.scatter(*data[distr_idx==idx,:].T, c=c, alpha=0.7, marker=marker)

    pl.show()
    
for (kern_name, kern) in [("Linear", LinearKernel()), 
                          ("Student-t", StudentKernel(0.1,10)), 
                          ("Gauss", GaussianKernel(0.1))
                         ]:
    (sim, dec) = kernel_mean_inner_prod_classification(data[distr_idx==1,:], data[distr_idx==0,:], kern)
    plot_with_contour(data, distr_idx, sim, 'Inner Product classif. '+kern_name, pl = plt)

data, distr_idx = sklearn.datasets.make_circles(n_samples=400, factor=.3, noise=.05)

for (kern_name, kern) in [("Linear", LinearKernel()), 
                          ("Stud", StudentKernel(0.1,10)), 
                          ("Gauss1", GaussianKernel(0.1)),
                         ]:
    (sim, dec) = kernel_mean_inner_prod_classification(data[distr_idx==1,:], data[distr_idx==0,:], kern)
    plot_with_contour(data, distr_idx, sim, 'Inner Product classif. '+kern_name, pl = plt)

