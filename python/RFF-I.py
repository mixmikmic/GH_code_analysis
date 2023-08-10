import numpy as np # Thinly−wrapped numpy
import pandas as pd
from matplotlib import cm 
import matplotlib as mpl
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt 
data = '../data/'

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from scipy.stats import cauchy, laplace
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel

class RFF(BaseEstimator):
    def __init__(self, gamma = 1, D = 50, metric = "rbf"):
        self.gamma = gamma
        self.metric = metric
        #Dimensionality D (number of MonteCarlo samples)
        self.D = D
        self.fitted = False
        
    def fit(self, X, y=None):
        """ Generates MonteCarlo random samples """
        d = X.shape[1]
        #Generate D iid samples from p(w) 
        if self.metric == "rbf":
            self.w = np.sqrt(2*self.gamma)*np.random.normal(size=(self.D,d))
        elif self.metric == "laplace":
            self.w = cauchy.rvs(scale = self.gamma, size=(self.D,d))
        
        #Generate D iid samples from Uniform(0,2*pi) 
        self.u = 2*np.pi*np.random.rand(self.D)
        self.fitted = True
        return self
    
    def transform(self,X):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components)"""
        if not self.fitted:
            raise NotFittedError("RBF_MonteCarlo must be fitted beform computing the feature map Z")
        #Compute feature map Z(x):
        Z = np.sqrt(2/self.D)*np.cos((X.dot(self.w.T) + self.u[np.newaxis,:]))
        return Z
    
    def compute_kernel(self, X):
        """ Computes the approximated kernel matrix K """
        if not self.fitted:
            raise NotFittedError("RBF_MonteCarlo must be fitted beform computing the kernel matrix")
        Z = self.transform(X)
        K = Z.dot(Z.T)
        return K
    

#size of data
N_SAMPLES, DIM = 1000, 200 
X = np.random.randn(N_SAMPLES,DIM)

gamma = 2
#Number of monte carlo samples D
Ds = np.arange(1,5000,200)
K_rbf, K_laplace = rbf_kernel(X, gamma=gamma), laplacian_kernel(X,gamma=gamma)
errors_rbf, errors_laplace = [] , []

for D in Ds:
    GAUSS = RFF(gamma=gamma, D=D, metric="rbf")
    GAUSS.fit(X)
    K_rbf_a = GAUSS.compute_kernel(X)

    LAPLACE = RFF(gamma=gamma, D=D, metric="laplace")
    LAPLACE.fit(X)
    K_laplace_a = LAPLACE.compute_kernel(X)

    errors_rbf.append(((K_rbf_a-K_rbf)**2).mean())
    errors_laplace.append(((K_laplace_a-K_laplace)**2).mean())

errors_rbf, errors_laplace = np.array(errors_rbf), np.array(errors_laplace)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,4))
for ax,data,title in zip(axes,[errors_laplace,errors_rbf],['RBF Kernel','Laplacian Kernel']):
    ax.plot(Ds, data)
    ax.set_ylabel("MSE")
    ax.set_xlabel("Number of MC samples D")
    ax.set_yscale("log")
    ax.set_title(title)
plt.show()



