# vim: set filetype=python:

import numpy as np
import matplotlib.pyplot as plt
from skbayes.mixture_models import VBGMMARD
from matplotlib.patches import Ellipse
import pandas as pd
get_ipython().magic('matplotlib inline')



def plotter(X, max_k,title, rand_state = 1, prune_thresh = 1e-5, mfa_max_iter = 10):
    '''
    Plotting function for VBGMMARD clustering
    
    Parameters:
    -----------
    X: numpy array of size [n_samples, n_features]
       Data matrix
       
    max_k: int
       Maximum number of components
       
    title: str
       Title of the plot
       
    Returns:
    --------
    :instance of VBGMMARD class 
    '''
    # fit model & get parameters
    gmm = VBGMMARD(n_components = max_k, prune_thresh = prune_thresh,
                   n_mfa_iter = mfa_max_iter)
    gmm.fit(X)
    centers = gmm.means_
    covars  = gmm.covars_
    k_selected = centers.shape[0]
    
    # plot data
    fig, ax = plt.subplots(figsize = (10,6))
    ax.plot(X[:,0],X[:,1],'bo', label = 'data')
    ax.plot(centers[:,0],centers[:,1],'rD', markersize = 8, label = 'means')
    for i in range(k_selected):
        plot_cov_ellipse(pos = centers[i,:], cov = covars[i], ax = ax)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc = 2)
    plt.title((title + ', {0} initial clusters, {1} selected clusters').format(max_k,k_selected))
    plt.show()
    return gmm
    
    
# plot_cov_ellipse function is taken from  
# https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, fill = False,
                    edgecolor = 'k',linewidth = 4,**kwargs)

    ax.add_artist(ellip)
    return ellip

# before running this script make sure you downloaded
# old faithful geyser data and put it in the same directory
# as this ipython notebook ():
# https://github.com/AmazaspShumik/sklearn-bayes/blob/master/ipython_notebooks_tutorials/mixture_models/faithful.csv

Data = pd.read_csv("faithful.csv")
Data = np.array(Data[['eruptions','waiting']])
geyser_model_gmm1  = plotter(Data, 20,'Old Faithful')

geyser_model_gmm2  = plotter(Data, 20,'Old Faithful', mfa_max_iter = 1)

X = np.zeros([600,2])
np.random.seed(0)
X[0:200,:]   = np.random.multivariate_normal(mean = (0,10), cov = [[3,0],[0,2]], size = 200)
X[200:400,:] = np.random.multivariate_normal(mean = (0,0) , cov = [[3,0],[0,2]], size = 200)
X[400:600,:] = np.random.multivariate_normal(mean = (0,-10) , cov = [[3,0],[0,2]], size = 200)

sy_gmm_1 = plotter(X,20,'Synthetic Example')

X = np.zeros([600,2])
np.random.seed(0)
X[0:200,:]   = np.random.multivariate_normal(mean = (0,0), cov = [[3,-2],[5,-8]], size = 200)
X[200:400,:] = np.random.multivariate_normal(mean = (0,0), cov = [[-10,10],[-10,10]], size = 200)
X[400:600,:] = np.random.multivariate_normal(mean = (0,0), cov = [[5,-1],[-5,-1]], size = 200)

sy_gmm_2 = plotter(X,20,'Synthetic Example', prune_thresh = 1e-5, mfa_max_iter = 10)

