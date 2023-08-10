import numpy as np
import pandas as pd
import utls
from os.path import join
import matplotlib.pyplot as plt
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

utls.reset_plots()

from scipy.stats import multivariate_normal

data_dir = '../Data/'

data = pd.read_csv(join(data_dir,'faithful.csv'), delimiter=',')
data.columns=['d','w']

fig, ax = plt.subplots(1,1)
ax.plot(data['d'],data['w'],'.k');
ax.set_xlabel('Eruption duration (mins)');
ax.set_ylabel('Eruption waiting time (mins)');

# z-transform data
data['d'] = (data['d'] - data['d'].mean())/data['d'].std(ddof=1)
data['w'] = (data['w'] - data['w'].mean())/data['w'].std(ddof=1)
data.head()

def compute_expected_sufficient_statistic(data,params):
    """
    Compute the responsibility cluster k takes for each data point
    """
    
    lik0 = multivariate_normal.pdf(data,mean=params['mu0'], cov=params['sig0'])
    lik1 = multivariate_normal.pdf(data,mean=params['mu1'], cov=params['sig1'])
    
    rik=np.array(zip(lik0, lik1))

    den = rik[:,0]*params['pi0'] + rik[:,1]*params['pi1']

    rik[:,0] = rik[:,0]*params['pi0']/den
    rik[:,1] = rik[:,1]*params['pi1']/den
    
    return rik # weighted probability data point xi belongs to cluster k
    
    

A = np.array([[1,2],[2,3]]); B = np.array([[4,2],[1,1]])
np.einsum('ij,jk->ik',A,B) == np.dot(A,B)

# Initialise parameter guess
nclusters = 2
c_guess = np.eye(nclusters)
params = {'pi0':0.5,'pi1':0.5,'mu0':np.array([-1,1]),'mu1':np.array([1,-1]),'sig0':c_guess,'sig1':c_guess}

params_ic = params.copy()

N = len(data)

n_iter = 50

d = data.as_matrix()

rik_all = np.zeros((len(data),2,n_iter))
for i in range(n_iter):
    
    if i % 10 == 0:
        print(i)
    
    # 1. Expectation
    # Compute expected sufficient statistics
    # i.e. probability data point i is in cluster k given parameters

    rik = compute_expected_sufficient_statistic(data,params) # (data index, probabilities)
    rik_all[:,:,i] = rik
    
    # 2. Maximization
    # Optimize expected complete data log likelihood w.r.t. parameters
    rk = rik.sum(axis=0) # sum over data (rows)
    
    for k in range(nclusters): 
        params['pi{}'.format(k)] = np.sum(rik[:,k])/N      
    
    means = np.dot(rik.T,d)/rk[:,None] # each row is the location of the mean of the clusterb
    
    # an array of covariance matrices
    covs = np.einsum('il,im,ik->klm',d,d,rik)/rk[:,None,None] - np.einsum('ij,ik->ijk',means, means) 
    # NB: np.einsum('j,k->jk',means[0],means[0]) == np.outer(means[0],means[0])
    
    for k in range(nclusters):     
        # Store params
        params['mu{}'.format(k)] = means[k,:]                
        params['sig{}'.format(k)] = covs[k]
        
        
    

cmap = plt.cm.coolwarm

X = np.linspace(-2,2)
Y = np.linspace(-2,2)
X, Y = np.meshgrid(X,Y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
Z0_i = utls.multivariate_gaussian(pos, mu=params_ic['mu0'], Sigma=params_ic['sig0'])
Z0_f = utls.multivariate_gaussian(pos, mu=params['mu0'], Sigma=params['sig0'])
Z1_i = utls.multivariate_gaussian(pos, mu=params_ic['mu1'], Sigma=params_ic['sig1'])
Z1_f = utls.multivariate_gaussian(pos, mu=params['mu1'], Sigma=params['sig1'])





rik_orig = compute_expected_sufficient_statistic(data,params_ic)

fig,axs = plt.subplots(1,2, figsize=(2*5,5))

ax = axs[0]
for row in data.iterrows():
    i = row[0]
    d = row[1]
    ax.plot(d['d'],d['w'],'.',color=cmap(int(round(rik_orig[i,0]*cmap.N))),alpha=0.5)

ax.contour(X, Y, Z0_i, cmap='Reds', alpha=0.5)
ax.contour(X, Y, Z1_i, cmap='Blues', alpha=0.5)

ax.set_title('Initial Guess');
ax.set_xlabel('Normalized duration');
ax.set_ylabel('Normalized waiting time');
ax.set_xlim([-2,2]);
ax.set_ylim([-2,2]);

ax = axs[1]
for row in data.iterrows():
    i = row[0]
    d = row[1]
    ax.plot(d['d'],d['w'],'.',color=cmap(int(round(rik[i,0]*cmap.N))),alpha=0.5)
ax.set_title('Final Answer');
ax.set_xlabel('Normalized duration');
ax.set_ylabel('Normalized waiting time');

ax.contour(X, Y, Z0_f, cmap='Reds')
ax.contour(X, Y, Z1_f, cmap='Blues')



plt.tight_layout()

