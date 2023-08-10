from msmbuilder.example_datasets import AlanineDipeptide
trajs = AlanineDipeptide().get().trajectories
from msmbuilder.cluster import MiniBatchKMedoids
kmeds = MiniBatchKMedoids(n_clusters=100,metric='rmsd',max_iter=10)
kmeds.fit(trajs)

dtrajs = kmeds.transform(trajs)

import pyemma

msm = pyemma.msm.estimate_markov_model(dtrajs,1)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.imshow(msm.transition_matrix,cmap='Blues',interpolation='none')
plt.colorbar()

lags=range(10)+range(10,101)[::10]
its = pyemma.msm.its(dtrajs,lags,nits=10)

pyemma.plots.plot_implied_timescales(its)

get_ipython().set_next_input('C = msm.count_matrix_active');get_ipython().magic('pinfo msm.count_matrix_active')

C = msm.count_matrix_active

C = msm.count_matrix_active

from scipy.stats import entropy as kl_div

def BACE_bayes_factor(C,i,j):
    C_hat = C.sum(1)
    T = C / C_hat[:,None]

    q = (C_hat[i]*T[i] + C_hat[j]*T[j]) / (C_hat[i]+C_hat[j])
    return C_hat[i] * kl_div(T[i],q) + C_hat[j] * kl_div(T[j],q)

import numpy as np
n = len(C)

mergeability = np.zeros((n,n))
mask = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        if i!=j and C[i,j]>0 and C[j,i] > 0:
            mergeability[i,j] = -BACE_bayes_factor(C,i,j)
            mask[i,j] = 1

np.max(mergeability[mergeability!=0])

plt.imshow(-mergeability,cmap='Blues',interpolation='none')
plt.colorbar()

(mergeability[mergeability!=0]).flatten()

x = np.random.rand(10,2)
x.sum(1)

x / x.sum(1)[:,None]

def fractional_metastability(T):
    return np.trace(T)/len(T)

def product_of_fractional_metastabilities(Ts):
    return np.prod([fractional_metastability(T) for T in Ts])

class RoseTree(object):
    # says which indices are lumped at which level
    
    def __init__(self,children):
        self.children = children
        
    def leaves(self):
        leaf_list = []

        for child in self.children:
            if type(child)==RoseTree:
                for leaf in child.leaves():
                    leaf_list.append(leaf)
            else:
                leaf_list.append(child)
        return leaf_list
    
def join(a,b):
    return RoseTree([a,b])

def absorb(a,b):
    a.children.append(b)

def collapse(a,b):
    return RoseTree(a.children+b.children)

T_0 = RoseTree([RoseTree([i]) for i in range(n)])

T = T_0.children[0]
while len(T_0.children)>0:
    T = join(T_0.children.pop(),T)

np.trace(msm.transition_matrix)/n

from sklearn.cluster import SpectralBiclustering
bic = SpectralBiclustering(6)
bic.fit(msm.transition_matrix)

msm.transition_matrix.shape

def cg_T(microstate_T, microstate_pi, cg_map):
    ''' Coarse-grain a microstate transition matrix by applying cg_map
    Parameters
    ----------
    microstate_T : (N,N), array-like, square
     microstate transition matrix
    microstate_pi : (N,), array-like
     microstate stationary distribution
    cg_map : (N,), array-like
     assigns each microstate i to a macrostate cg_map[i]
    Returns
    -------
    T : numpy.ndarray, square
     macrostate transition matrix
    '''

    n_macrostates = np.max(cg_map)+1
    n_microstates = len(microstate_T)

    # compute macrostate stationary distribution
    macrostate_pi = np.zeros(n_macrostates)
    for i in range(n_microstates):
        macrostate_pi[cg_map[i]] += microstate_pi[i]
    macrostate_pi /= np.sum(macrostate_pi)

    # accumulate macrostate transition matrix
    T = np.zeros((n_macrostates,n_macrostates))
    for i in range(n_microstates):
        for j in range(n_microstates):
            T[cg_map[i],cg_map[j]] += microstate_pi[i] * microstate_T[i,j]

    # normalize
    for a in range(n_macrostates):
        T[a] /= macrostate_pi[a]
    
    return T

T = msm.transition_matrix
pi = msm.stationary_distribution
macro_T = cg_T(T,pi,bic.row_labels_)

fractional_metastability(macro_T)

fractional_metastability(T[bic.row_labels_==0][:,bic.row_labels_==0])

fractional_metastability(T[bic.row_labels_==1][:,bic.row_labels_==1])

fractional_metastability(T[bic.row_labels_==2][:,bic.row_labels_==2])

np.trace(T)

np.prod([np.trace(T[bic.row_labels_==i][:,bic.row_labels_==i]) for i in range(3)])

dumb_labels = np.zeros(n)
dumb_labels[n/2:] = 1

np.prod([np.trace(T[dumb_labels==i][:,dumb_labels==i]) for i in range(2)])

np.trace(T[dumb_labels==0][:,dumb_labels==0])

np.trace(T[dumb_labels==1][:,dumb_labels==1])

def plot_contiguous(T,mapping):
    sorted_inds = np.array(sorted(range(len(T)),key=lambda i:mapping[i]))
    plt.imshow(T[sorted_inds][:,sorted_inds],interpolation='none',cmap='Blues')
    plt.colorbar()

plot_contiguous(T,dumb_labels)

plot_contiguous(T,bic.row_labels_)

submatrix = lambda i:T[bic.row_labels_==i][:,bic.row_labels_==i]
T_0 = submatrix(5)

bic2 = SpectralBiclustering(2)
bic2.fit(T_0)
bic2.row_labels_

plot_contiguous(T_0,bic2.row_labels_)

macro_Ts = []
Ns = range(2,30)

for i in Ns:
    bic = SpectralBiclustering(i)
    bic.fit(msm.transition_matrix)
    macro_T = cg_T(T,pi,bic.row_labels_)
    macro_Ts.append(macro_T)

plt.plot(Ns,[np.trace(t) for t in macro_Ts])

plt.plot(Ns,[np.trace(t)/len(t) for t in macro_Ts])

plt.plot(Ns,[np.trace(t**2) / (len(t)) for t in macro_Ts])

plt.plot(Ns,[np.trace(t)**2 / (len(t)) for t in macro_Ts])

plt.plot(Ns,[np.log(np.trace(t)) /(len(t)) for t in macro_Ts])

plt.plot(Ns,[np.trace(t)/(len(t)**2) for t in macro_Ts])

np.linalg.norm(T-np.diag(np.diag(T)))

np.trace(T)

np.sum(msm.count_matrix_active)

np.trace(msm.count_matrix_active)/np.sum(msm.count_matrix_active)

np.sum(msm.count_matrix_active[dumb_labels==0][:,dumb_labels==0])

msm.count_matrix_active

likelihood_unnorm = np.prod(msm.transition_matrix**msm.count_matrix_active)
likelihood_unnorm

np.min(msm.transition_matrix**msm.count_matrix_active)

msm = pyemma.msm.BayesianMSM(1,nsamples=1000)
msm.fit(dtrajs)

mats = np.array([m.transition_matrix for m in msm.samples[:10]])

plt.imshow(mats.mean(0),interpolation='none',cmap='Blues')
plt.title('Mean')
plt.colorbar()

plt.figure()
stdev = mats.std(0)
plt.imshow(stdev,interpolation='none',cmap='Blues')
plt.title('Standard deviation')
plt.colorbar()

plt.figure()
stderr = mats.std(0)/np.sqrt(len(mats))
plt.imshow(stderr,interpolation='none',cmap='Blues')
plt.title('Standard error')
plt.colorbar()

for sample in msm.samples:
    plt.plot(sample.pi)

stat_dists=np.array([sample.pi for sample in msm.samples])

stat_dists.mean(0).shape

plt.plot(stat_dists.mean(0))

stderr = stat_dists.std(0)#/msm.nsamples
plt.fill_between(np.arange(msm.nstates),stat_dists.mean(0)+stderr,stat_dists.mean(0)-stderr,alpha=0.4)

np.argmax(stat_dists.std(0)/stat_dists.mean(0))

plt.plot(stat_dists.std(0))

plt.scatter(stat_dists.std(0),stat_dists.mean(0))

plt.plot(stat_dists.std(0)/stat_dists.mean(0))



