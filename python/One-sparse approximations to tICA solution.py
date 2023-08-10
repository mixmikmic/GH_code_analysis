# test case: the example MetEnkephalin dataset
from msmbuilder.example_datasets import MetEnkephalin
print(MetEnkephalin().get().DESCR)
trajs = MetEnkephalin().get().trajectories
import pyemma
feat = pyemma.coordinates.featurizer(trajs[0].top)
feat.add_distances(range(trajs[0].n_atoms))
X = map(feat.transform, trajs)
tica = pyemma.coordinates.tica(X, lag=10, dim=50)
X_tica = tica.get_output()
kmeans = pyemma.coordinates.cluster_mini_batch_kmeans(X_tica, k=500, max_iter=10)
dtrajs = [dtraj.flatten() for dtraj in kmeans.get_output()]



import nglview
superposed = trajs[0].superpose(trajs[0])

view = nglview.show_mdtraj(superposed)
view.clear_representations()
view.add_hyperball()
view

500 * nanoseconds / (2 * femtoseconds)

import pyemma

# let's do diffusion distance instead?
from scipy.spatial.distance import squareform, pdist

def diffusion_cluster(msm, tau=10, n_states=6):
    affinity_matrix = 1 - 0.5 * squareform(pdist(np.linalg.matrix_power(msm.P,tau), p = 1))
    clust = SpectralClustering(n_states,affinity='precomputed')
    clust.fit(affinity_matrix)
    argsorted= np.argsort(clust.labels_)
    plt.imshow(np.log(msm.P)[argsorted][:,argsorted],
               interpolation='none',cmap='Blues')
    plt.figure()
    plt.imshow(affinity_matrix[argsorted][:,argsorted],
               interpolation='none',cmap='Blues')
    plt.colorbar()
    
    return affinity_matrix, clust.labels_

msm = pyemma.msm.estimate_markov_model(dtrajs, 10)

affinity_matrix, cg = diffusion_cluster(msm, tau = 1000, n_states=2)

trajs[0].xyz.shape

from simtk.unit import *
(5 * picosecond) / (2 * femtosecond)

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.hist(np.vstack(X_tica)[:,0],bins=50);

plt.plot(X_tica[0][:,0])



# what are some of the coordinates most correlated with tIC1

plt.plot(sorted(np.abs(tica.feature_TIC_correlation[:,0]))[::-1][:10])

inds = np.argsort(np.abs(tica.feature_TIC_correlation[:,0]))[::-1]

for i in range(8):
    plt.plot(X[0][:,inds[i]],alpha=0.5, label=feat.describe()[inds[i]])
plt.legend(loc=(1,0))

i = 0
plt.title(feat.describe()[inds[i]])
plt.plot(X[0][:,inds[i]],alpha=0.5)

transition_frame = np.argmax(X[0][:,inds[i]] < 0.30)
transition_frame

ind_x = inds[0]
for x in X:
    plt.figure()
    plt.plot(x[:,ind_x])
    plt.ylim(0.15,0.4)

n_buffer = 1
reactive_traj = trajs[0][transition_frame - n_buffer : transition_frame + n_buffer]
reactive_traj = reactive_traj.superpose(reactive_traj)

view = nglview.show_mdtraj(reactive_traj)
view.clear_representations()
view.add_hyperball()
view

plt.plot(X[0][:,inds[0]], X[0][:,inds[1]])

from glob import glob
import mdtraj as md
fnames = glob('data/fah/10470/*.h5')
trajs = map(md.load, fnames)

trajs[0].top.n_residues

sum(map(len, trajs))

len(trajs)

traj = trajs[0]

get_ipython().run_cell_magic('time', '', "\nprojections = [md.compute_contacts(traj, contacts=[[50,150]], scheme = 'closest')[0] for traj in trajs]")

plt.hist(np.vstack(projections).flatten(),bins=50);

plt.plot(md.compute_contacts(traj, contacts=[[50,150]], scheme = 'closest')[0].flatten())

get_ipython().run_cell_magic('time', '', "projections = [ md.compute_contacts(traj, contacts=[[60,160]], scheme = 'closest')[0] for traj in trajs[::10]]\nplt.hist(np.vstack(projections).flatten(),bins=50);")

# let's pick a random subset of contacts
np.random.seed(0)
all_pairs = []
for i in range(trajs[0].top.n_residues):
    for j in range(i):
        all_pairs.append((i,j))
np.random.shuffle(all_pairs)



pairs = all_pairs[:2000]

1.0 * len(pairs) / len(all_pairs)

feat = pyemma.coordinates.featurizer(trajs[0].top)
feat.add_residue_mindist(pairs, scheme = 'closest')

#%%time
#def transform(traj):
#    feat = pyemma.coordinates.featurizer(trajs[0].top)
#    feat.add_residue_mindist(pairs, scheme = 'closest')
#    return feat.transform(traj)
#
#from multiprocessing import Pool
#pool = Pool(8)
#X = pool.map(transform, trajs)

from tqdm import tqdm

X = [feat.transform(traj) for traj in tqdm(trajs)]

lag_time = 10 * nanoseconds
recording_interval = 250 * picoseconds
lag_frames = int(lag_time / recording_interval)
lag_frames



tica = pyemma.coordinates.tica(X, lag = lag_frames, dim=50)
tica.ndim

X_tica = tica.get_output()

plt.hexbin(np.vstack(X_tica)[:,0], np.vstack(X_tica)[:,1], cmap='Blues', bins='log')

def sparsify(X, feat, tica):
    ''' given a tICA object, the top 2 tICs with their most correlated input feature'''
    
    ind_x, ind_y = np.argmax(np.abs(tica.feature_TIC_correlation),0)[:2]
    corr_x, corr_y = np.max(np.abs(tica.feature_TIC_correlation),0)[:2]
    
    print(corr_x, corr_y)
    plt.hexbin(np.vstack(X)[:,ind_x], np.vstack(X)[:,ind_y],cmap='Blues', bins='log')
    plt.xlabel(feat.describe()[ind_x])
    plt.ylabel(feat.describe()[ind_y])
    
sparsify(X, feat, tica)





inds = np.argmax(np.abs(tica.feature_TIC_correlation),0)
corrs = np.max(np.abs(tica.feature_TIC_correlation),0)
plt.plot(corrs)

def compute_eigenvalue_of_trial_direction(trial_direction):
    A = np.reshape(trial_direction, (len(trial_direction), 1))
    C = tica.cov_tau
    S = tica.cov

    return np.trace((A.T.dot(C).dot(A)).dot(np.linalg.inv(A.T.dot(S).dot(A))))

compute_eigenvalue_of_trial_direction(np.ones(len(X[0].T)))

n_features = X[0].shape[1]
eigs = np.zeros(n_features)

for i in range(n_features):
    trial_direction = np.zeros(n_features)
    trial_direction[i] = 1
    eigs[i] = compute_eigenvalue_of_trial_direction(trial_direction)

np.max(eigs)

np.argmax(eigs)

plt.hist(np.vstack(X)[:,np.argmax(eigs)],bins=50);
plt.title(feat.describe()[np.argmax(eigs)])

for x in X[::10]:
    plt.plot(x[:,np.argmax(eigs)])

tica.eigenvalues[0]

(tica.timescales[0] * recording_interval).value_in_unit(microsecond)

plt.plot(np.cumsum(tica.eigenvalues))
plt.plot()
plt.plot(np.cumsum(tica.eigenvalues**2))

X_tica = tica.get_output()
kmeans = pyemma.coordinates.cluster_mini_batch_kmeans(X_tica, k=500, max_iter=100)
dtrajs = [dtraj.flatten() for dtraj in kmeans.get_output()]
msm = pyemma.msm.estimate_markov_model(dtrajs, lag_frames)

np.trace(msm.P)

plt.plot((msm.timescales()[:50] * recording_interval).value_in_unit(nanosecond),'.')
#plt.yscale('log')
plt.ylabel('Implied timescale (ns)')
plt.xlabel('Process index')

from sklearn.cluster import SpectralClustering
clust = SpectralClustering(4,affinity='precomputed')
clust.fit(msm.P)

labels = clust.labels_

plt.imshow(np.log(msm.P)[np.argsort(labels)][:,np.argsort(labels)],interpolation='none',cmap='Blues')

timescale_of_interest = 100 * nanoseconds
tau = int(timescale_of_interest / lag_time)
tau

affinity_matrix, cg = diffusion_cluster(msm, tau = tau)

vals,vecs = np.linalg.eigh(affinity_matrix)
plt.plot(vals[::-1][:20],'.')
plt.yscale('log')

# what happens if I discretize directly in the space of "1-sparse approximate tICs"?
# I may not have _the slowest_ possible input order parameters, but at least we can see what they are...
    
def unique_ify_but_preserve_order(array):
    '''
    Given an array with possibly non-unique elements, remove the redundant elements, but preserve their order 
    (i.e. don't just do list(set(array)))
    
    Returns
    -------
    unique_array : array of now unique elements from array
    
    '''
    
    # if the elements are already all unique, return the array un-modified
    if len(set(array)) == len(array): return range(len(array)), array
    
    # keep track of the elements we've seen so far
    seen_so_far = set()
    new_list = []
    for i, element in enumerate(array):
        if element not in seen_so_far:
            new_list.append(element)
            seen_so_far.add(element)
    return np.array(new_list)

def feature_select(tica, max_components = 50):
    '''
    "x-axis (tIC1) is a weighted combination of these inter-residue distances" is difficult to interpret.
    Luckily, tIC1 often turns out to be 
    '''
    
    # get the list of feature indices that are maximally correlated with each independent component
    possibly_redundant_features = np.argmax(np.abs(tica.feature_TIC_correlation),0)[:max_components]
    features = unique_ify_but_preserve_order(possibly_redundant_features)[:max_components]
    
    ## also get their correlation values?
    #corrs = np.max(np.abs(tica.feature_TIC_correlation),0)[inds]
    
    return features
                
def compute_eigenvalue_of_trial_direction(tica, trial_direction):
    A = np.reshape(trial_direction, (len(trial_direction), 1))
    C = tica.cov_tau
    S = tica.cov

    return np.trace((A.T.dot(C).dot(A)).dot(np.linalg.inv(A.T.dot(S).dot(A))))

def get_eigenvalues(tica, features):
    '''
    We would also like to weight each axis by how "slow" it is, so that the 1-sparse approximation is
    still roughly a kinetic distance.
    '''
    eigs = np.array(len(features))
    for i, feat in enumerate(features):
        trial_direction = np.zeros(len(tica.mean))
        trial_direction[feat] = 1
        eigs[i] = compute_eigenvalue_of_trial_direction(tica, trial_direction)
    return eigs
    
def get_one_sparse_approximate_projection(X, tica, max_components = 50):
    features = feature_select(tica, max_components = max_components)
    eigenvalues = get_eigenvalues(tica, features)
    return [x[:,features] * eigenvalues for x in X]

(tica.mean).shape

inds = feature_select(tica)
X_ = [x[:, inds] for x in X]
features = [feat.describe()[i] for i in inds]

kmeans_ = pyemma.coordinates.cluster_mini_batch_kmeans(X_, k=500, max_iter=10)
dtrajs_ = [dtraj.flatten() for dtraj in kmeans_.get_output()]

msm_ = pyemma.msm.estimate_markov_model(dtrajs_, 10)
np.trace(msm_.P)

plt.plot((msm_.timescales()[:50] * recording_interval).value_in_unit(nanosecond),'.')
#plt.yscale('log')
plt.ylabel('Implied timescale (ns)')
plt.xlabel('Process index')

# what happens if we just greedily take the 50 slowest features, in this case? (they may be very redundant)

inds = np.argsort(eigs)[::-1][:50]
X_ = [x[:, inds] for x in X]
features = [feat.describe()[i] for i in inds]
kmeans_ = pyemma.coordinates.cluster_mini_batch_kmeans(X_, k=500, max_iter=10)
dtrajs_ = [dtraj.flatten() for dtraj in kmeans_.get_output()]
msm_ = pyemma.msm.estimate_markov_model(dtrajs_, 10)
np.trace(msm_.P)

plt.plot((msm_.timescales()[:50] * recording_interval).value_in_unit(nanosecond),'.')
#plt.yscale('log')
plt.ylabel('Implied timescale (ns)')
plt.xlabel('Process index')

affinity_matrix, cg = diffusion_cluster(msm_, tau = 5, n_states=10)

msm_.active_count_fraction, msm_.active_state_fraction

cg_dtrajs = [np.array([cg[i] for i in dtraj],dtype=int) for dtraj in dtrajs_]

from msmbuilder.msm import MarkovStateModel

msm = MarkovStateModel(10)

msm.fit(cg_dtrajs[::2])
msm.score_, msm.score(cg_dtrajs[1::2])

msm.fit(cg_dtrajs)
msm.score_

msm.fit(dtrajs_[::2])
msm.score_, msm.score(dtrajs_[1::2])



