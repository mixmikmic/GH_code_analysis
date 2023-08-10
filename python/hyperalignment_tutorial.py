get_ipython().magic('matplotlib inline')
import numpy as np
from scipy.spatial.distance import pdist, cdist
from mvpa2.datasets.base import Dataset
from mvpa2.mappers.zscore import zscore
from mvpa2.misc.surfing.queryengine import SurfaceQueryEngine
from mvpa2.algorithms.searchlight_hyperalignment import SearchlightHyperalignment
from mvpa2.base.hdf5 import h5save, h5load
# Alternatively, all those above can be imported using
# from mvpa2.suite import *
import matplotlib.pyplot as plt
from mvpa2.support.nibabel.surf import read as read_surface

dss_train = []
dss_test = []
subjects = ['rid000005', 'rid000011', 'rid000014']

for subj in subjects:
    ds = Dataset(np.load('raiders/{subj}_run00_lh.npy'.format(subj=subj)))
    ds.fa['node_indices'] = np.arange(ds.shape[1], dtype=int)
    zscore(ds, chunks_attr=None)
    dss_train.append(ds)
    ds = Dataset(np.load('raiders/{subj}_run01_lh.npy'.format(subj=subj)))
    ds.fa['node_indices'] = np.arange(ds.shape[1], dtype=int)
    zscore(ds, chunks_attr=None)
    dss_test.append(ds)

# Each run has 336 time points and 10242 features per subject.
print(dss_train[0].shape)
print(dss_test[0].shape)

sl_radius = 5.0
qe = SurfaceQueryEngine(read_surface('fsaverage.lh.surf.gii'), radius=sl_radius)

hyper = SearchlightHyperalignment(
    queryengine=qe,
    compute_recon=False, # We don't need to project back from common space to subject space
    nproc=1, # Number of processes to use. Change "Docker - Preferences - Advanced - CPUs" accordingly.
)

# mappers = hyper(dss_train)
# h5save('mappers.hdf5.gz', mappers, compression=9)

mappers = h5load('mappers.hdf5.gz') # load pre-computed mappers

dss_aligned = [mapper.forward(ds) for ds, mapper in zip(dss_test, mappers)]
_ = [zscore(ds, chunks_attr=None) for ds in dss_aligned]

def compute_average_similarity(dss, metric='correlation'):
    """
    Returns
    =======
    sim : ndarray
        A 1-D array with n_features elements, each element is the average
        pairwise correlation similarity on the corresponding feature.
    """
    n_features = dss[0].shape[1]
    sim = np.zeros((n_features, ))
    for i in range(n_features):
        data = np.array([ds.samples[:, i] for ds in dss])
        dist = pdist(data, metric)
        sim[i] = 1 - dist.mean()
    return sim

sim_test = compute_average_similarity(dss_test)
sim_aligned = compute_average_similarity(dss_aligned)

plt.figure(figsize=(6, 6))
plt.scatter(sim_test, sim_aligned)
plt.xlim([-.2, .5])
plt.ylim([-.2, .5])
plt.xlabel('Surface alignment', size='xx-large')
plt.ylabel('SL Hyperalignment', size='xx-large')
plt.title('Average pairwise correlation', size='xx-large')
plt.plot([-1, 1], [-1, 1], 'k--')
plt.show()

def movie_segment_classification_no_overlap(dss, window_size=6, dist_metric='correlation'):
    """
    Parameters
    ==========
    dss : list of ndarray or Datasets
    window_size : int, optional
    dist_metric : str, optional

    Returns
    =======
    cv_results : ndarray
        An n_subjects x n_segments boolean array, 1 means correct classification.
    """
    dss = [ds.samples if hasattr(ds, 'samples') else ds for ds in dss]
    def flattern_movie_segment(ds, window_size=6):
        n_seg = ds.shape[0] // window_size
        ds = ds[:n_seg*window_size, :].reshape((n_seg, window_size, -1))
        ds = ds.reshape((n_seg, -1))
        return ds
    dss = [flattern_movie_segment(ds, window_size=window_size) for ds in dss]
    n_subj, n_seg = len(dss), dss[0].shape[0]
    ds_sum = np.sum(dss, axis=0)
    cv_results = np.zeros((n_subj, n_seg), dtype=bool)
    for i, ds in enumerate(dss):
        dist = cdist(ds, (ds_sum - ds) / float(n_subj - 1), dist_metric)
        predicted = np.argmin(dist, axis=1)
        acc = (predicted == np.arange(n_seg))
        cv_results[i, :] = acc
    return cv_results

acc_test = movie_segment_classification_no_overlap(dss_test)
acc_aligned = movie_segment_classification_no_overlap(dss_aligned)
print('Classification accuracy with surface alignment: %.1f%%' % (acc_test.mean()*100, ))
print('Classification accuracy with SL hyperalignment: %.1f%%' % (acc_aligned.mean()*100, ))

print('Classification accuracy with surface alignment per subject:', acc_test.mean(axis=1))
print('Classification accuracy with SL hyperalignment per subject:', acc_aligned.mean(axis=1))



