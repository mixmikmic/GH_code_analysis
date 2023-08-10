get_ipython().magic('matplotlib inline')

import numpy as np

from nilearn import datasets

# parameter n_subjects=None gives us all 146 subjects
# For 10 subjects, n_subjects=10
data = datasets.fetch_cobre(n_subjects=None)

# To see the files fetched related to the datasets.
print(data.keys())

# Functional datasets are stored in variable 'func'
functional_datasets = data.func

n_datasets = len(functional_datasets)
print("Total functional datasets of %d subjects are fetched and loaded." % n_datasets)
print(functional_datasets[0]) # to check the location of the data storage

# Phenotypic or demographic variables are stored in 'phenotypic'
phenotypic = data.phenotypic

# Its corresponding matlab files are stored in variable 'mat_files'
mat_files = data.mat_files

from nilearn import datasets

# asymmetric version of clusters can be selected using version='asym'
parcellations = datasets.fetch_atlas_basc_multiscale_2015(version='asym')

# By default are scales are fetched and stored in output assigned variable 'parcellations'
# To launch specific scale of networks
networks_122 = parcellations['scale122']
# To launch specific scale of networks
networks_444 = parcellations['scale444']

# To view the atlas, we can use of nilearn plotting utilities to see what has
# been fetched and how does it look like. For this, we import plotting and use
# `plot_roi` to visualize the atlas maps in 3D and colormap of our own choice
from nilearn import plotting

plotting.plot_roi(networks_122, cmap=plotting.cm.bwr,
                  title='BASC atlas of 122 clusters')
plotting.plot_roi(networks_444, cmap=plotting.cm.bwr_r,
                 title='BASC atlas of 444 clusters')

# Import and initialize `NiftiLabelsMasker` object and call `fit_transform` to
# extract timeseries signals from provided MSDL functional atlas.
from nilearn.input_data import NiftiLabelsMasker

# The parameters used are maps_img as parcellations, resampling to maps image,
# smoothing of 6mm, detrending, standardizing and filtering (TR in sec). These later
# parameters are applied automatically when extracting timeseries data.
labels_masker = NiftiLabelsMasker(labels_img=networks_122, resampling_target='labels',
                                  smoothing_fwhm=6, standardize=True, detrend=True,
                                  t_r=2.5, low_pass=0.1, high_pass=0.01)

print(labels_masker) # to see the initialization of the parameters.

# We loop over all the functional datasets to fetch confounds for each subject
# data and simultaneously used with other preprocessing parameters to clean
# timeseries signals data.
subjects_timeseries = []
for index in range(len(functional_datasets)):
    each_dataset = functional_datasets[index]
    #print("Loading as Nifti-like image object from each subject path, filename:%s" % each_dataset)
    signals = labels_masker.fit_transform(each_dataset)
    subjects_timeseries.append(signals)

print("Finished extracting timeseries signals for each subject")
print("Array subjects_timeseries contains list of '%d' subject specific timeseries signals" % len(subjects_timeseries))

# Visualizing extracted timeseries signals. We import matplotlib.pyplot
import matplotlib.pyplot as plt

# We show only for one subject
timeseries = subjects_timeseries[4]
print(timeseries.shape) # (number of scans/time points, number of brain regions/parcellations)
plt.plot(timeseries)
plt.title('Timeseries for single subject shown for 122 brain regions')
plt.xlabel('Number of regions')
plt.ylabel('Normalized signal')
plt.show()

from nilearn.connectome import ConnectivityMeasure

print("Measuring Connectivity")

connectivity = ConnectivityMeasure(kind='correlation')
connectivity_matrices = connectivity.fit_transform(subjects_timeseries)
print "Finished Measuring Connectomes"

measure = 'correlation'
mean_connectivity_matrices = connectivity_matrices.mean(axis=0)
# Visualizing goes here

title=('Connectivity Matrix measure "%s"' % measure)
plt.figure()
plt.title(title)
plt.imshow(mean_connectivity_matrices, )
plt.show()

# We import SVC, StratifiedShuffleSplit, cross_val_score from scikit learn for
# classification problem using cross validation procedure and import sym_to_vec
# from nilearn.connectome module to compute connectivity coefficients for each
# measure to use them in classification.
from nilearn.connectome import sym_to_vec
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score

measure = 'correlation'
# Connectivity coefficients
connectivity_coefs = sym_to_vec(connectivity_matrices)
print("-- Support Vector Classification --")
# Get the phenotypic variables of Subject type for classification
classes = phenotypic['sz']
# Initialize Stratified Shuffle Split with 100 iterations and test size
stratified_shuffle = StratifiedShuffleSplit(classes, n_iter=100,
                                            test_size=0.25, random_state=0)
svc = SVC(kernel='linear', class_weight='balanced', random_state=0)
_, classes_ = np.unique(classes, return_inverse=True)
cross_validation = cross_val_score(svc, connectivity_coefs, classes_,
                                   cv=stratified_shuffle)
print("Classification scores '%s': %1.2f +/- %1.2f" % (measure, cross_validation.mean(),
                                                     cross_validation.std()))

measure = 'tangent'
connectivity_coefs = sym_to_vec(connectivity_matrices[measure])
cross_validation = cross_val_score(svc, connectivity_coefs, classes_,
                                   cv=stratified_shuffle)
print("Classification scores '%s': %1.2f +/- %1.2f" % (measure, cross_validation.mean(),
                                                     cross_validation.std()))

measure = 'tangent'

title=('Connectivity Matrix measure "%s"' % measure)
plt.figure()
plt.title(title)
plt.imshow(mean_tangent, )
plt.show()

