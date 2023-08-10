# This line allows plotting directly in the notebook
get_ipython().magic('matplotlib inline')

# Python scientific package
import numpy as np

from nilearn.datasets import fetch_abide_pcp


# We specify the site and number of subjects we want to download
abide = fetch_abide_pcp(derivatives=['func_preproc'], SITE_ID=['NYU'], n_subjects=3)

# We look at the available data in this dataset
print(abide.keys())

print(abide.description)

# To get the functional dataset, we have to retrieve the variable 'func_preproc'
func = abide.func_preproc

# We can also look at where the data is loaded
print(func[1])

from nilearn import decomposition

# CanICA is nilearn's approach of group ICA. It directly embeds a masker.
canica = decomposition.CanICA(n_components=20, mask_strategy='background')
canica.fit(func)

# Retrieve the components
components = canica.components_

# Use CanICA's masker to project the components back into 3D space
components_img = canica.masker_.inverse_transform(components)

# We visualize the generated atlas
from nilearn import plotting, image

plotting.plot_stat_map(image.index_img(components_img, 9), title='DMN')
plotting.show()

# Import and initialize `NiftiMapsMasker` object and call `fit_transform` to
# extract timeseries signals from computed atlas.
from nilearn.input_data import NiftiMapsMasker


# The parameters used are maps_img as parcellations, resampling to maps image,
# smoothing of 6mm, detrending, standardizing and filtering (TR in sec). These later
# parameters are applied automatically when extracting timeseries data.
masker = NiftiMapsMasker(components_img,
                         smoothing_fwhm=6, standardize=True, detrend=True,
                         t_r=2.5, low_pass=0.1, high_pass=0.01)

print(masker) # to see the initialization of the parameters.

# We loop over the subjects to extract the time series

subjects_timeseries = []
for subject_func in func:
    subjects_timeseries.append(masker.fit_transform(subject_func))

print("Finished extracting timeseries signals for each subject")

# Visualizing extracted timeseries signals. We import matplotlib.pyplot
import matplotlib.pyplot as plt


# We show them for a single subject
timeseries = subjects_timeseries[0]
print(timeseries.shape) # (number of scans/time points, number of brain regions/parcellations)
plt.plot(timeseries)
plt.title('Timeseries for single subject shown for 20 brain regions')
plt.xlabel('Number of regions')
plt.ylabel('Normalized signal')
plt.show()

from nilearn.regions import RegionExtractor


extractor = RegionExtractor(components_img, threshold=2.,
                            thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            standardize=True, min_region_size=1350)

# Just call fit() to process for regions extraction
extractor.fit()

# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_

# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]

# Visualization of region extraction results
title = ('%d regions are extracted from %d components.'
         % (n_regions_extracted, 20))
plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                         title=title, threshold=0.008)

abide = fetch_abide_pcp(derivatives=['rois_cc200'], SITE_ID=['NYU'], pipeline='cpac')
print(abide.rois_cc200[0].shape)

from nilearn.connectome import ConnectivityMeasure


conn_est = ConnectivityMeasure(kind='partial correlation')
conn_matrices = conn_est.fit_transform(abide.rois_cc200)
conn_matrices.shape

plt.imshow(conn_matrices[0], vmax=.20, vmin=-.20, cmap='RdBu_r')
plt.colorbar()
plt.title('Connectivity matrix of subject 0')

from nilearn.connectome import sym_to_vec


X = sym_to_vec(conn_matrices)
X.shape

from sklearn.cross_validation import StratifiedShuffleSplit


ids = []
for site_id, dx in abide.phenotypic[['SITE_ID', 'DX_GROUP']]:
    ids.append(str(site_id) + str(dx))
cv = StratifiedShuffleSplit(ids, n_iter=10, test_size=.2)

from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
import numpy as np


# DX_GROUP are the labels of the ABIDE dataset. 1=ASD, 2=Healthy
y = abide.phenotypic['DX_GROUP']
predictor = LinearSVC(C=0.01)
np.mean(cross_val_score(predictor, X, y, cv=cv))

from sklearn.linear_model import RidgeClassifier


measures = ['correlation', 'partial correlation', 'tangent']
predictors = [
    ('svc_l2', LinearSVC(C=1)),
    ('svc_l1', LinearSVC(C=1, penalty='l1', dual=False)),
    ('ridge_classifier', RidgeClassifier()),
]

for measure in measures:
    conn_est = ConnectivityMeasure(kind=measure)
    conn_matrices = conn_est.fit_transform(abide.rois_cc200)
    X = sym_to_vec(conn_matrices)
    for name, predictor in predictors:
        print(measure, name, np.mean(cross_val_score(predictor, X, y, cv=cv)))

# Fetch the full data and update phenotypic data and cross_validation
abide = fetch_abide_pcp(derivatives=['rois_cc200'], pipeline='cpac')
y = abide.phenotypic['DX_GROUP']
ids = []
for site_id, dx in abide.phenotypic[['SITE_ID', 'DX_GROUP']]:
    ids.append(str(site_id) + str(dx))
cv = StratifiedShuffleSplit(ids, n_iter=10, test_size=.2)

for measure in measures:
    conn_est = ConnectivityMeasure(kind=measure)
    conn_matrices = conn_est.fit_transform(abide.rois_cc200)
    X = sym_to_vec(conn_matrices)
    for name, predictor in predictors.items():
        print(measure, name, np.mean(cross_val_score(predictor, X, y, cv=cv)))

