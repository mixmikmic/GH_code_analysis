import warnings
warnings.filterwarnings("ignore")
import pylab as plt
import os.path as op
path_data = op.expanduser('~/data_ucsf/machine_learning')
get_ipython().magic('matplotlib inline')

from nilearn import datasets
from nilearn import plotting

# By default the 2nd subject will be fetched
haxby_dataset = datasets.fetch_haxby(data_dir=path_data)
fmri_filename = haxby_dataset.func[0]

# print basic information on the dataset
print('First subject functional nifti images (4D) are at: %s' %
      fmri_filename)  # 4D data

mask_filename = haxby_dataset.mask_vt[0]
plotting.plot_roi(mask_filename, bg_img=haxby_dataset.anat[0],
                  cmap='Paired')

from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_filename, standardize=True)
masker

# Fitting the transformer initializes it to operate on new data
masker.fit(fmri_filename)

# Now we'll transform our fMRI data
fmri_masked = masker.transform(fmri_filename)

print(fmri_masked)

print(fmri_masked.shape)

import pandas as pd
import numpy as np

# Load target information as string and give a numerical identifier to each
labels = pd.read_csv(haxby_dataset.session_target[0], delimiter=' ')
print(labels.head())

print(labels.shape)
print(fmri_masked.shape)

print(labels['labels'].values[:50])

# Create a mask w/ Pandas
condition_mask = labels.eval('labels in ["face", "cat"]').values

# Create a mask w/ Numpy
# condition_mask = np.logical_or(target == b'face', target == b'cat')

# We apply this mask in the sample direction to restrict the
# classification to the face vs cat discrimination
fmri_masked = fmri_masked[condition_mask]
targets = labels[condition_mask]['labels'].values

print(fmri_masked.shape)

from sklearn.svm import SVC
svc = SVC(kernel='linear')
print(svc)

svc.fit(fmri_masked, targets)

prediction = svc.predict(fmri_masked)
print(prediction)

from sklearn.metrics import accuracy_score

print(accuracy_score(targets, prediction))

svc.fit(fmri_masked[:-30], targets[:-30])

prediction = svc.predict(fmri_masked[-30:])
print((prediction == targets[-30:]).sum() / float(len(targets[-30:])))

# Create the KFold object
from sklearn.model_selection import KFold
n_splits = 15
cv = KFold(n_splits=n_splits)

cv_sample = np.zeros([len(fmri_masked), n_splits])
for ii, (tr,tt) in enumerate(cv.split(fmri_masked)):
    cv_sample[tt, ii] = 1
fig, ax = plt.subplots()
ax.imshow(cv_sample, cmap='coolwarm', aspect='auto', interpolation='nearest')
_ = ax.set(xlabel='Iteration', ylabel='Data Index',
           title='Cross Validation indices\n(red = Test set)')

for train, test in cv.split(fmri_masked):
    svc.fit(fmri_masked[train], targets[train])
    prediction = svc.predict(fmri_masked[test])
    print(accuracy_score(targets[test], prediction))

from sklearn.cross_validation import cross_val_score
cv_score = cross_val_score(svc, fmri_masked, targets)
print(cv_score)

cv_score = cross_val_score(svc, fmri_masked, targets,
                           cv=cv.split(fmri_masked))
print(cv_score)

def plot_classifier_scores(scores):
    fig, ax = plt.subplots()
    ax.hist(scores)
    ax.axvline(.5, ls='--', c='r')
    ax.set_xlabel('Model Score')
plot_classifier_scores(cv_score)

# Find out list of session number
session_label = labels['chunks'][condition_mask]

# Iterate through sessions, validating on a held-out session
from sklearn.cross_validation import LeaveOneLabelOut
cv = LeaveOneLabelOut(session_label)
cv_score = cross_val_score(svc, fmri_masked, targets, cv=cv)
plot_classifier_scores(cv_score)

coef_ = svc.coef_
print(coef_[:5])

print(svc.coef_.shape)
print(fmri_masked.shape)

coef_img = masker.inverse_transform(coef_)
print(coef_img)

coef_img.to_filename('haxby_svc_weights.nii.gz')

from nilearn.plotting import plot_stat_map, show

plot_stat_map(coef_img, bg_img=haxby_dataset.anat[0],
              title="SVM weights")

