# First, we need to import some Python packages
import numpy as np
import pandas as pd
import os.path as op
import warnings
import matplotlib.pyplot as plt
plt.style.use('classic')
warnings.filterwarnings("ignore")
get_ipython().magic('matplotlib inline')

# The onset times are loaded as pandas dataframe with three columns: 
# onset times (in seconds) (column 1), durations (column 2), and conditions (column 3).
# N.B.: condition 0 = passive, condition 1 = active
stim_info = pd.read_csv(op.join('example_data', 'onsets.csv'), sep='\t',
                        names=['onset', 'duration', 'trial_type'])

stim_info[['onset', 'duration']] = (stim_info[['onset', 'duration']] / 2).astype(int)

# ANSWER
n_trials = 40
stim_vec = np.zeros((162, n_trials))

for itrial in range(n_trials):
    idx = stim_info['onset'][itrial]
    stim_vec[idx, itrial] = 1

np.testing.assert_array_equal(stim_vec, np.load('example_data/stim_vec.npy'))
print("Well done!")

from glob import glob

import os
# the images are in img/ on Linux/Mac systems, but in img\ on Windows (hence the "os.sep" thingie)
my_search_string = 'img' + os.sep + '*.png'
png_files = glob(my_search_string)
print(png_files)

# ANSWER
tstat_paths = glob(op.join('..', 'data', 'pi0070', 'wm.feat', 'stats', 'tstat*.nii.gz'))

# To check your answer, run this cell
assert(len(tstat_paths) == 40)
print("Well done! You globbed all the 40 tstat-files correctly!")

# Let's fix it
from functions import sort_nifti_paths
tstat_paths = sort_nifti_paths(tstat_paths)

import nibabel as nib
data = nib.load(tstat_paths[0]).get_data()

# ANSWER
voxel_dims = (80, 80, 37) # The data is in EPI-space
X = np.zeros((len(tstat_paths), np.prod(voxel_dims)))

for trial, tstat_path in enumerate(tstat_paths):
    data = nib.load(tstat_path).get_data()
    data = data.ravel()
    X[trial,:] = data

# Can we check if X is correct here? Would be a good check before continuing to part 2
np.testing.assert_almost_equal(X, np.load('example_data/X_section1.npz')['X'])
print("Well done!")

# ANSWER
y = stim_info['trial_type'].values

np.testing.assert_equal(np.array(y), np.load('example_data/y.npy'))
print('Well done!')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # Here we initialize the StandardScaler object
scaler.fit(X)             # Here we "fit" the StandardScaler to our entire dataset (i.e. calculates means and stds of each feature)
X = scaler.transform(X)   # And here we transform the dataset using the calculated means/stds

means = np.mean(X, axis=0)
np.testing.assert_almost_equal(means, np.zeros(X.shape[1]))
print("Each column (feature) has mean 0!")
stds = X.std(axis=0)
np.testing.assert_almost_equal(stds[stds != 0], np.ones((stds != 0).sum()))
print("Each column (feature) has std 1!")

# Scikit-learn is always imported as 'sklearn'
from sklearn.svm import SVC

# clf = CLassiFier
clf = SVC(kernel='linear')

print('Fitting SVC ...', end='')
clf.fit(X, y)
print(' done.')

coefs = clf.coef_
print("Shape of coefficients: %r" % (coefs.shape,))

y_hat = clf.predict(X)
print("The predictions for my samples are:\n %r" % y_hat.tolist())

# ANSWER
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
lda.predict(X)

# ANSWER
accuracy = (y_hat==y).mean() 
print(accuracy)

# or alternatively
accuracy = (y_hat == y).sum() / y.size # or: len(y)

# or using scikit-learn metrics
from sklearn.metrics import accuracy_score
accuracy_score(y, y_hat)

from sklearn.model_selection import train_test_split

if not isinstance(y, np.ndarray):
    y = np.array(y)

# The argument "test_size" indicates the test-size as a proportion
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y,
                                                    random_state=5)
    
print("Shape X_train: %r" % (X_train.shape,))
print("Shape y_train: %r" % (y_train.shape,))
print("Shape X_test: %r" % (X_test.shape,))
print("Shape y_test: %r" % (y_test.shape,))

# ANSWER
is_equal_ratio = y_train.mean() == y_test.mean()
print("Is the class-ratio the same? --> %s (namely %.1f)" % (is_equal_ratio, y_train.mean()))

# ANSWER
clf.fit(X=X_train, y=y_train)
y_hat_train = clf.predict(X=X_train)
y_hat_test = clf.predict(X=X_test)

print("Accuracy train: %.3f" % (y_hat_train == y_train).mean())
print("Accuracy test: %.3f" % (y_hat_test == y_test).mean())

# ANSWER
clf.fit(X=X_train, y=y_train)
y_hat_train = clf.predict(X=X_train)
y_hat_test = clf.predict(X=X_test)

print("ROC-AUC train: %.3f" % roc_auc_score(y_train, y_hat_train))
print("ROC-AUC test: %.3f" % roc_auc_score(y_test, y_hat_test))

# scikit-learn is imported as 'sklearn'
from sklearn.model_selection import StratifiedKFold

# They call folds 'splits' in scikit-learn
skf = StratifiedKFold(n_splits=5)

folds = skf.split(X, y)

# Notice how we "unpack" the train- and test-indices at the start of the loop
i = 1
for train_idx, test_idx in folds:
    
    print("Processing fold %i" % i)
    print("Train-indices: %s" % train_idx)
    print("Test-indices: %s\n" % test_idx)
    
    i += 1

# EXAMPLE ANSWER
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# clf now is a logistic regression model
clf = LogisticRegression()
# run split() again to generate folds
folds = skf.split(X, y)

performance = np.zeros(skf.n_splits)

for i, (train_idx, test_idx) in enumerate(folds):
    
    X_train = X[train_idx,:]
    y_train = y[train_idx]
    
    X_test = X[test_idx,:]
    y_test = y[test_idx]
    
    # ToDo: call fit (on train) and predict (on test)
    model = clf.fit(X=X_train, y=y_train)
    y_hat = model.predict(X=X_test)
    
    # ToDo: calculate accuracy
    performance[i] = roc_auc_score(y_test, y_hat)

# ToDo: calculate average accuracy
print('Mean performance: %.3f' % np.mean(performance))

X_r = np.random.randn(80, 1000)
print("Shape of X: %s" % (X_r.shape, ), '\n')

y_r = np.tile([0, 1], 40)
print("Shape of y: %s" % (y_r.shape, ))
print("Y labels:\n%r" % y_r.tolist(), '\n')
runs = np.repeat([1, 2, 3, 4], 20)
print("Shape of runs: %s" % (runs.shape, ))
print("Run-indices: \n%r" % runs.tolist())

# Import from model_selection module
from sklearn.model_selection import GroupKFold

# In fact, when we initialize GroupKFold with 4 splits, as below, it is exactly the same as
# the LeaveOneGroupOut cross-validator, since we only have 4 groups
gkf = GroupKFold(n_splits=4)
for train_idx, test_idx in gkf.split(X=X_r, y=y_r, groups=runs):
    
    print("Indices of our test-samples: %r" % test_idx.tolist())
    print("... which correspond to following runs: %r" % runs[test_idx].tolist(), '\n')

# ANSWER
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.2)

from sklearn.feature_selection import SelectKBest, f_classif

# f_classif is a scikit-learn specific implementation of the F-test
select2000best = SelectKBest(score_func=f_classif, k=2000)

from sklearn.feature_selection import SelectFwe, chi2
selectfwe_transformer = SelectFwe(score_func=chi2, alpha=0.01)

# Fit the transformer ...
select2000best.fit(X, y)

# ... which calculates the following attributes (.scores_ and .pvalues_)
# Let's check them out
scores = select2000best.scores_
pvalues =  select2000best.pvalues_

# As you can see, each voxel gets its own score (in this case: an F-score)
print(scores.size)

# and its own p-value:
print(pvalues.size)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

scores_3d = scores.reshape((80, 80, 37))
plt.figure(figsize=(20, 5))

for i, slce in enumerate(np.arange(15, 65, 5)):
    plt.subplot(2, 5, (i+1))
    plt.title('X = %i' % slce, fontsize=20)
    plt.imshow(scores_3d[slce, :, :].T, origin='lower', cmap='hot')
    plt.axis('off')
plt.tight_layout()
plt.show()

# ANSWER
from sklearn.decomposition import PCA
X_train_tmp, X_test_tmp = train_test_split(X, test_size=0.5)

pca = PCA(n_components=5)
pca.fit(X_train_tmp)
X_train_pca_transformed = pca.transform(X_train_tmp)
X_test_pca_transformed = pca.transform(X_test_tmp)

print("Shape X_train (after PCA): %s" % (X_train_pca_transformed.shape,))
print("Shape X_test (after PCA): %s" % (X_test_pca_transformed.shape,))

# ANSWER
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
folds = skf.split(X, y)

performance = np.zeros(skf.n_splits)
select1000best = SelectKBest(score_func=f_classif, k=1000)

for i, (train_idx, test_idx) in enumerate(folds):
    
    X_train = X[train_idx,:]
    X_test = X[test_idx,:]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    select1000best.fit(X=X_train, y=y_train)
    X_train = select1000best.transform(X=X_train)
    X_test = select1000best.transform(X=X_test)
    
    # ToDo: calculate accuracy
    clf.fit(X=X_train, y=y_train)
    y_test_hat = clf.predict(X=X_test)
    performance[i] = roc_auc_score(y_test, y_test_hat)

# ToDo: calculate average accuracy
print('Mean performance: %.3f' % np.mean(performance))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
ufs = SelectKBest(score_func=f_classif, k=1000)
pca = PCA(n_components=10)  # we want to reduce the features to 10 components
svc = SVC(kernel='linear')

from sklearn.pipeline import Pipeline

pipeline_to_make = [('preproc', scaler),
                    ('ufs', ufs),
                    ('pca', pca),
                    ('clf', svc)]

my_pipe = Pipeline(pipeline_to_make)

X_train, y_train = X[0::2], y[0::2]
X_test, y_test = X[1::2], y[1::2]

my_pipe.fit(X_train, y_train)
predictions = my_pipe.predict(X_test)
performance = roc_auc_score(y_test, predictions)
print("Cross-validated performance on test-set: %.3f" % performance)

# ANSWER
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('varthres', VarianceThreshold()),
    ('cluster', KMeans(15)),
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier())
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
performance = roc_auc_score(y_test, predictions)
print("Cross-validated performance on test-set: %.3f" % performance)

# Example answer
participant_numbers = glob(op.join('..', 'data', 'pi*'))
print(participant_numbers)

# Next, we need to extract the participant numbers from the paths you just obtained. We do this for you here.
participant_numbers = [x.split('/')[-1] for x in participant_numbers]
print('\nThere are the participant numbers:\n%s' % participant_numbers)

# ANSWER

skf = StratifiedKFold(n_splits=5)
select100best = SelectKBest(f_classif, k=100)
clf = SVC(kernel='linear')
pipe = Pipeline([('scaler', StandardScaler()), ('ufs', select1000best), ('clf', clf)])

all_performance = np.zeros(len(participant_numbers))

### Start loop over subjects ###
for i_sub, participant_number in enumerate(participant_numbers):
    
    path_this_participant = op.join('..', 'data', participant_number, 'wm.feat', 'stats', 'tstat*.nii.gz')
    t_stat_paths_this_participant = sort_nifti_paths(glob(path_this_participant))
    voxel_dims = nib.load(t_stat_paths_this_participant[0]).header.get_data_shape()
    X = np.zeros((len(t_stat_paths_this_participant), np.prod(voxel_dims)))

    for trial, tstat_path in enumerate(t_stat_paths_this_participant):
        data = nib.load(tstat_path).get_data()
        data = data.ravel()
        X[trial,:] = data
    
    performance_this_participant = np.zeros(skf.n_splits)
    
    # Loop over the folds
    for i_fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        performance = roc_auc_score(y_test, preds)        
        performance_this_participant[i_fold] = performance

    mean_performance_this_participant = performance_this_participant.mean()
    print('Mean performance for participant %s: %.3f' % (participant_number, mean_performance_this_participant))
    
    all_performance[i_sub] = mean_performance_this_participant

print('\nFinal performance mean (std): %.3f (%.3f)' % (all_performance.mean(), all_performance.std()))

from scipy.stats import ttest_1samp

# Example answer
t, p = ttest_1samp(all_performance, 0.5)
print('The t-value is %.3f, with a p-value of %.5f' % (t, p))

