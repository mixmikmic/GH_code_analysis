import mne
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
import matplotlib.pyplot as plt

from modelingtools import (delay_time_series, plot_cv_indices,
                           cross_validate_alpha)
import modelingtools

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale, StandardScaler
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_curve, roc_auc_score, r2_score
from sklearn.pipeline import make_pipeline, Pipeline

from itertools import product
import warnings
warnings.simplefilter('ignore')

get_ipython().magic('matplotlib inline')

feature_path = '../features/'
raw_path = '../raw_data/'
sfreq_new = 100  # Cut sfreq in half to save computation time
tmin, tmax = -.5, 4

ecog = mne.read_epochs(feature_path + 'hfa_ecog-epo.fif', preload=True)
ecog.resample(100)
ecog.crop(tmin, tmax)

_ = ecog.plot(scalings='auto', n_epochs=5)

activity = ecog.copy()
_ = mne.baseline.rescale(activity._data, activity.times, (None, 0),
                         mode='zscore', copy=False)
activity = activity._data.mean(0).mean(-1)
use_elec = np.argmax(activity)

# Read in the data
spec = mne.read_epochs(feature_path + 'spectrogram-epo.fif')

# Take the log so it's more normally-distributed in each frequency band
spec._data = np.log(spec._data)

# Resample to save space
spec.resample(sfreq_new)
spec.crop(tmin, tmax)

# Create a delayed representation
delays = np.linspace(-.4, 0, 20)
X_delayed = delay_time_series(spec._data, delays, spec.info['sfreq'])
X_delayed = X_delayed.reshape(X_delayed.shape[0], -1, X_delayed.shape[-1])

n_cv_iterations = 10
indices = np.arange(1000)
n_inds = indices.shape[0]
cv_example = cv.ShuffleSplit(n_inds, n_iter=n_cv_iterations)
plot_cv_indices(cv_example)

n_cv_iterations = 10
indices = np.arange(1000)
n_inds = indices.shape[0]
cv_example = cv.KFold(n_inds, n_folds=n_cv_iterations)
plot_cv_indices(cv_example)

# Iterating across trials (labels)
trials = np.arange(len(ecog))
cv_iterators = [cv.KFold(len(trials), n_folds=n_cv_iterations),
                cv.KFold(len(trials), n_folds=n_cv_iterations, shuffle=True),
                cv.LabelShuffleSplit(trials, n_iter=n_cv_iterations,
                                     test_size=.1)]

fig, axs = plt.subplots(1, len(cv_iterators), figsize=(15, 5),
                        sharey=True, sharex=True)
for ax, icv in zip(axs, cv_iterators):
    ax = plot_cv_indices(icv, ax)
    ax.set_title(type(icv).__name__, fontsize=18)
fig.suptitle('Cross validation over trials (black = test set)',
             fontsize=20, y=1.05)
plt.tight_layout()

# Times to use in fitting
mask_time = mne.utils._time_mask(ecog.times, -.2, 4)

# We'll create the delays for the spectrogram ahead of time
X_delayed = delay_time_series(spec._data, delays, spec.info['sfreq'])
X = X_delayed.reshape(X_delayed.shape[0], -1, X_delayed.shape[-1])
X = X[..., mask_time]
y = ecog._data[:, use_elec, :][..., mask_time]

# We'll use the KFold iterator, shuffling trial numbers first
cross_val_iterator = cv.KFold(len(trials), n_folds=5, shuffle=True)
model = Ridge(alpha=1e5)

fig, ax = plt.subplots(figsize=(6, 8))
y_plt = scale(y)
for ii, (tr, tt) in enumerate(cross_val_iterator):
    # Here we'll define the inner-loop cross validator, which would be used to tune hyperparamters
    # For each inner loop iteration, we'll shade the validation set slightly lighter
    cv_inner = cv.KFold(len(tr), n_folds=4, shuffle=True)
    alphas = np.ones(len(trials))
    for (val_tr, val_tt), trans in zip(cv_inner, [.25, .50, .75, 1.]):
        alphas[tr[val_tt]] = trans
        
    # Now create colors according to training / test sets
    colors = np.zeros(y.shape[0]).astype(bool)
    colors[tt] = True
    colors = np.where(colors, 'orangered', 'royalblue')

    # Make the plot
    for jj, (col, alpha, i_data) in enumerate(zip(colors, alphas, y_plt)):
        ixs = np.arange(len(i_data)) + jj * y_plt.shape[-1]
        i_plt = i_data + ii * 10
        ax.plot(i_plt, ixs, c=col, alpha=alpha)
ax.set_xlabel('Outer CV Iterations')
ax.set_xticklabels(range(len(cross_val_iterator) + 1))
ax.set_title('Training / Test Output Data in Cross Validation')
ax.set_ylabel('Time (samples)')
ax.text(48, ixs.min(), 'Training Set', color='royalblue',
        rotation=-90, fontsize=20, horizontalalignment='left')
ax.text(51, ixs.min(), 'Test Set', color='orangered',
        rotation=-90, fontsize=20, horizontalalignment='left')
ax.axis('tight')
_ = plt.setp(ax.get_yticklabels(), visible=False)

fig, axs = plt.subplots(len(cross_val_iterator), 1,
                        figsize=(10, 5*len(cross_val_iterator)),
                        sharex=True)
axs[0].set_title('Predicted and Actual High-Frequency Activity')
axs[1].set_ylabel('Amplitude (a.u.)')
axs[-1].set_xlabel('Time (s)')

# Iterate through cross-validation splits
for ax, (tr, tt) in zip(axs, cross_val_iterator):
    # Pull the training / testing data for the ecog data
    y_tr = np.hstack(y[tr]).T    
    y_tt = np.hstack(y[tt]).T

    # Pull the training / testing data for the spectrogram
    X_tr = np.hstack(X[tr]).T
    X_tt = np.hstack(X[tt]).T
    
    # Scale all the features for simplicity
    X_tr = scale(X_tr)
    X_tt = scale(X_tt)
    y_tr = scale(y_tr)
    y_tt = scale(y_tt)
    
    # Fit the model, and use it to predict on new data
    model.fit(X_tr, y_tr)
    predictions = model.predict(X_tt)
    
    # Plot the predicted and actual values.
    # We'll subtract the mean from each so it's easier to visualize
    ax.plot(scale(y_tt), color='k', alpha=.2, lw=2)
    ax.plot(scale(predictions), color='r', lw=2)
    ax.axis('tight')
    
    # Finally, plot the coefficient of determination (R2)
    print(r2_score(y_tt, predictions))

_ = axs[0].legend(['Actual', 'Predicted'])

# In this case, we'll keep all the electrodes
y_map = ecog._data[..., mask_time]
tr, tt = next(iter(cross_val_iterator))

# Pull the training / testing data for the ecog data
y_tr = np.hstack(y_map[tr]).T    
y_tt = np.hstack(y_map[tt]).T

# Pull the training / testing data for the spectrogram
X_tr = np.hstack(X[tr]).T
X_tt = np.hstack(X[tt]).T

# Scale all the features for simplicity
X_tr = scale(X_tr)
X_tt = scale(X_tt)
y_tr = scale(y_tr)
y_tt = scale(y_tt)

# Fit the model, and use it to predict on new data
model.fit(X_tr, y_tr)
predictions = model.predict(X_tt)
scores = r2_score(y_tt, predictions)

scores = []
for y_true, y_pred in zip(y_tt.T, predictions.T):
    scores.append(r2_score(y_true, y_pred))

im = plt.imread(raw_path + 'brain.png')
melec = pd.read_csv(raw_path + 'meta_elec.csv')
xy = melec[['x_2d', 'y_2d']].values

ax = modelingtools.plot_activity_on_brain(scores, im, xy[:, 0], xy[:, 1],
                                          size_scale=3000, vmin=-.1, vmax=.1,
                                          cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Prediction Scores ($R^2$)', fontsize=20)

# Choose some log-spaced values for alpha
alphas = np.logspace(1, 8, 8)
n_cv_outer = 3
n_cv_inner = 3

# This is our outer loop iterator
outer_iterator = cv.KFold(len(trials), n_folds=n_cv_outer, shuffle=True)
scores, coefs = cross_validate_alpha(X, y, outer_iterator, alphas)

# Calculate the mean score across CV iterations (both inner and outer)
mean_scores = scores.mean(0).mean(0)

# Plot these values
fig, ax = plt.subplots()
ax.plot(np.log10(alphas), mean_scores)
ix_max = np.argmax(mean_scores)
ann_plt = (np.log10(alphas)[ix_max], mean_scores[ix_max])
ax.annotate('Chosen $\lambda$', ann_plt, ann_plt + np.array([0, -.25]),
            arrowprops=dict(arrowstyle='->'), fontsize=18)
ax.set_xlabel('Alpha (log10)')
ax.set_ylabel('Mean score ($R^2$)')
ax.set_title('Scores for multiple values of alpha')

fig, ax = plt.subplots()
ax.pcolormesh(delays, range(len(spec.ch_names)),
              coefs.mean(0).mean(0)[ix_max, :].reshape([-1, len(delays)]),
              cmap=plt.cm.coolwarm, vmin=-.004, vmax=.004)
ax.set_yticks(np.arange(len(spec.ch_names))[::5])
ax.set_yticklabels(spec.ch_names[::5])
ax.axis('tight')
ax.set_title('Model Coefficients')
ax.set_xlabel('Time Lag (s)')
ax.set_ylabel('Frequency')

# Create random pairs of trials / timepoints and shuffle all the data
# Note that this *shouldn't* change anything if the data is IID (it's not tho)
y_bad = y.copy()
X_bad = X.copy()
ixs = np.array(list(product(range(y_bad.shape[0]), range(y_bad.shape[-1]))))
rand_sample = np.random.permutation(range(len(ixs)))
ixs_rand = ixs[rand_sample]

for (ii_old, jj_old), (ii_new, jj_new) in zip(ixs, ixs_rand):
    y_bad[ii_new, jj_new] = y[ii_old, jj_old] 
    X_bad[ii_new, :, jj_new] = X[ii_old, :, jj_old] 

scores_bad, coefs_bad = cross_validate_alpha(X_bad, y_bad,
                                             outer_iterator, alphas)

# Calculate the mean score across CV iterations for each alpha
mean_scores = scores_bad.mean(1).mean(0)

# Plot these values
fig, ax = plt.subplots()
ax.plot(np.log10(alphas), mean_scores)
ix_max = np.argmax(mean_scores)
ann_plt = (np.log10(alphas)[ix_max], mean_scores[ix_max])
ax.annotate('Chosen $\lambda$', ann_plt, ann_plt + np.array([0, -.25]),
            arrowprops=dict(arrowstyle='->'), fontsize=18)
ax.set_xlabel('Alpha (log10)')
ax.set_ylabel('Mean score ($R^2$)')
ax.set_title('Scores for multiple values of alpha')

fig, ax = plt.subplots()
ax.pcolormesh(delays, range(len(spec.ch_names)),
              coefs_bad.mean(0).mean(0)[ix_max, :].reshape([-1, len(delays)]),
              cmap=plt.cm.coolwarm)
ax.set_yticks(np.arange(len(spec.ch_names))[::5])
ax.set_yticklabels(spec.ch_names[::5])
ax.axis('tight')
ax.set_title('Overfit Coefficients')
ax.set_xlabel('Time Lag (s)')
ax.set_ylabel('Frequency')

# Pull ECoG phoneme data
phonemes = mne.read_epochs(feature_path + 'ecog_phonemes-epo.fif')
phoneme_labels = pd.read_csv(raw_path + 'phoneme_labels.csv', index_col=0)
phoneme_labels = phoneme_labels.query('phoneme in @phonemes.event_id.keys()')

# Define our targets
target_ph = ['vowels']
non_target_ph = ['plosives']
phonemes_targets = phoneme_labels.query('kind in @target_ph')['phoneme'].values
phonemes_non_targets = phoneme_labels.query('kind in @non_target_ph')['phoneme'].values

# Only include the epochs that include one of our phonemes of choice
ecog_targets = phonemes[list(phonemes_targets)]._data.mean(-1)
ecog_non_targets = phonemes[list(phonemes_non_targets)]._data.mean(-1)

# Now binarize the phoneme labels so we can classify
labels_targets = np.ones(ecog_targets.shape[0])
labels_non_targets = np.zeros(ecog_non_targets.shape[0])

X = np.vstack([ecog_targets, ecog_non_targets])
y = np.hstack([labels_targets, labels_non_targets])

# Define our CV and model
cv_classify = cv.StratifiedShuffleSplit(y, n_iter=100, test_size=.1)
model = RandomForestClassifier()

# This is the ratio of targets to non targets
ratio_targets = float(np.sum(y)) / y.shape[0]
print(ratio_targets)

# Loop through CV iterations
# fit the model and predict the class of testing data
all_predictions = []
all_true = []
for tr, tt in cv_classify:
    y_tr = y[tr]
    y_tt = y[tt]
    
    X_tr = X[tr]
    X_tt = X[tt]
    
    # Only scale X, since y is a label
    X_tr = scale(X_tr)
    X_tt = scale(X_tt)
    
    model.fit(X_tr, y_tr)
    predictions = model.predict_proba(X_tt)
    all_predictions.append(predictions)
    all_true.append(y_tt)

fig, ax = plt.subplots(figsize=(5, 5))
fpr, tpr, thresholds = roc_curve(all_true[0], all_predictions[0][:, 0],
                                 pos_label=1)
ax.plot(fpr, tpr, color='k')
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--', color='r')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve for one iteration')

fig, ax = plt.subplots(figsize=(10, 10))
results = []
scores = []
scores_chance = []
for i_true, i_pred in zip(all_true, all_predictions):
    fpr, tpr, thresholds = roc_curve(i_true, i_pred[:, 0], pos_label=1)
    score = roc_auc_score(i_true, i_pred[:, 0])
    score_chance = roc_auc_score(i_true, np.repeat(ratio_targets, len(i_true)))
    df = pd.DataFrame.from_dict({'fpr': fpr, 'tpr': tpr, 'thresh': thresholds})
    results.append(df)
    scores.append(score)
    scores_chance.append(score_chance)
    ax.plot(fpr, tpr, alpha=.05, color='k', lw=2)
results = pd.concat(results)
results['thresh_rnd'] = results['thresh'].round(2)
av_results = results.groupby('thresh_rnd').mean().sort_values('fpr')
av_results.plot('fpr', 'tpr', ax=ax, color='r', lw=4, label='average ROC')
ax.plot((0, 1), (0, 1), ls='--', color='k', lw=3)
ax.set_xticks([0, .5, 1])
ax.set_yticks([0, .5, 1])
_ = plt.setp(ax.get_xticklabels() + ax.get_yticklabels() +
             [ax.xaxis.label, ax.yaxis.label], fontsize=18)
ax.get_legend().set_visible(False)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('All ROC Curves', fontsize=22)

# Create the confidence intervals w/ a bootstrap
random_means = [np.mean(np.random.choice(scores, len(scores)))
                for _ in range(1000)]
lo, hi = np.percentile(random_means, [.5, 99.5])

fig, ax = plt.subplots(figsize=(10, 5))
_ = ax.axvline(.5, ls='--', color='r', lw=2)
_ = ax.hist(scores, color='k')
_ = ax.hlines(25, lo, hi, lw=4)
ax.set_ylim([0, 30])
ax.set_title('AUC Scores for all CVs\n99% Confidence Interval Above')
ax.set_xlabel('AUC')

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Set up some variables that let us define targets for each phoneme
rev_event_id = {value: key for key, value in phonemes.event_id.items()}

# We'll classify the phoneme class so we have fewer total classes
label_mapping = {ph: kind for kind, ph in phoneme_labels.values}

# These are integers, one for each class
targets = np.array([rev_event_id[ii] for ii in phonemes.events[:, -1]])
targets = np.array([label_mapping.get(ii, None) for ii in targets])

# Calculate the mean electrode activity just after each phoneme as input
X = phonemes._data.mean(-1)

# Remove phonemes we don't want
include_targets = ['fricatives', 'nasals', 'plosives', 'vowels']
mask_use = [target in include_targets for target in targets.astype(str)]
X = X[mask_use]
targets = targets[mask_use]
df_labels = np.unique(targets)

# Instantiate the model + cross validation
model = RandomForestClassifier(n_estimators=20)
cv = StratifiedShuffleSplit(test_size=0.05, n_splits=10)

# Matrices will store the confusion matrix for each CV iteration
# It will count false classifications for each phoneme
matrices = []
for train, test in cv.split(X, targets):
    model.fit(X[train], targets[train])
    predictions = model.predict(X[test])
    matrix = confusion_matrix(targets[test], predictions,
                              labels=df_labels)
    matrix = matrix / matrix.sum(-1)  # Divide by total "true" classes for this iteration to gat a %
    matrices.append(matrix)
matrix = np.array(matrices).mean(0)

# Now we'll show the heatmap
matrix = pd.DataFrame(matrix, index=df_labels, columns=df_labels)
ax = sns.heatmap(matrix, cmap='viridis')

