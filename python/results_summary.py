get_ipython().magic('matplotlib inline')
import pickle
import matplotlib.pyplot as plt
import numpy as np

from mne import read_epochs
from mne.decoding import TimeDecoding, GeneralizationAcrossTime

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVR

from jr.gat import subscore
from jr.gat import AngularRegression, scorer_angle
from jr.stats import corr_linear_circular
from jr.plot import pretty_gat, pretty_decod

# Load first subject epochs (~41 Mo)
epochs = read_epochs('../data/subject_01-epo.fif', verbose=False)
with open('../data/subject_01-eve.pkl', 'rb') as f:
    events = pickle.load(f)

# Let's have a look at the average response across trials
evoked = epochs.average()
evoked.plot_joint(title='all')
plt.show()

# Let's see whether we can isolate the target evoke response.
# For this we can subtract the absent trials (mask only) from the present:
present = events['target_present']
evo_present = epochs[present].average()
evo_absent = epochs[-present].average()

evoked.data = evo_present.data - evo_absent.data
evoked.plot_joint(title='Present - Absent')
plt.show()

# Since there's a large class imbalance, we use a probabilistic output
# and an ROC scorer.
def scorer(y_true, y_pred):
    return roc_auc_score(y_true, y_pred[:, 1])
td = TimeDecoding(predict_method='predict_proba', scorer=scorer, n_jobs=-1)
td.fit(epochs, y=events['target_present'])
td.score(epochs, y=events['target_present'])
td.plot(chance=.5)
plt.show()

# The decoding tells very little about the underlying neural substrates.
# Temporal generalization can help clarify the functional architecture that
# underlies the signal.

# /!\ On the server, there's only one core, so be patient!

gat = GeneralizationAcrossTime(predict_method='predict_proba', scorer=scorer,
                               n_jobs=-1)
y = events['target_present']  # in machine learning, y is the regressor.

# Fit models with cross validation.
gat.fit(epochs, y=y)

# Predict orientation and score the difference between true and predicted
# presence of the stimulus.
gat.score(epochs, y=y)

# Plot the scores
gat.plot()
plt.show()

# These scores reflects both seen and unseen trials. Let's see how these
# two categories differ with one another.
# For clarity purposes, we'll compute the ROC versus all absent trials in
# each conditions.
seen = events['detect_button'] > 0
absent = -events['target_present']  
subselects = dict(seen=np.where(seen | absent)[0],
                  unseen=np.where(-seen | absent)[0],)
y = events['target_present']

fig, ax = plt.subplots(1)
for name, idx in subselects.iteritems():
    # Do not refit, just score the seen and unseen separately
    scores = gat.scores_ = subscore(gat, y=y[idx], sel=idx)
    # Plot each condition
    gat.plot_diagonal(chance=.5, ax=ax, label=name, show=False)
plt.show()

# define the regressor: here the probe angle
y = np.array(events['probe_circAngle'].values)
X = epochs._data

# We need a 2D X (trials x dimension)
n_trial, n_chan, n_time = epochs._data.shape
X = X.reshape([-1, n_chan * n_time])

# linear circular correlation between MEG and stim angle
_, R2, _ = corr_linear_circular(X, y)

# plot the effect size using MNE
R2 = R2.reshape([n_chan, n_time])
evoked.data = R2
evoked.plot_joint()

# Let's use a double regressor to estimate the angle from the sine and cosine
# See `method_model_types.ipynb` to see understand how this is built.

#/!\ On the server, there's only one core, so be patient!

clf_angle = make_pipeline(StandardScaler(), AngularRegression(clf=LinearSVR()))
td = TimeDecoding(clf=clf_angle, scorer=scorer_angle, n_jobs=-1)

# Define regressor
y = np.array(events['probe_circAngle'].values)

# This can take a while, so let's only decode around probe onset
epochs_probe = epochs.crop(.700, copy=True)

# Fit models with cross validation
td.fit(epochs_probe, y=y)

# Predict orientation and score the difference between true and predicted
# orientations
td.score(epochs_probe, y=y)

# Plot the scores
td.plot(chance=0.)
plt.show()

# Load average ERF effects (~14 Mo)
with open('../data/results_evoked.pkl', 'rb') as f:
    erf_analyses = pickle.load(f)

# Here, we will only plot a subselection only to avoid crowding.
# But you can play with the data!
erf_analyses = [a for a in erf_analyses if a['name'] in [
                'target_present', 'target_circAngle', 'probe_circAngle']]

# Define time regions of interest for which you want a topography
tois = [(-.100, 0.050), (.100, .250), (.300, .800), (.900, 1.050)]

# Plot mean effect size within each toi.
for analysis in erf_analyses:
    name, evoked = analysis['name'], analysis['evoked']
 
    # Note that the grads have already been combined.
    # Adjust color scale from 10th and 90th percentile of combined grads
    vmin, vmax = np.percentile(evoked.data[::3, :], [10, 90])

    # Plot topo of mean effect on TOI
    fig, axes = plt.subplots(1, len(tois), figsize=[6, 2])
    fig.subplots_adjust(wspace=0.01, left=0.)
    for ax, toi in zip(axes, tois):
        evoked.plot_topomap(times=[np.mean(toi)], average=np.ptp(toi),
                            cmap='afmhot_r', ch_type='grad', show=False,
                            contours=False, scale=1, colorbar=False,
                            sensors=False, axes=ax, vmin=vmin, vmax=vmax,
                            title=name)
    # Plot butterfly
    evoked.plot()

# Load data  (~31 Mo)
with open('../data/results_decoding.pkl', 'rb') as f:
    gat_analyses = pickle.load(f)

fig, axes = plt.subplots(len(gat_analyses), 1, figsize=[6.5, 11])
for (analysis, ax) in zip(gat_analyses, axes):
    chance = analysis['chance']
    times = analysis['times']
    color = analysis['color']
    
    # Get diagonal scores for each subject to get SEM
    scores_diag = np.array([np.diag(score) for score in analysis['scores']])

    # p values cluster corrected for diagonal results
    pval_diag = analysis['pval_diag']

    # Plot
    pretty_decod(scores_diag, times=times, sig=pval_diag < 0.01,
                 chance=chance, color=color, fill=True, ax=ax)
    ax.set_title(analysis['name'])

analysis = gat_analyses[0]  # try changing the analysis yourself to inspect the results
chance = analysis['chance']
times = analysis['times']
# mean across subjects
scores = np.mean(analysis['scores'], axis=0)
# p values cluster corrected for whole GAT matrix
pval = analysis['pval']
# plot
fig, ax = plt.subplots(1)
pretty_gat(scores, times=times, sig=pval < 0.01, ax=ax, chance=chance)
ax.set_title(analysis['name'])
plt.show()

