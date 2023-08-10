get_ipython().magic('matplotlib inline')

import matplotlib
from matplotlib import pyplot as plt
from __future__ import division

import seaborn

import numpy as np

from scipy import stats

# Mean and standard deviation for N (dummy variables)
mu = 0.0
sigma = 1.0

alpha=0.05

# Number of samples
half_n = 25

# Number of variables
d = 1000

np.random.seed(2)

# Training dataset
X_tr = mu+sigma*np.random.randn(2*half_n, d)
y_tr = np.ones(2*half_n)
y_tr[:half_n] = 0

# Test dataset
X_ts = mu+sigma*np.random.randn(2*half_n, d)
y_ts = np.ones(2*half_n)
y_ts[:half_n] = 0

p_values = np.array([stats.ttest_ind(X_tr[half_n:,j], X_tr[:half_n,j])[1] for j in range(d)])
sorted_variables = np.argsort(p_values)

# Without Bonferroni correction
selected_variables = np.where(p_values[sorted_variables] < alpha)[0]
print("[No Bonferroni] - {} variables pass the T-test".format(len(selected_variables)))

# With Bonferroni correction
B_selected_variables = np.where(p_values[sorted_variables] < alpha/d)[0]
print("[With Bonferroni] - {} variables pass the T-test".format(len(B_selected_variables)))

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

kf = KFold(n_splits=5)
clf = RidgeClassifier(alpha=0)
accuracies = []
k_values = np.arange(1, 200, 10)

for k in k_values:
    selected = sorted_variables[:k]
    
    # KFCV
    acc_kf = []
    for train_index, vld_index in kf.split(X_tr):
        clf.fit(X_tr[train_index][:,selected], y_tr[train_index])  # fit model on the training set
        y_pred = clf.predict(X_tr[vld_index][:,selected])  # predict test set
        acc_kf.append(accuracy_score(y_tr[vld_index], y_pred))  # estimate accuracy
    
    # Save the KFCV mean accuracy for k
    accuracies.append(np.mean(acc_kf))
accuracies = np.array(accuracies)

fig, ax1 = plt.subplots(figsize=(18, 10))

ax1.plot(k_values, accuracies*100)
ax1.axhline(50.0, ls='--', c='orange', label='chance')
# plt.axhline(0.5, ls='--', c='orange', label='chance')
ax1.axvline(len(B_selected_variables), ls='--',
            label='p-value selected', c='forestgreen')
ax1.set_xlim([0,200])
ax1.set_ylim([0,105])
ax1.set_xlabel('Top k selected')
ax1.set_ylabel('Accuracy (%)')
ax1.legend();

# Choose the number of selected variables that maximize the prediction accuracy
CV_error_selected = k_values[np.argmax(accuracies)]

fig, ax1 = plt.subplots(figsize=(18, 10))
ax1.plot(k_values, accuracies*100)
ax1.axhline(50.0, ls='--', c='orange', label='chance')
# ax1.axvline(len(B_selected_variables), ls='--',
#             label='p-value selected {}'.format(len(B_selected_variables)), c='forestgreen')
ax1.axvline(CV_error_selected, ls='--',
            label='CV-error selected {}'.format(CV_error_selected), c='indianred')

ax1.set_xlim([0,200])
ax1.set_ylim([0,105])
ax1.set_xlabel('Top k selected')
ax1.set_ylabel('Accuracy (%)')
ax1.legend();

# Restrict the data matrix to the first CV_error_selected variables
X_tr_small = X_tr[:, sorted_variables[:CV_error_selected]]

# Fit a linear classifier on the training set
clf.fit(X_tr_small, y_tr)

# Predict the test set and measure the prediction accuracy
y_pred = clf.predict(X_ts[:, sorted_variables[:CV_error_selected]])
print('Out-of-samples prediction accuracy: {}%'.format(100*accuracy_score(y_ts, y_pred)))



