get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as md
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn import svm
import warnings

from lob_data_utils import lob

sns.set_style('whitegrid')

warnings.filterwarnings('ignore')

stocks = ['10795', '12098', '11618', '2051', '4481', '3107', '1243', '11234'][:5]

dfs = {}
dfs_cv = {}
dfs_test = {}

for s in stocks:
    df, df_cv, df_test = lob.load_data(s, data_dir='data/INDEX/', cv=True)
    dfs[s] = df
    dfs_cv[s] = df_cv
    dfs_test[s] = df_test

dfs[stocks[0]].head()

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

def svm_classification(d, kernel, gamma='auto', C=1.0, degree=3, coef0=0.0):
    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, degree=degree, coef0=coef0)
    X = d['queue_imbalance'].values.reshape(-1, 1)
    y = d['mid_price_indicator'].values.reshape(-1, 1)
    clf.fit(X, y)
    return clf

reg_log = {}
pred_log = {}
pred_out_of_sample = {}
for s in stocks:
    reg_log[s] = lob.logistic_regression(dfs[s], 0, len(dfs[s]))
    pred_log[s] = reg_log[s].predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))

s = '10795'
reg_svm = svm_classification(dfs[s], 'linear', C=0.005)
pred_svm_out_of_sample = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_svm_out_of_sample)
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_svm_out_of_sample)
plt.plot(fpr, tpr, label='{} (area = {})'.format('linear C=0.005', logit_roc_auc))

reg_svm = svm_classification(dfs[s], 'rbf')
pred_svm_out_of_sample = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_svm_out_of_sample)
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_svm_out_of_sample)
plt.plot(fpr, tpr, label='{} (area = {})'.format('rbf default', logit_roc_auc))

reg_svm = svm_classification(dfs[s], 'rbf', C=0.1, gamma=0.5)
pred_svm_out_of_sample = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_svm_out_of_sample)
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_svm_out_of_sample)
plt.plot(fpr, tpr, label='{} (area = {})'.format('rbf C=0.1 g=0.5', logit_roc_auc))

logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_log[s])
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_log[s])
plt.plot(fpr, tpr, label='{} (area = {})'.format('logisitc', logit_roc_auc))

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

s = '12098'
reg_svm = svm_classification(dfs[s], 'linear', C=0.005)
pred_svm_out_of_sample = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_svm_out_of_sample)
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_svm_out_of_sample)
plt.plot(fpr, tpr, label='{} (area = {})'.format('linear C=0.005', logit_roc_auc))

reg_svm = svm_classification(dfs[s], 'rbf', C=1.5, gamma=0.001)
pred_svm_out_of_sample = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_svm_out_of_sample)
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_svm_out_of_sample)
plt.plot(fpr, tpr, label='{} (area = {})'.format('rbf C=1.5 gamma=0.001', logit_roc_auc))

logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_log[s])
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_log[s])
plt.plot(fpr, tpr, label='{} (area = {})'.format('logisitc', logit_roc_auc))

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

s = '11618'

reg_svm = svm_classification(dfs[s], 'linear')
pred_svm_out_of_sample = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_svm_out_of_sample)
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_svm_out_of_sample)
plt.plot(fpr, tpr, label='{} (area = {})'.format('linear default', logit_roc_auc))

reg_svm = svm_classification(dfs[s], 'rbf', C=1000, gamma=0.1)
pred_svm_out_of_sample = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_svm_out_of_sample)
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_svm_out_of_sample)
plt.plot(fpr, tpr, label='{} (area = {})'.format('rbf C=1000 g=0.1', logit_roc_auc))

reg_svm = svm_classification(dfs[s], 'rbf')
pred_svm_out_of_sample = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_svm_out_of_sample)
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_svm_out_of_sample)
plt.plot(fpr, tpr, label='{} (area = {})'.format('rbf defalt', logit_roc_auc))

reg_svm = svm_classification(dfs[s], 'sigmoid', C=0.1, gamma=0.1, coef0=0.5)
pred_svm_out_of_sample = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_svm_out_of_sample)
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_svm_out_of_sample)
plt.plot(fpr, tpr, label='{} (area = {})'.format('sigmoid coef0=0.5', logit_roc_auc))

logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_log[s])
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_log[s])
plt.plot(fpr, tpr, label='{} (area = {})'.format('logisitc', logit_roc_auc))

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

s = '2051'
reg_svm = svm_classification(dfs[s], 'sigmoid', C=0.01, gamma=1000, coef0=0.5)
pred_svm_out_of_sample = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_svm_out_of_sample)
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_svm_out_of_sample)
plt.plot(fpr, tpr, label='{} (area = {})'.format('sigmoid C=0.01 g=1000 coef0=0.5', logit_roc_auc))

reg_svm = svm_classification(dfs[s], 'linear')
pred_svm_out_of_sample = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_svm_out_of_sample)
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_svm_out_of_sample)
plt.plot(fpr, tpr, label='{} (area = {})'.format('linear default', logit_roc_auc))

reg_svm = svm_classification(dfs[s], 'rbf')
pred_svm_out_of_sample = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_svm_out_of_sample)
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_svm_out_of_sample)
plt.plot(fpr, tpr, label='{} (area = {})'.format('rbf default', logit_roc_auc))


logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_log[s])
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_log[s])
plt.plot(fpr, tpr, label='{} (area = {})'.format('logisitc', logit_roc_auc))

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

s = '4481'
reg_svm = svm_classification(dfs[s], 'rbf', C=1.5, gamma=0.1)
pred_svm_out_of_sample = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_svm_out_of_sample)
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_svm_out_of_sample)
plt.plot(fpr, tpr, label='{} (area = {})'.format('rbf C=1.5 g=0.1', logit_roc_auc))

reg_svm = svm_classification(dfs[s], 'rbf', C=10, gamma=0.001)
pred_svm_out_of_sample = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_svm_out_of_sample)
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_svm_out_of_sample)
plt.plot(fpr, tpr, label='{} (area = {})'.format('rbf C=10 g=0.001', logit_roc_auc))

reg_svm = svm_classification(dfs[s], 'sigmoid', C=0.1, gamma=0.1, coef0=0.05)
pred_svm_out_of_sample = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_svm_out_of_sample)
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_svm_out_of_sample)
plt.plot(fpr, tpr, label='{} (area = {})'.format('sigmoid coef0=0.05', logit_roc_auc))

logit_roc_auc = roc_auc_score(dfs_test[s]['mid_price_indicator'], pred_log[s])
fpr, tpr, thresholds = roc_curve(dfs_test[s]['mid_price_indicator'].values, pred_log[s])
plt.plot(fpr, tpr, label='{} (area = {})'.format('logisitc', logit_roc_auc))

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

