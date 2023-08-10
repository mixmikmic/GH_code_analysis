get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import requests
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from lob_data_utils import roc_results
from lob_data_utils import lob
from lob_data_utils import db_result

data_length = 10000

results = db_result.get_svm_results_for_data_length(data_length, 'test')
result_cv = db_result.get_svm_results_for_data_length(data_length, 'cv')

df_cv = pd.DataFrame(result_cv)

df_cv.drop('algorithm_id', axis=1, inplace=True)
df_cv.drop('svm_id', axis=1, inplace=True)
df_cv.drop('name', axis=1, inplace=True)
df_cv.drop('id', axis=1, inplace=True)
df_cv.drop('data_length', axis=1, inplace=True)
df_cv.head()

bests = []
df_best_agg = df_cv.groupby('stock', as_index=False)['roc_auc_score'].idxmax()
df_bests = df_cv.loc[df_best_agg]
df_bests.index = df_bests['stock']

test_roc_auc_scores = []
for i, row in df_bests.iterrows():
    res = db_result.get_svm_results_by_params(
            row['stock'], row['kernel'], data_type='test', data_length=data_length, 
            gamma=row['gamma'], c=row['c'], coef0=row['coef0'])
    test_roc_auc_scores.append(res[0].get('roc_auc_score'))

df_bests['test_roc_auc_score'] = test_roc_auc_scores 

log_res = []
for i, row in df_bests.iterrows():
    log_res.append(roc_results.result_test_10000.get(row['stock']))
df_bests['test_log_roc_auc_score'] = log_res
df_bests['diff'] = df_bests['test_roc_auc_score'] - log_res
df_bests.sort_values(by='roc_auc_score', inplace=True)
df_bests.head()

stocks = df_bests['stock'].values
plt.plot(np.zeros(len(stocks)) + 0.5, 'r--', label='null hypothesis')
df_bests['test_roc_auc_score'].plot(marker='.', label='best svm on test set')
df_bests['roc_auc_score'].plot(marker='.', label='best svm on validation set')
print('Testing - min: ', df_bests['test_roc_auc_score'].min(), 'max:',  df_bests['test_roc_auc_score'].max())
print('Validation - min: ', df_bests['roc_auc_score'].min(), 'max:',  df_bests['roc_auc_score'].max())
plt.ylabel('Score')
plt.legend()
plt.ylim(0.45, 0.65)
plt.title('Best svm roc area score vs null hypothesis')

df_bests['test_roc_auc_score'].plot('hist')
plt.title('Scores frequency for test set')
plt.xlabel('ROC area score')

(df_bests['test_roc_auc_score'] - df_bests['roc_auc_score']).plot(kind='hist')
plt.title('Frequency of differences between score on testing and validation sets')
plt.xlabel('ROC area score difference')

df_bests['test_roc_auc_score'].plot('kde', label='best svm score on testing set')
df_bests['roc_auc_score'].plot('kde', label='best svm score on validation set')
plt.title('Density of scores')
plt.legend()

print('Number of better SVMs:', 
      len(df_bests[df_bests['test_log_roc_auc_score'] < df_bests['test_roc_auc_score']]['stock'].unique()), 
      'per', len(df_bests['stock'].unique()))

stocks = df_bests['stock'].values
plt.plot(np.zeros(len(stocks)) + 0.5, 'r--', label='null hypothesis')
df_bests['test_roc_auc_score'].plot(marker='.', label='best svm on test set', figsize=(16, 8))
df_bests['test_log_roc_auc_score'].plot(marker='.', label='logistic regression on test set')

plt.ylabel('Score')
plt.legend()
plt.ylim(0.45, 0.65)
plt.title('Best svm roc area score vs logistic regression')

df_bests['test_roc_auc_score'].plot('kde', label='roc area score on testing set')
df_bests['test_log_roc_auc_score'].plot('kde', label='logistic regression score on testing set')
plt.title('Density of scores')
plt.xlabel('ROC area score')
plt.legend()

df_bests['diff'].plot(kind='hist')
plt.title('Difference between logistic regression and best choice SVM on testing sets')
plt.xlabel('Score difference')

df_best_agg = df_cv.groupby(['stock', 'kernel'], as_index=False)['roc_auc_score'].idxmax()
df_bests_by_kernels = df_cv.loc[df_best_agg]

test_roc_auc_scores = []
for i, row in df_bests_by_kernels.iterrows():
    res = db_result.get_svm_results_by_params(
            row['stock'], row['kernel'], data_type='test', data_length=data_length, 
            gamma=row['gamma'], c=row['c'], coef0=row['coef0'])
    test_roc_auc_scores.append(res[0].get('roc_auc_score'))

df_bests_by_kernels['test_roc_auc_score'] = test_roc_auc_scores 

log_res = []
for i, row in df_bests_by_kernels.iterrows():
    log_res.append(roc_results.result_test_10000.get(row['stock']))
df_bests_by_kernels['test_log_roc_auc_score'] = log_res
df_bests_by_kernels['diff'] = df_bests_by_kernels['test_roc_auc_score'] - log_res
df_bests_by_kernels.head()

df_kernels_test = df_bests_by_kernels.pivot(index='stock', columns='kernel', values='test_roc_auc_score')
df_kernels_test.sort_index(inplace=True)
df_kernels_test.head()

df_kernels_val = df_bests_by_kernels.pivot(index='stock', columns='kernel', values='roc_auc_score')
df_kernels_val.sort_index(inplace=True)
df_kernels_val.head()

f, ax = plt.subplots(figsize=(8, 8))
sns.violinplot(x='kernel', 
           y='roc_auc_score', data=df_bests_by_kernels, cut=2, bw=0.2)
sns.despine(left=True)
plt.title('Kernel distribution against roc auc score on validation set')

df_kernels_test.sort_values(by=['linear', 'sigmoid', 'rbf']).plot(figsize=(16, 8))
plt.plot(np.zeros(len(df_kernels_test)) + 0.5, 'y--', label='null hypothesis')
plt.ylabel('Score')
plt.legend()
plt.ylim(0.45, 0.65)
plt.title('Scores on testing set for different kernels')

(df_kernels_test['linear'] - 0.5).plot(kind='hist')
plt.xlabel('Score difference')
plt.legend()
plt.title('Score difference frequency between linear kernels and null hypothesis')

df_bests.sort_index(inplace=True)
df_kernels_test['linear'].plot(figsize=(16,8), label='SVM with linear kernel')
df_bests['test_log_roc_auc_score'].plot(label='logistic regression')
plt.plot(np.zeros(len(df_kernels_test)) + 0.5, 'y--', label='null hypothesis')

plt.ylabel('Score')
plt.legend()
plt.ylim(0.45, 0.65)
plt.title('Linear SVM kernels vs logistic regression')

(df_kernels_test['linear'] - df_bests['test_log_roc_auc_score']).plot(kind='hist', label='linear')
plt.xlabel('score difference')
plt.legend()
plt.title('Score difference between linear kernel and logistic regression')

df_bests.sort_index(inplace=True)
df_kernels_test['linear'].plot(figsize=(16,8), label='linear kernel on testing')
df_kernels_val['linear'].plot(figsize=(16,8), label='linear kernel on validation')
plt.plot(np.zeros(len(df_kernels_test)) + 0.5, 'y--', label='null hypothesis')

plt.ylabel('score')
plt.legend()
plt.ylim(0.45, 0.65)
plt.title('Best linear SVM kernels on testing and validation sets')

(df_kernels_test['linear'] - df_kernels_val['linear']).plot(kind='hist')
(df_kernels_test['linear'] - df_kernels_val['linear']).plot(kind='kde')
plt.xlabel('score difference')
plt.legend()
plt.title('Score difference between linear kernels on testing set and validation set')

(df_kernels_test['rbf'] - 0.5).plot(kind='hist')
plt.xlabel('Score difference')
plt.legend()
plt.title('Score difference between rbf kernels and null hypothesis')

df_bests.sort_index(inplace=True)
df_kernels_test['rbf'].plot(figsize=(16,8), label='SVM with rbf kernel')
df_bests['test_log_roc_auc_score'].plot(label='logistic regression')
plt.plot(np.zeros(len(df_kernels_test)) + 0.5, 'y--', label='null hypothesis')

plt.ylabel('Score')
plt.legend()
plt.ylim(0.45, 0.65)
plt.title('RBF SVM kernels vs logistic regression')

(df_kernels_test['rbf'] - df_bests['test_log_roc_auc_score']).plot(kind='hist', label='rbf')
plt.xlabel('score difference')
plt.legend()
plt.title('Score difference between rbf kernel and logistic regression')

df_bests.sort_index(inplace=True)
df_kernels_test['rbf'].plot(figsize=(16,8), label='rbf kernel on testing')
df_kernels_val['rbf'].plot(figsize=(16,8), label='rbf kernel on validation')
plt.plot(np.zeros(len(df_kernels_test)) + 0.5, 'y--', label='null hypothesis')

plt.ylabel('score')
plt.legend()
plt.ylim(0.45, 0.65)
plt.title('Best rbf SVM kernels on testing and validation sets')

(df_kernels_test['rbf'] - df_kernels_val['rbf']).plot(kind='hist')
(df_kernels_test['rbf'] - df_kernels_val['rbf']).plot(kind='kde')
plt.xlabel('score difference')
plt.legend()
plt.title('Score difference between rbf kernels on testing set and validation set')

(df_kernels_test['sigmoid'] - 0.5).plot(kind='hist')
plt.xlabel('Score difference')
plt.legend()
plt.title('Score difference between sigmoid kernel and null hypothesis')

df_bests.sort_index(inplace=True)
df_kernels_test['sigmoid'].plot(figsize=(16,8), label='SVM with sigmoid kernel')
df_bests['test_log_roc_auc_score'].plot(label='logistic regression')
plt.plot(np.zeros(len(df_kernels_test)) + 0.5, 'y--', label='null hypothesis')

plt.ylabel('score')
plt.legend()
plt.ylim(0.45, 0.65)
plt.title('Sigmoid SVM kernels vs logistic regression')

(df_kernels_test['sigmoid'] - df_bests['test_log_roc_auc_score']).plot(kind='hist', label='sigmoid')
plt.xlabel('score difference')
plt.legend()
plt.title('Score difference between sigmoid kernel and logistic regression')

df_bests.sort_index(inplace=True)
df_kernels_test['sigmoid'].plot(figsize=(16,8), label='sigmoid kernel on testing')
df_kernels_val['sigmoid'].plot(figsize=(16,8), label='sigmoid kernel on validation')
plt.plot(np.zeros(len(df_kernels_test)) + 0.5, 'y--', label='null hypothesis')

plt.ylabel('score')
plt.legend()
plt.ylim(0.45, 0.65)
plt.title('Best sigmoid SVM kernels on testing and validation sets')

(df_kernels_test['sigmoid'] - df_kernels_val['sigmoid']).plot(kind='hist')
(df_kernels_test['sigmoid'] - df_kernels_val['sigmoid']).plot(kind='kde')
plt.xlabel('score difference')
plt.legend()
plt.title('Score difference between sigmoid kernels on testing set and validation set')

