get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import requests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from lob_data_utils.roc_results import results_10000 as results
from lob_data_utils import lob
from lob_data_utils import db_result

result = db_result.get_svm_results_for_data_length(10000, 'cv')

df = pd.DataFrame(result)
print('Data length: ', len(df))
df.drop('algorithm_id', axis=1, inplace=True)
df.drop('svm_id', axis=1, inplace=True)
df.drop('name', axis=1, inplace=True)
df.drop('id', axis=1, inplace=True)
df.drop('data_type', axis=1, inplace=True)
df.drop('data_length', axis=1, inplace=True)

log_res = []
for i, row in df.iterrows():
    log_res.append(results.get(row['stock']))
df['log_res'] = log_res
df['log_diff'] = df['roc_auc_score'] - log_res
df.head()

bests = []
df_best_agg = df.groupby('stock', as_index=False)['roc_auc_score'].idxmax()
df_bests = df.loc[df_best_agg]
df_bests.index = df_bests['stock']
df_bests.sort_values(by='roc_auc_score', inplace=True)
df_bests.head()

df_bests.groupby(['kernel'])[['kernel']].count().plot(kind='bar')
plt.title('Kernels count')
plt.ylabel('count')
plt.xlabel('kernel')

df_bests[df_bests['kernel'] == 'rbf'].groupby(['c', 'gamma'])[['kernel']].count().plot(kind='bar')
plt.title('RBF best kernel parameters counts')
plt.ylabel('count')
plt.xlabel('parameter $c$ and $\gamma$')

df_bests[df_bests['kernel'] == 'sigmoid'].groupby(['c', 'gamma', 'coef0'])[['coef0']].count().plot(
    kind='bar')
plt.title('Sigmoid best kernel parameters counts')
plt.ylabel('count')
plt.xlabel('parameter $c$, $\gamma and coef0 $')

stocks = df_bests['stock'].values
plt.plot(np.zeros(len(stocks)) + 0.5, 'r--', label='null hypothesis')
df_bests['roc_auc_score'].plot(marker='.', label='best svm')
print('min: ', df_bests['roc_auc_score'].min(), 'max:',  df_bests['roc_auc_score'].max())
plt.ylabel('score')
plt.xlabel('stock')
plt.legend()
plt.ylim(0.4, 0.7)
plt.title('Best svm roc area score vs null hypothesis')

df_bests['roc_auc_score'].plot('hist')
plt.xlabel('ROC area score')
plt.legend()
plt.title('Frequencies of ROC scores')

print('Number of better SVMs:', 
      len(df[df['log_res'] < df['roc_auc_score']]['stock'].unique()), 'per', len(df['stock'].unique()))

stocks = df_bests['stock'].values
plt.plot(np.zeros(len(stocks)) + 0.5, 'r--', label='null hypothesis')
df_bests['roc_auc_score'].plot(marker='.', label='best svm', figsize=(16, 8))
df_bests['log_res'].plot(marker='.', label='logistic regression')

plt.ylabel('score')
plt.legend()
plt.ylim(0.45, 0.65)
plt.title('Best svm roc area score vs logistic regression')

df_bests['roc_auc_score'].plot('kde', label='best svm score')
df_bests['log_res'].plot('kde', label='logistic regression score')
plt.title('Density of scores')
plt.xlabel('score')
plt.legend()

df_bests['log_diff'].plot(kind='hist')
plt.title('Difference between logistic regression and best choice SVM')
plt.xlabel('score diff')

df_best_agg = df.groupby(['stock', 'kernel'], as_index=True)['roc_auc_score'].idxmax()
df_bests_by_kernels = df.loc[df_best_agg]
df_bests_by_kernels.head(9)

df_kernels = df_bests_by_kernels.pivot(index='stock', columns='kernel', values='roc_auc_score')
df_kernels.sort_index(inplace=True)
df_kernels.head()

df_kernels.sort_values(by=['linear', 'sigmoid', 'rbf']).plot(figsize=(16, 8))
plt.plot(np.zeros(len(df_kernels)) + 0.5, 'y--', label='null hypothesis')
plt.ylabel('score')
plt.legend()
plt.ylim(0.45, 0.65)
plt.title('Best svm kernels vs null hypothesis by kernels')

df_bests.sort_index(inplace=True)
df_kernels['linear'].plot(figsize=(16,8))
df_bests['log_res'].plot()
plt.plot(np.zeros(len(df_kernels)) + 0.5, 'y--', label='null hypothesis')

plt.ylabel('score')
plt.legend()
plt.ylim(0.45, 0.65)
plt.title('Best linear kernels SVM vs logistic regression')

(df_kernels['linear'] - df_bests['log_res']).plot(kind='hist')
plt.xlabel('Score difference')
plt.title('Linear SVM kernels vs logistic regression - score difference freqency')

df_bests.sort_index(inplace=True)
df_kernels['rbf'].plot(figsize=(16,8))
df_bests['log_res'].plot()
plt.plot(np.zeros(len(df_kernels)) + 0.5, 'y--', label='null hypothesis')

plt.ylabel('Score')
plt.legend()
plt.ylim(0.45, 0.65)
plt.title('Best rbf kernels SVM vs logistic regression')

(df_kernels['rbf'] - df_bests['log_res']).plot(kind='hist')
plt.xlabel('score difference')
plt.title('RBF SVM kernels vs logistic regression - score difference freqency')

df_bests.sort_index(inplace=True)
df_kernels['sigmoid'].plot(figsize=(16,8))
df_bests['log_res'].plot()
plt.plot(np.zeros(len(df_kernels)) + 0.5, 'y--', label='null hypothesis')

plt.ylabel('score')
plt.legend()
plt.ylim(0.45, 0.65)
plt.title('Best sigmoid kernels SVM vs logistic regression')

(df_kernels['sigmoid'] - df_bests['log_res']).plot(kind='hist')
plt.xlabel('Score difference')
plt.title('Sigmoid SVM kernels vs logistic regression - score difference freqency')

