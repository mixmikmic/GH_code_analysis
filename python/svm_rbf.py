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

stocks = ['10795', '12098', '11618', '1243', '11234']

dfs = {}
dfs_cv = {}
dfs_test = {}

for s in stocks:
    df, df_cv, df_test = lob.load_prepared_data(s, cv=True)
    dfs[s] = df
    dfs_cv[s] = df_cv
    dfs_test[s] = df_test

dfs[stocks[0]].head(5)

def svm_classification(d, kernel, gamma='auto', C=1.0):
    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C)
    X = d['queue_imbalance'].values.reshape(-1, 1)
    y = d['mid_price_indicator'].values.reshape(-1, 1)
    clf.fit(X, y)
    return clf

cs = [0.08, 0.09, 0.1, 0.12, 1, 1.25, 1.9, 2, 6.4, 6.5, 6.6, 
      7, 7.1, 107, 108, 
      108.5]
# 0.06, 0.07
# 6.3, 6.7
# 7.2, 7.5, 8, 9
# 109, 110, 111

df_css = {}

ax = plt.subplot()
ax.set_xscale("log", basex=10)
for s in stocks:
    df_cs = pd.DataFrame(index=cs)
    df_cs['roc'] = np.zeros(len(df_cs))
    for c in cs:
        reg_svm = svm_classification(dfs[s], 'rbf', C=c)
        prediction = reg_svm.predict(dfs_cv[s]['queue_imbalance'].values.reshape(-1, 1))
        score = roc_auc_score(dfs_cv[s]['mid_price_indicator'], prediction)
        df_cs.loc[c] = score
    plt.plot(df_cs, linestyle='--', label=s, marker='x', alpha=0.5)
    df_css[s] = df_cs
    
plt.legend()

for s in stocks:
    idx = df_css[s]['roc'].idxmax()
    print('For {} the best is {}'.format(s, idx))

for s in stocks:
    err_max = df_css[s]['roc'].max()
    err_min = df_css[s]['roc'].min()
    print('For {} the diff between best and worst {}'.format(s, err_max - err_min))

gammas = [0.0008, 0.001, 0.09, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 100.5, 101, 101.5]

# 0.1
# 102
# 1, 10, 99

df_gammas = {}

ax = plt.subplot()
ax.set_xscale("log", basex=10)
for s in stocks:
    df_gamma = pd.DataFrame(index=gammas)
    df_gamma['roc'] = np.zeros(len(df_gamma))
    for g in gammas:
        reg_svm = svm_classification(dfs[s], 'rbf', gamma=g)
        pred_svm_out_of_sample = reg_svm.predict(dfs_cv[s]['queue_imbalance'].values.reshape(-1, 1))

        logit_roc_auc = roc_auc_score(dfs_cv[s]['mid_price_indicator'], pred_svm_out_of_sample)
        df_gamma.loc[g] = logit_roc_auc
    plt.plot(df_gamma, linestyle='--', label=s, marker='x', alpha=0.7)
    df_gammas[s] = df_gamma
    
plt.legend()

for s in stocks:
    idx = df_gammas[s]['roc'].idxmax()
    print('For {} the best is {}'.format(s, idx))

for s in stocks:
    err_max = df_gammas[s]['roc'].max()
    err_min = df_gammas[s]['roc'].min()
    print('For {} the diff between best and worst {}'.format(s, err_max - err_min))

df_results = pd.DataFrame(index=stocks)
df_results['logistic'] = np.zeros(len(stocks))
df_results['rbf-naive'] = np.zeros(len(stocks))
df_results['gamma-naive'] = np.zeros(len(stocks))
df_results['c-naive'] = np.zeros(len(stocks))
df_results['rbf-default'] = np.zeros(len(stocks))

plt.subplot(121)

for s in stocks:
    reg_svm = svm_classification(dfs[s], 'rbf', C=df_css[s]['roc'].idxmax(), 
                                 gamma=df_gammas[s]['roc'].idxmax())
    roc_score = lob.plot_roc(df_test, reg_svm, stock=s, title='ROC for test set with the naive')
    df_results['rbf-naive'][s] = roc_score
    df_results['gamma-naive'][s] = df_gammas[s]['roc'].idxmax()
    df_results['c-naive'][s] = df_css[s]['roc'].idxmax()
    
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
plt.subplot(122)
for s in stocks:
    reg_svm = svm_classification(dfs[s], 'rbf')
    roc_score = lob.plot_roc(df_test, reg_svm, stock=s, title='ROC for test set with the defaults')
    df_results['rbf-default'][s] = roc_score
    
    reg_log = lob.logistic_regression(dfs[s], 0, len(dfs[s]))
    roc_score = lob.plot_roc(df_test, reg_log, stock=s, title='ROC for test set with logistic', 
                             c=colors[stocks.index(s)], linestyle='--')
    df_results['logistic'][s] = roc_score

plt.subplots_adjust(left=0, wspace=0.1, top=1, right=2)

df_results

df_params = {}

for s in stocks:
    print(s)
    params = []
    for c in cs:
        for g in gammas:
            reg_svm = svm_classification(dfs[s], 'rbf', C=c, gamma=g)
            prediction = reg_svm.predict(dfs_cv[s]['queue_imbalance'].values.reshape(-1, 1))
            score = roc_auc_score(dfs_cv[s]['mid_price_indicator'], prediction)
            params.append({'score': score, 'gamma': g, 'c': c})
    df_params[s] = pd.DataFrame(params)

for s in stocks:
    df_g = df_params[s].pivot(index='c', columns='gamma', values='score')
    sns.heatmap(df_g)
    plt.title('Best params for ' + s)
    plt.figure()

for s in stocks:
    print(s, df_params[s].iloc[df_params[s]['score'].idxmax()])

df_results['rbf-grid'] = np.zeros(len(stocks))
df_results['c-grid'] = np.zeros(len(stocks))
df_results['gamma-grid'] = np.zeros(len(stocks))
plt.subplot(121)
for s in stocks:
    best_idx = df_params[s]['score'].idxmax()
    reg_svm = svm_classification(dfs[s], 'rbf', C=df_params[s].iloc[best_idx]['c'], 
                                 gamma=df_params[s].iloc[best_idx]['gamma'])
    roc_score = lob.plot_roc(df_test, reg_svm, stock=s, title='ROC for test set with the best params')
    df_results['rbf-grid'][s] = roc_score
    df_results['gamma-grid'][s] = df_params[s].iloc[best_idx]['gamma']
    df_results['c-grid'][s] = df_params[s].iloc[best_idx]['c']

plt.subplot(122)
for s in stocks:
    reg_svm = svm_classification(dfs[s], 'rbf')
    prediction = reg_svm.predict(dfs_test[s]['queue_imbalance'].values.reshape(-1, 1))
    roc_score = lob.plot_roc(df_test, reg_svm, stock=s, title='ROC for test set with defaults')
    df_results['rbf-default'][s] = roc_score
plt.subplots_adjust(left=0, wspace=0.1, top=1, right=2)

plt.subplot(121)
for s in stocks:
    best_idx = df_params[s]['score'].idxmax()
    reg_svm = svm_classification(dfs[s], 'rbf', C=df_params[s].iloc[best_idx]['c'], 
                                 gamma=df_params[s].iloc[best_idx]['gamma'])
    roc_score = lob.plot_roc(df_test, reg_svm, stock=s, title='ROC for test set with the best params')
    df_results['rbf-grid'][s] = roc_score

plt.subplot(122)
for s in stocks:
    reg_log = lob.logistic_regression(dfs[s], 0, len(dfs[s]))
    roc_score = lob.plot_roc(df_test, reg_log, stock=s, title='ROC for test set with the best params')
    df_results['logistic'][s] = roc_score

plt.subplots_adjust(left=0, wspace=0.1, top=1, right=2)

df_results[['logistic', 'rbf-naive', 'rbf-default', 'rbf-grid']]

df_results

