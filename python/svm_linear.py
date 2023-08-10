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

def svm_classification(d, kernel, gamma='auto', C=1.0, degree=3, coef0=0.0, decision_function_shape='ovr'):
    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, degree=degree, coef0=coef0, 
                  decision_function_shape=decision_function_shape)
    X = d['queue_imbalance'].values.reshape(-1, 1)
    y = d['mid_price_indicator'].values.reshape(-1, 1)
    clf.fit(X, y)
    return clf

cs = [0.0001, 0.001, 0.00305, 0.0031, 0.0032, 0.0033, 
      0.00335, 0.00337, 0.00338, 0.00339, 0.0034, 0.00342, 0.00345,
      0.00347, 0.00348,  0.00349,
      0.0035, 0.00352, 0.0036, 0.005, 0.006, 
      0.012, 0.013, 0.014, 0.015, 0.0155, 0.016, 
      0.017, 0.018, 0.02, 0.23, 0.04, 0.05, 0.06,
      0.1, 0.5, 1, 1.5, 10, 80, 85, 90, 93, 94, 95, 96, 97, 98, 
      99, 100, 101, 110, 1000]

df_css = {}

ax = plt.subplot()
ax.set_xscale("log", basex=10)
for s in stocks:
    df_cs = pd.DataFrame(index=cs)
    df_cs['roc'] = np.zeros(len(df_cs))
    for c in cs:
        reg_svm = svm_classification(dfs[s], 'linear', C=c)
        pred_svm_out_of_sample = reg_svm.predict(dfs_cv[s]['queue_imbalance'].values.reshape(-1, 1))
        logit_roc_auc = roc_auc_score(dfs_cv[s]['mid_price_indicator'], pred_svm_out_of_sample)
        df_cs.loc[c] = logit_roc_auc
    plt.plot(df_cs, linestyle='--', label=s, marker='x', alpha=0.6)
    df_css[s] = df_cs
plt.legend()
plt.xlabel('C parameter')
plt.ylabel('roc_area value')
plt.title('roc_area vs C')

for s in stocks:
    idx = df_css[s]['roc'].idxmax()
    print('For {} the best is {}'.format(s, idx))

for s in stocks:
    err_max = df_css[s]['roc'].max()
    err_min = df_css[s]['roc'].min()
    print('For {} the diff between best and worst {}'.format(s, err_max - err_min))

df_results = pd.DataFrame(index=stocks)
df_results['logistic'] = np.zeros(len(stocks))
df_results['linear-default'] = np.zeros(len(stocks))
df_results['linear-tunned'] = np.zeros(len(stocks))
df_results['C'] = np.zeros(len(stocks))

plt.subplot(121)
for s in stocks:
    reg_svm = svm_classification(dfs[s], 'linear', C=df_css[s].idxmax())
    roc_score = lob.plot_roc(df_test, reg_svm, stock=s, title='ROC for test set with the best C param')
    df_results['linear-tunned'][s] = roc_score
    df_results['C'][s] = df_css[s].idxmax()

plt.subplot(122)
for s in stocks:
    reg_svm = svm_classification(dfs[s], 'linear')
    roc_score = lob.plot_roc(df_test, reg_svm, stock=s, title='ROC for test set with defaults')
    df_results['linear-default'][s] = roc_score

plt.subplots_adjust(left=0, wspace=0.1, top=1, right=2)

plt.subplot(121)

for s in stocks:
    reg_svm = svm_classification(dfs[s], 'linear')
    roc_score = lob.plot_roc(df_test, reg_svm, stock=s)
    df_results['linear-tunned'][s] = roc_score

plt.subplot(122)
for s in stocks:
    reg_log = lob.logistic_regression(dfs[s], 0, len(dfs[s]))
    roc_score = lob.plot_roc(df_test, reg_log, stock=s, title='Logistic classication')
    df_results['logistic'][s] = roc_score

plt.subplots_adjust(left=0, wspace=0.1, top=1, right=2)

df_results

