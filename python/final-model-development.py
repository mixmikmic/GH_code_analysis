from __future__ import print_function

# Import libraries
import numpy as np
import pandas as pd
import matplotlib
import sklearn
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties # for unicode fonts
import sys
import datetime as dt
import mp_utils as mp

from collections import OrderedDict

# used to print out pretty pandas dataframes
from IPython.display import display, HTML

from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# used to impute mean for data and standardize for computational stability
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

# logistic regression is our favourite model ever
from sklearn import linear_model
from sklearn import ensemble

# used to calculate AUROC/accuracy
from sklearn import metrics

# used to create confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

# gradient boosting - must download package https://github.com/dmlc/xgboost
import xgboost as xgb

#from eli5 import show_weights

# default colours for prettier plots
col = [[0.9047, 0.1918, 0.1988],
    [0.2941, 0.5447, 0.7494],
    [0.3718, 0.7176, 0.3612],
    [1.0000, 0.5482, 0.1000],
    [0.4550, 0.4946, 0.4722],
    [0.6859, 0.4035, 0.2412],
    [0.9718, 0.5553, 0.7741],
    [0.5313, 0.3359, 0.6523]];
marker = ['v','o','d','^','s','o','+']
ls = ['-','-','-','-','-','s','--','--']

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('X_design_matrix.csv').set_index('icustay_id')

# create X by dropping idxK and the outcome
y = df['death'].values
idxK = df['idxK']
X = df.drop(['death','idxK'],axis=1).values

print('{} observations. Outcome rate: {:2.2f}%.'.format(X.shape[0],
                                                        100.0*np.mean(y)))

# Rough timing info:
#     rf - 3 seconds per fold
#    xgb - 30 seconds per fold
# logreg - 4 seconds per fold
#  lasso - 8 seconds per fold
np.random.seed(7390984)

# parameters from grid search
xgb_mdl = xgb.XGBClassifier(colsample_bytree=0.7, silent=1,
                            learning_rate = 0.01, n_estimators=1000,
                            subsample=0.8, max_depth=9)

models = {'xgb': xgb_mdl,
          'lasso': linear_model.LassoCV(cv=5,fit_intercept=True,normalize=True,max_iter=10000),
          'logreg': linear_model.LogisticRegression(fit_intercept=True),
          'l2': linear_model.LogisticRegressionCV()
          #'rf': ensemble.RandomForestClassifier()
         }


# create k-fold indices
K = 5 # number of folds
idxK = np.random.permutation(X.shape[0])
idxK = np.mod(idxK,K)

mdl_val = dict()
results_val = dict()
pred_val = dict()
pred_val_merged = dict()
for mdl in models:
    print('=============== {} ==============='.format(mdl))
    mdl_val[mdl] = list()
    results_val[mdl] = list() # initialize list for scores
    pred_val[mdl] = dict()
    pred_val_merged[mdl] = np.zeros(X.shape[0])
    
    if mdl == 'xgb':
        # no pre-processing of data necessary for xgb
        estimator = Pipeline([(mdl, models[mdl])])

    else:
        estimator = Pipeline([("imputer", Imputer(missing_values='NaN',
                                          strategy="mean",
                                          axis=0)),
                      ("scaler", StandardScaler()),
                      (mdl, models[mdl])]) 

    for k in range(K):
        # train the model using all but the kth fold
        curr_mdl = sklearn.base.clone(estimator).fit(X[idxK != k, :], y[idxK != k])

        # get prediction on this dataset
        if mdl in ('lasso','ridge'):
            curr_prob = curr_mdl.predict(X[idxK == k, :])
        else:
            curr_prob = curr_mdl.predict_proba(X[idxK == k, :])
            curr_prob = curr_prob[:,1]
            
        pred_val_merged[mdl][idxK==k] = curr_prob
        pred_val[mdl][k] = curr_prob

        # calculate score (AUROC)
        curr_score = metrics.roc_auc_score(y[idxK == k], curr_prob)

        # add score to list of scores
        results_val[mdl].append(curr_score)

        # save the current model
        mdl_val[mdl].append(curr_mdl)
        
        print('{} - Finished fold {} of {}. AUROC {:0.3f}.'.format(dt.datetime.now(), k+1, K, curr_score))
    
tar_val = dict()
for k in range(K):
    tar_val[k] = y[idxK==k]

# average AUROC + min/max
for mdl in models:
    curr_score = np.zeros(K)
    for k in range(K):
        curr_score[k] = metrics.roc_auc_score(tar_val[k], pred_val[mdl][k])
    print('{}\t{:0.3f} [{:0.3f}, {:0.3f}]'.format(mdl, np.mean(curr_score), np.min(curr_score), np.max(curr_score)))

# average AUPRC + min/max
for mdl in models:
    curr_score = np.zeros(K)
    for k in range(K):
        curr_score[k] = metrics.average_precision_score(tar_val[k], pred_val[mdl][k])
    print('{}\t{:0.3f} [{:0.3f}, {:0.3f}]'.format(mdl, np.mean(curr_score), np.min(curr_score), np.max(curr_score)))

