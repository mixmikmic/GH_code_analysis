get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import os,sys

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

basepath = os.path.expanduser('~/Desktop/src/AllState_Claims_Severity/')
sys.path.append(os.path.join(basepath, 'src'))

np.random.seed(2016)

from data import *

# load files
train      = pd.read_csv(os.path.join(basepath, 'data/raw/train.csv'))
test       = pd.read_csv(os.path.join(basepath, 'data/raw/test.csv'))
sample_sub = pd.read_csv(os.path.join(basepath, 'data/raw/sample_submission.csv'))

cont_features = [col for col in train.columns[1:-1] if 'cont' in col] # this could probably be shifted to common utility
cat_features  = [col for col in train.columns[1:-1] if 'cat' in col]  

# create an indicator for somewhat precarious values for loss. ( only to reduce the number of training examples. )
train['loss_indicator'] = train.loss.map(lambda x: int(x < 4e3))

# create a stratified sample

skf = StratifiedKFold(train.loss_indicator, n_folds=2, shuffle=True, random_state=111)
itrain, itest = next(iter(skf))

train_ = train.iloc[itrain]

# train_cont           = np.round(train_[cont_features], decimals=1)
# test_cont            = np.round(test[cont_features], decimals=1)
train_cont            = train_[cont_features]
test_cont             = test[cont_features]

train_cat, test_cat  = encode_categorical_features(train_[cat_features], test[cat_features])

# X = train_cont[['cont2', 'cont3', 'cont7', 'cont11', 'cont13']]
X = pd.concat((train_cont, train_cat), axis=1)
y = (train_.loss) # take into log space

test_processed = pd.concat((test_cont, test_cat), axis=1)

itrain, itest = train_test_split(range(len(X)), stratify=train_.loss_indicator, test_size=0.2, random_state=123)

X_train = X.iloc[itrain]
X_test  = X.iloc[itest]

y_train = y.iloc[itrain]
y_test  = y.iloc[itest]

def get_cv_scores(train_sub, X_train, y_train, n_estimators=10):
    
    skf = StratifiedKFold(train_sub.loss_indicator, n_folds=3, shuffle=True, random_state=112)
    scores = []

    for itr, ite in skf:
        X_tr = X_train.iloc[itr]
        X_te = X_train.iloc[ite]

        y_tr = y_train.iloc[itr]
        y_te = y_train.iloc[ite]
        
        pipe_1 = Pipeline([
                ('scale', StandardScaler()),
                ('select', SelectKBest(f_regression, k=100)),
                ('model', RandomForestRegressor(n_estimators=n_estimators, max_depth=15, n_jobs=-1, random_state=1222))
            ])
        
#         pipe_2 = Pipeline([
#                 ('scale', StandardScaler()),
#                 ('select', SelectKBest(f_regression, k=80)),
#                 ('model', ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=2324))
#             ])
        
        pipe_1.fit(X_tr, y_tr)
#         pipe_2.fit(X_tr, y_tr)

        y_pred1  = (pipe_1.predict(X_te))
#         y_pred2  = (pipe_2.predict(X_te))

        y_pred = 1.0 * y_pred1

        scores.append(mean_absolute_error(y_te, y_pred))
    
    return scores

scores = get_cv_scores(train_.iloc[itrain], X_train, y_train, 150)

print('Scores for every fold during cross validation ', scores)

# est = Lasso()
est1 = RandomForestRegressor(n_estimators=150, max_depth=15, n_jobs=-1, random_state=12442)
est2 = ExtraTreesRegressor(n_estimators=125, max_depth=15, n_jobs=-1, random_state=23141)

# retrain the model
est1.fit(X_train, y_train)
est2.fit(X_train, y_train)

y_pred1  = (est1.predict(X_test))
y_pred2  = (est2.predict(X_test))
    
y_pred = 0.5 * y_pred1 + 0.5 * y_pred2

print('Mean Absolute Error on unseen examples ', mean_absolute_error(y_test, y_pred))

def perf_by_estimators(train_sub, X_train, y_train, estimators=[10, 25, 50, 75]):
    mean_cv_scores = []
    for estimator in estimators:
        scores = get_cv_scores(train_sub, X_train, y_train, n_estimators=estimator)
        mean_cv_scores.append(np.mean(scores))
    
    plt.scatter(estimators, mean_cv_scores)
    plt.xlabel('Number of Trees')
    plt.ylabel('MAE score')
    return mean_cv_scores

perf_by_estimators(train.iloc[itrain], X_train, y_train)

# train on the full dataset
est1 = RandomForestRegressor(n_estimators=125, max_depth=5, n_jobs=-1, random_state=12442)
est2 = ExtraTreesRegressor(n_estimators=125, max_depth=5, n_jobs=-1, random_state=23141)

est1.fit(X, y)
est2.fit(X, y)

# predict on unseen examples
pred1 = est1.predict(test_processed)
pred2 = est2.predict(test_processed)

predictions = 0.5 * pred1 + 0.5 * pred2

sample_sub['loss'] = predictions

sample_sub.to_csv(os.path.join(basepath, 'submissions/benchmark_lower_depth.csv'), index=False)



