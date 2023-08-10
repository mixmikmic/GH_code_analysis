get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import os,sys

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.externals import joblib

from scipy.stats.mstats import gmean

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

basepath = os.path.expanduser('~/Desktop/src/AllState_Claims_Severity/')
sys.path.append(os.path.join(basepath, 'src'))

np.random.seed(2016)

from data import *
from utils import *

# load the dataset
train, test, sample_sub = load_data()

# concat train and test dataframes
data = pd.concat((train, test))

categorical_features = get_categorical_features(data.columns)
numerical_features   = get_numerical_features(data.columns)

def mean_by_target(data, categorical_features):
    for col in categorical_features:
        data[col+'_mean_by_target'] = data.groupby([col])['loss'].transform(lambda x: x.mean())        
    return data

data = mean_by_target(data, categorical_features)

# label encoding
data = label_encoding(data, categorical_features)

# save the processed data to disk
joblib.dump(len(train), os.path.join(basepath, 'data/processed/n_train'))
joblib.dump(data, os.path.join(basepath, 'data/processed/processed_data.pkl'))

# load data from disk
data    = joblib.load(os.path.join(basepath, 'data/processed/processed_data.pkl'))
n_train = joblib.load(os.path.join(basepath, 'data/processed/n_train')) 

features = data.columns[116:].drop(['id', 'loss'])

train_   = data[:n_train][features]
test_    = data[n_train:][features]

y        = np.log(data[:n_train].loss) # take it into log domain

X_train, X_test, y_train, y_test = train_test_split(train_, y, test_size=0.33, random_state=1239137)

print(X_train.shape)
print(X_test.shape)

scores = cv_xgboost(X_train, np.exp(y_train))

scores

np.mean(scores)

def mae(y, y0):
    
    y0=y0.get_label()    
    return 'error',mean_absolute_error(np.exp(y), np.exp(y0))

params = {}

params['max_depth']        = 8
params['objective']        = 'reg:linear'
params['eta']              = 0.03
params['nthread']          = 4
params['gamma']            = 4
params['min_child_weight'] = 7
params['subsample']        = 0.8
params['colsample_bytree'] = 0.4

n_rounds = 600

plst   = list(params.items())

Dtrain = xgb.DMatrix(X_train, y_train)
Dval   = xgb.DMatrix(X_test, y_test)
    
# define a watch list to observe the change in error for training and holdout data
watchlist  = [ (Dtrain, 'train'), (Dval, 'eval')]
 
model = xgb.train(plst, 
                  Dtrain, 
                  n_rounds,
                  feval=mae,  # custom evaluation function
                 )

yhat = np.exp(model.predict(Dval))
print('MAE on unseen set ', mean_absolute_error(np.exp(y_test), yhat))

DTRAIN = xgb.DMatrix(train_, y)
DTEST  = xgb.DMatrix(test_.fillna(-99999))

# train on full dataset

model = xgb.train(plst, 
                  DTRAIN, 
                  n_rounds,
                  feval=mae  # custom evaluation function
                 )

predictions = model.predict(DTEST)
predictions = np.exp(predictions)

sample_sub['loss'] = predictions
sample_sub.to_csv(os.path.join(basepath, 'submissions/xgboost_mean_by_target.csv'), index=False)



