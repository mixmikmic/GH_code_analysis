import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import scipy as sp
import os
import datetime
import xgboost as xgb
import warnings
get_ipython().magic('matplotlib inline')

train = pd.read_csv('train_cleaned.csv')
test = pd.read_csv('test_cleaned.csv')

#drop training rows where full_sq is missing
train = train[~train['full_sq'].isnull()]

#clean up dates in training data
train['year'] = train['timestamp'].map(lambda x: int(x.split('-')[0]))
train['month'] = train['timestamp'].map(lambda x: int(x.split('-')[1]))
train['day'] = train['timestamp'].map(lambda x: int(x.split('-')[2]))

train['date'] = train.apply(lambda x: datetime.date(x['year'], x['month'], x['day']), axis=1)

#clean up dates in test data
test['year'] = test['timestamp'].map(lambda x: int(x.split('-')[0]))
test['month'] = test['timestamp'].map(lambda x: int(x.split('-')[1]))
test['day'] = test['timestamp'].map(lambda x: int(x.split('-')[2]))

test['date'] = test.apply(lambda x: datetime.date(x['year'], x['month'], x['day']), axis=1)

#impute 5 missing full_sq in test data
test.loc[test['id'] == 30938, 'full_sq'] = 37.80
test.loc[test['id'] == 35857, 'full_sq'] = 42.07
test.loc[test['id'] == 34670, 'full_sq'] = 122.60

#np.nanmedian(test.full_sq)

test.loc[test['id'] == 36824, 'full_sq'] = 50.42
test.loc[test['id'] == 35108, 'full_sq'] = 50.42

## log transformation
train.loc[:, 'log_price_doc'] = np.log(train['price_doc'] + 1)

#impute test product_type as investment

test.loc[test['product_type'].isnull(),'product_type'] = "Investment"

## label encoding
from sklearn.preprocessing import LabelEncoder

label_encoder = dict()

for feature in ['product_type', 'sub_area']:
    print('encoding feature: {}'.format(feature))
    label_encoder[feature] = LabelEncoder()
    label_encoder[feature].fit(train[feature])
    train.loc[:, feature + '_le'] = label_encoder[feature].transform(train[feature])
    test.loc[:, feature + '_le'] = label_encoder[feature].transform(test[feature])

#make test and train sets
# Convert to numpy values
model_features = ['full_sq', 'floor', 'month', 'cpi', 'usdrub_3m_vol',
                  'material', 'build_year', 'num_room', 'usdrub',
                  'state', 'product_type_le', 'sub_area_le']

X_train = train_macro[model_features].values
Y_train = train_macro['log_price_doc'].values
X_test = test_macro[model_features].values

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)

from sklearn.cross_validation import train_test_split

X_train_sub, X_val, Y_train_sub, Y_val = train_test_split(X_train, Y_train, random_state=1, train_size=0.7)

#size_ = 7000
#X_train_sub, Y_train_sub = X_train[:-size_],  Y_train[:-size_]
#X_val, Y_val = X_train[-size_:],  Y_train[-size_:]

dtrain = xgb.DMatrix(X_train, 
                    Y_train, 
                    feature_names=model_features)
dtrain_sub = xgb.DMatrix(X_train_sub, 
                        Y_train_sub, 
                        feature_names=model_features)
d_val = xgb.DMatrix(X_val, 
                    Y_val, 
                    feature_names=model_features)
dtest = xgb.DMatrix(X_test, 
                    feature_names=model_features)

# hyperparameters
xgb_params = {
    'eta': 0.05,
    'gamma': 0,
    'alpha': 1,
    'max_depth': 6,
    'subsample': .8,
    'colsample_bytree': 0.75,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

sub_model = xgb.train(xgb_params, 
                      dtrain_sub, 
                      num_boost_round=6000,
                      evals=[(d_val, 'val')],
                      early_stopping_rounds=50, 
                      verbose_eval=50)

xgb.plot_importance(sub_model)

full_model = xgb.train(xgb_params,
                       dtrain, 
                       num_boost_round=sub_model.best_iteration,
                       verbose_eval=20)

log_y_pred = full_model.predict(dtest)
y_pred = np.exp(log_y_pred) - 1

submit = pd.DataFrame({'id': np.array(test.index), 'price_doc': y_pred})
submit.to_csv('submission.csv', index=False)

macro_test = pd.read_csv('macro_test.csv')
macro_train = pd.read_csv('macro_train.csv')

macro_test.head()

macro_train.head()

#clean up dates in training data
macro_train['year'] = macro_train['date'].map(lambda x: int(x.split('-')[0]))
macro_train['month'] = macro_train['date'].map(lambda x: int(x.split('-')[1]))
macro_train['day'] = macro_train['date'].map(lambda x: int(x.split('-')[2]))

macro_train['date'] = macro_train.apply(lambda x: datetime.date(x['year'], x['month'], x['day']), axis=1)

#clean up dates in training data
macro_test['year'] = macro_test['date'].map(lambda x: int(x.split('-')[0]))
macro_test['month'] = macro_test['date'].map(lambda x: int(x.split('-')[1]))
macro_test['day'] = macro_test['date'].map(lambda x: int(x.split('-')[2]))

macro_test['date'] = macro_test.apply(lambda x: datetime.date(x['year'], x['month'], x['day']), axis=1)

train.head()

train_macro = pd.merge(left=train, 
         right=macro_train[['cpi', 'usdrub', 'usdrub_3m_vol', 'date']], 
         how="left", on="date")

test_macro = pd.merge(left=test, 
         right=macro_test[['cpi', 'usdrub', 'usdrub_3m_vol', 'date']], 
         how="left", on="date")

test_macro.head()

train_macro[model_features]['cpi'].isnull().sum()

