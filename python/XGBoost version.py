from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib
from collections import Counter

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.learning_curve import validation_curve
from sklearn.cross_validation import StratifiedKFold
from xgboost.sklearn import XGBClassifier

from sklearn.grid_search import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import normalize

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

df_train = pd.read_csv('train.data', names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv('test.data', names=COLUMNS, skipinitialspace=True, skiprows=1)

# remove NaN elements
df_train = df_train.dropna(how='any', axis=0)
df_test = df_test.dropna(how='any', axis=0)

df_train[LABEL_COLUMN] = (
  df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test[LABEL_COLUMN] = (
  df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

df_train.head()

# Normalizing isn't required for XGBoost but might be for other models
# For XGBoost it doesn't improve accuracy etc.
def normalize_dataframe(dataframe):
    return pd.DataFrame(normalize(dataframe, axis=0), columns=dataframe.columns)

def convert_dataframe(dataframe, normalize=False):
    df_cat_one_hot = pd.get_dummies(dataframe[CATEGORICAL_COLUMNS])
    df_continous_cols = dataframe[CONTINUOUS_COLUMNS]
    if normalize:
        df_continous_cols = normalize_dataframe(df_continous_cols)
    df_one_hot = pd.concat([df_continous_cols, df_cat_one_hot], axis=1)
    print("shape: %s" % (df_one_hot.shape,))
    return df_one_hot

data_train = convert_dataframe(df_train)
data_test = convert_dataframe(df_test)

dtrain = xgb.DMatrix(data_train, label=df_train[LABEL_COLUMN])
dtest = xgb.DMatrix(data_test, label=df_test[LABEL_COLUMN])

params = {
    'objective': 'binary:logistic',
    'n_estimators': 50,
    'max_depth': 8,
    'silent':0,
    'eta': 0.5,
#     'colsample_bytree': 0.6,
#     'subsample': 0.7, 
    'min_child_weight': 100,
#     'gamma': 0.01    
}

num_rounds = 20
watchlist  = [(dtest,'test'), (dtrain,'train')]
bst = xgb.train(params, dtrain, num_rounds, watchlist)

accuracy_score(df_test[LABEL_COLUMN].values, np.round(bst.predict(dtest)))

bst.get_fscore()

print(bst.get_fscore())

pie_labels = [x for x in bst.get_fscore()]
pie_vals = bst.get_fscore().values()

plt.figure(figsize=(17, 12))
plt.legend(patches, pie_labels, loc=2)
patches, text = plt.pie(pie_vals, radius=2)
pie_fscore = plt.axis('equal')

plt.figure(figsize=(17, 12))
ind = np.arange(len(pie_vals))
plt.barh(ind, pie_vals, tick_label=pie_labels)

X_train, y_train = data_train, df_train[LABEL_COLUMN]
X_test, y_test = data_test, df_test[LABEL_COLUMN]

params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'learning_rate': 1.0,
    'silent': 1.0,
    'n_estimators': 5
}

bst2 = XGBClassifier(**params).fit(X_train, y_train)

accuracy_score(y_test, bst2.predict(X_test))

cv = StratifiedKFold(y_train, n_folds=10, shuffle=True, random_state=123)
n_estimators_range = np.linspace(1, 20, 10).astype('int')

train_scores, test_scores = validation_curve(
    XGBClassifier(**params),
    X_train, y_train,
    param_name = 'n_estimators',
    param_range = n_estimators_range,
    cv=cv,
    scoring='accuracy'
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fig = plt.figure(figsize=(10, 6), dpi=100)

plt.title("Validation Curve with XGBoost (eta = 0.3)")
plt.xlabel("number of trees")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.1)

plt.plot(n_estimators_range,
             train_scores_mean,
             label="Training score",
             color="r")

plt.plot(n_estimators_range,
             test_scores_mean, 
             label="Cross-validation score",
             color="g")

plt.fill_between(n_estimators_range, 
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, 
                 alpha=0.2, color="r")

plt.fill_between(n_estimators_range,
                 test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std,
                 alpha=0.2, color="g")

plt.axhline(y=1, color='k', ls='dashed')

plt.legend(loc="best")
plt.show()

i = np.argmax(test_scores_mean)
print("Best cross-validation result ({0:.2f}) obtained for {1} trees".format(test_scores_mean[i], n_estimators_range[i]))

params_grid = {
    'max_depth': [2, 3, 5, 8],
    'n_estimators': [10, 25, 50],
    'learning_rate': [0.5, 0.1, 0.05, 0.001], #np.linspace(1e-16, 1, 3)
    'colsample_bytree': [0.3, 0.6],
    'subsample': [0.3, 0.5, 0.7], 
}

params_fixed = {
    'objective': 'binary:logistic',
    'silent': 1
}

bst_grid = GridSearchCV(
    estimator=XGBClassifier(**params_fixed),
    param_grid=params_grid,
    cv=cv,
    scoring='accuracy'
)

bst_grid.fit(X_train, y_train)

df_bst_grid_params = pd.DataFrame(list(bst_grid.cv_results_['params']))
df_bst_grid_test_score = pd.DataFrame(bst_grid.cv_results_['mean_test_score']*100, columns=['mean_test_score'])
df_bst_grid_train_score = pd.DataFrame(bst_grid.cv_results_['mean_train_score']*100, columns=['mean_train_score'])
df_bst_grid = pd.concat([df_bst_grid_params, df_bst_grid_test_score, df_bst_grid_train_score], axis=1)
df_bst_grid.plot(figsize=(17,15), subplots=True)

print("Best accuracy obtained: {0}".format(bst_grid.best_score_))
print("Parameters:")
for key, value in bst_grid.best_params_.items():
    print("\t{}: {}".format(key, value))

bst_grid.best_params_

confusion_matrix(y_test, bst_grid.best_estimator_.predict(X_test))

bst_grid.best_score_, bst_grid.best_params_

y_train_summary = Counter(y_train)
y_train_summary.items()

weights = np.zeros(len(y_train))
weights[y_train == 0] = 1
weights[y_train == 1] = 1.5

bst_params = params_fixed
bst_params.update(bst_grid.best_params_)
bst_params['silent'] = 0
bst_params

dtrain_weighted = xgb.DMatrix(X_train, label=y_train, weight=weights) # weights added

num_rounds = 20
watchlist  = [(dtest,'test'), (dtrain_weighted,'train')]
bst_weighted = xgb.train(bst_params, dtrain_weighted, num_rounds, watchlist)

y_pred_weighted = np.round(bst_weighted.predict(dtest))
print(accuracy_score(y_test, y_pred_weighted))
print(precision_score(y_test, y_pred_weighted))
print(recall_score(y_test, y_pred_weighted))

confusion_matrix(y_test, np.round(bst_weighted.predict(dtest)))

y_pred_bst = np.round(bst.predict(dtest))
print(accuracy_score(y_test, y_pred_bst))
print(precision_score(y_test, y_pred_bst))
print(recall_score(y_test, y_pred_bst))

confusion_matrix(y_test, y_pred_bst)

df_train2 = pd.read_csv('train.data', names=COLUMNS, skipinitialspace=True)
df_test2 = pd.read_csv('test.data', names=COLUMNS, skipinitialspace=True, skiprows=1)

# remove NaN elements
df_train2 = df_train.dropna(how='any', axis=0)
df_test2 = df_test.dropna(how='any', axis=0)

df_train2[LABEL_COLUMN] = (
  df_train2["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test2[LABEL_COLUMN] = (
  df_test2["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

df_test2.head()

def df_to_indices(df):
    return df.apply(lambda x: pd.factorize(x)[0], axis=0)

def convert_dataframe_to_indices(dataframe, normalize=False):
    df_cat_indices = df_to_indices(dataframe[CATEGORICAL_COLUMNS])
    df_continous_cols = dataframe[CONTINUOUS_COLUMNS]
    if normalize:
        df_continous_cols = normalize_dataframe(df_continous_cols)
    df_indices = pd.concat([df_continous_cols, df_cat_indices], axis=1)
    print("shape: %s" % (df_indices.shape,))
    return df_indices

data_train2 = convert_dataframe_to_indices(df_train2)
data_test2 = convert_dataframe_to_indices(df_test2)

data_train2.head()

X_train2, y_train2 = data_train2, df_train2[LABEL_COLUMN]
X_test2, y_test2 = data_test2, df_test2[LABEL_COLUMN]

params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'learning_rate': 1.0,
    'silent': 1.0,
    'n_estimators': 5
}

bst_indices = XGBClassifier(**bst_params).fit(X_train2, y_train2)

accuracy_score(y_test2, bst_indices.predict(X_test2))

