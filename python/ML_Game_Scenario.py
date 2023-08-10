import time
import random
import itertools
import math

from csv import DictReader

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz

seed = 2018
np.random.seed(seed)
random.seed(seed)

data_test = pd.read_csv('data/train.csv', nrows=20000)
data_test.head()

target = data_test.click
data_test.drop(['id', 'click'], axis=1, inplace=True)
for arg in data_test.columns:
    data_test[arg] = data_test[arg].astype(str)

X = pd.get_dummies(data_test)

X.head()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA

pca=PCA(n_components=40)
tr=pca.fit(X)
plt.plot(tr.explained_variance_[0:40])

pca=PCA(n_components=20)
XX=pca.fit(X).transform(X)

XX=pd.DataFrame(XX)

target=target.astype('float')
X_train, X_test, y_train, y_test = train_test_split(XX, 
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=seed,
                                                       shuffle=True)

algs = [SVC, DecisionTreeClassifier, RandomForestClassifier, GaussianNB, LogisticRegression, KNeighborsClassifier, MLPClassifier] # type the name of the algorithm class from the list above

model = []
acc_train_initial, acc_test_initial = [], []
training_time, prediction_time = [], []

X_all = X_train.append(X_test, ignore_index=True).reset_index().drop('index', axis=1)
y_all = y_train.append(y_test, ignore_index=True).reset_index().loc[:, 'click']
#cv_fold = np.hstack([np.zeros((y_all.shape[0] / 2)), np.ones((y_all.shape[0] / 2))])
#ps = PredefinedSplit(cv_fold)

ts_start = time.time()

for alg in algs:
    model.append(alg.__name__)

    if 'random_state' in alg().get_params().keys():
        clf = alg(random_state=seed)
    else:
        clf = alg()

    ts = time.time()
    clf.fit(X_train, y_train)
    training_time.append(round(time.time() - ts, 4))

    ts = time.time()
    preds_train = clf.predict(X_train)
    preds_test = clf.predict(X_test)
    prediction_time.append(round(time.time() - ts, 5))
    
    acc_train_initial.append(round(roc_auc_score(y_train, preds_train) * 100, 4))
    acc_test_initial.append(round(roc_auc_score(y_test, preds_test) * 100, 4))

    print('Tested {} in {:.2f} seconds'.format(alg.__name__, time.time() - ts_start))
    
models = pd.DataFrame({
    'MODEL': model,
    'TRAIN_ROC_AUC_INITIAL': acc_train_initial,
    'TEST_ROC_AUC_INITIAL': acc_test_initial,
    'TRAINING_TIME': training_time,
    'PREDICTION_TIME': prediction_time})

models.sort_values(by='TEST_ROC_AUC_INITIAL', ascending=False).reset_index().loc[:, ['MODEL', 'TRAIN_ROC_AUC_INITIAL', 'TEST_ROC_AUC_INITIAL', 'TRAINING_TIME', 'PREDICTION_TIME']]

search_space = {
  'LinearSVC': [{'random_state': [seed]}, {
    'random_state': [seed],
    'C':  np.logspace(0, 4, 5),
  }],
  'SVC': [{'random_state': [seed]}, {
    'random_state': [seed],
    'probability': [True],
    "kernel": ['rbf'],
    'C': np.logspace(0, 4, 5),
    'gamma': np.logspace(-5, -1, 5)
  }],
  'DecisionTreeClassifier': [{'random_state': [seed]}, {
    'random_state': [seed],
    'max_features': ['sqrt', 'log2', None],
    "max_depth": [10, 50, 100, 200, None]
  }],
  'RandomForestClassifier': [{'random_state': [seed]}, {
    'random_state': [seed],
    'n_estimators': [100, 300, 500, 700],
    'max_features': ['sqrt', 'log2', None],
    "max_depth": [10, 50, 100, 200, None]
  }],
  'GaussianNB': {},
  'LogisticRegression': [{'random_state': [seed]}, {
    'random_state': [seed],
    'C': np.logspace(0, 4, 5)
  }],
  'KNeighborsClassifier': [{}, {
    'n_neighbors': [1, 5, 10, 20, 50],
    'weights': ['uniform', 'distance'],
  }],
  'MLPClassifier': [{'random_state': [seed]}, {
    'random_state': [seed],
    'activation': ['logistic', 'relu'],
    'hidden_layer_sizes': [(50,), (100,), (300,), (50, 50), (100, 100), (300, 300)],
    'max_iter': [500],
    'early_stopping': [True]
  }]
}

from sklearn import metrics

model = []
clfs=[]
acc_train_tuned, acc_test_tuned, = [], []
tuning_time, training_time, prediction_time = [], [], []
roc_auc=[]

X_all = X_train.append(X_test, ignore_index=True).reset_index().drop('index', axis=1)
y_all = y_train.append(y_test, ignore_index=True).reset_index().loc[:, 'click']
#cv_fold = np.hstack([np.zeros((y_all.shape[0] / 2)), np.ones((y_all.shape[0] / 2))])
#ps = PredefinedSplit(cv_fold)

ts_start = time.time()
for alg in algs:
    print('Train model:', alg.__name__)
    model.append(alg.__name__)
    ts = time.time()
    clf = GridSearchCV(alg(), search_space[alg.__name__], scoring='roc_auc', cv=3, 
                   return_train_score=True, verbose=1)
    clf.fit(X_all, y_all)
    tuning_time.append(round(time.time() - ts, 4))
    clfs.append(clf)

    ts = time.time()
    clf = alg(**clf.best_params_)
    clf.fit(X_train, y_train)
    training_time.append(round(time.time() - ts, 4))
    
    fpr, tpr, thresholds =metrics.roc_curve(y_test,clf.predict_proba(X_test)[:,1])
    roc_auc.append(metrics.auc(fpr,tpr))

    ts = time.time()
    preds_train = clf.predict(X_train)
    preds_test = clf.predict(X_test)
    prediction_time.append(round(time.time() - ts, 5))

    acc_train_tuned.append(round(accuracy_score(y_train, preds_train) * 100, 4))
    acc_test_tuned.append(round(accuracy_score(y_test, preds_test) * 100, 4))

    print('Tested {} in {:.2f} seconds'.format(alg.__name__, time.time() - ts_start))
    
models = pd.DataFrame({
    'MODEL': model,
    'TRAIN_ACC_TUNED': acc_train_tuned,
    'TEST_ACC_TUNED': acc_test_tuned,
    'TUNING_TIME': tuning_time,
    'TRAINING_TIME': training_time,
    'PREDICTION_TIME': prediction_time,
    'ROC-AUC':roc_auc})











clf=GridSearchCV(alg(), search_space[alg.__name__], scoring='accuracy', cv=ps, 
                   return_train_score=True, verbose=0, n_jobs=-1)
clf.fit(X_train, y_train)

clf

models.sort_values(by='TEST_ACC_TUNED', ascending=False).reset_index().loc[:, ['MODEL', 'TRAIN_ACC_TUNED', 'TEST_ACC_TUNED', 'TUNING_TIME', 'TRAINING_TIME', 'PREDICTION_TIME']]

clf = GridSearchCV(alg(), search_space[alg.__name__], scoring='accuracy', cv=ps, 
                   return_train_score=True, verbose=0, n_jobs=-1)

search_space[alg.__name__]

from sklearn import metrics

clf=LogisticRegression()
clf.fit(X_train, y_train)
preds_test = clf.predict(X_test)


fpr, tpr, thresholds =metrics.roc_curve(y_test,clf.predict_proba(X_test)[:,1])
print(clf.score(X_test, y_test))
print(metrics.auc(fpr,tpr))

SVC()

