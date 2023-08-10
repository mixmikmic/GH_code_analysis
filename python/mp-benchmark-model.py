# Import libraries
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties # for unicode fonts
import psycopg2
import sys
import datetime as dt
import mp_utils as mp

import sklearn

from sklearn.pipeline import Pipeline

# used for train/test splits and cross validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier

# used to impute mean for data and standardize for computational stability
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

# logistic regression is our favourite model ever
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV # l2 regularized regression
from sklearn.linear_model import LassoCV

# used to calculate AUROC/accuracy
from sklearn import metrics

# used to create confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import cross_val_score

# gradient boosting - must download package https://github.com/dmlc/xgboost
import xgboost as xgb

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

from __future__ import print_function

# below config used on pc70
sqluser = 'alistairewj'
dbname = 'mimic'
schema_name = 'mimiciii'

# Connect to local postgres version of mimic
con = psycopg2.connect(dbname=dbname, user=sqluser)
cur = con.cursor()
cur.execute('SET search_path to ' + schema_name)

# exclusion criteria:
#   - less than 16 years old
#   - stayed in the ICU less than 4 hours
#   - never have any chartevents data (i.e. likely administrative error)
query = """
with t1 as
(
select ie.icustay_id
    , adm.HOSPITAL_EXPIRE_FLAG
    , ROW_NUMBER() over (partition by ie.subject_id order by intime) as rn
from icustays ie
inner join admissions adm
    on ie.hadm_id = adm.hadm_id
inner join patients pat
    on ie.subject_id = pat.subject_id
    and ie.intime > (pat.dob + interval '16' year)
where adm.HAS_CHARTEVENTS_DATA = 1
and 
not (
       (lower(diagnosis) like '%organ donor%' and deathtime is not null)
    or (lower(diagnosis) like '%donor account%' and deathtime is not null)
    )
and (ie.outtime - ie.intime) >= interval '4' hour
)
select 
    icustay_id
    , HOSPITAL_EXPIRE_FLAG
from t1
"""
co = pd.read_sql_query(query,con)
co.set_index('icustay_id',inplace=True)

# extract static vars into a separate dataframe
df_static = pd.read_sql_query('select * from mpap_static_vars',con)
for dtvar in ['intime','outtime','deathtime']:
    df_static[dtvar] = pd.to_datetime(df_static[dtvar])
df_static.set_index('icustay_id',inplace=True)

cur.close()
con.close()

vars_static = [u'male', u'emergency', u'age',
               u'cmed', u'csurg', u'surg', u'nsurg',
               u'surg_other', u'traum', u'nmed',
               u'omed', u'ortho', u'gu', u'gyn', u'ent']

# Connect to local postgres version of mimic
con = psycopg2.connect(dbname=dbname, user=sqluser)
cur = con.cursor()
cur.execute('SET search_path to ' + schema_name)
query = """
select 
    icustay_id
    , oasis
from oasis
"""
oa = pd.read_sql_query(query,con)
oa.set_index('icustay_id',inplace=True)

cur.execute('SET search_path to ' + schema_name)
query = """
select s.icustay_id, s.sofa
from sofa s
order by s.icustay_id
"""

sofa = pd.read_sql_query(query,con)
sofa.set_index('icustay_id',inplace=True)

cur.execute('SET search_path to ' + schema_name)
query = """
select s.icustay_id, s.saps
from saps s
order by s.icustay_id
"""

saps = pd.read_sql_query(query,con)
saps.set_index('icustay_id',inplace=True)

cur.execute('SET search_path to ' + schema_name)
query = """
select s.icustay_id, s.sapsii
from sapsii s
order by s.icustay_id
"""

sapsii = pd.read_sql_query(query,con)
sapsii.set_index('icustay_id',inplace=True)

cur.execute('SET search_path to ' + schema_name)
query = """
select icustay_id
, APSIII
from apsiii
order by icustay_id
"""

apsiii = pd.read_sql_query(query,con)
apsiii.set_index('icustay_id',inplace=True)

cur.close()
con.close()

#analyses = ['base', 'base_nodeathfix', '00', '04', '08','16',
#            '24','fixed', 'wt8', 'wt16', 'wt24',
#            'wt8_00', 'wt8_08', 'wt8_16', 'wt8_24']

seeds = {'base': 473010,
        'base_nodeathfix': 217632,
        '00': 724311,
        '04': 952227,
        '08': 721297,
        '16': 968879,
        '24': 608972,
        'fixed': 585794,
        'wt8': 176381,
        'wt16': 658229,
        'wt24': 635170,
        'wt8_00': 34741,
        'wt8_08': 95467,
        'wt8_16': 85349,
        'wt8_24': 89642,
        'wt24_fixed': 761456}

data_ext = 'base'

# SVM parameters tuned by cross-validation
#svm_parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10]}

# use a full grid over all parameters
# specify parameters and distributions to sample from
N_FEAT = X.shape[1]
param_dist = {"max_depth": [3, 7, None],
              "max_features": sp.stats.randint(1, N_FEAT),
              "min_samples_split": sp.stats.randint(1, N_FEAT),
              "min_samples_leaf": sp.stats.randint(1, N_FEAT),
              "n_estimators": sp.stats.randint(50, 500),
              "criterion": ["gini", "entropy"]}

# set up randomized search for RF
n_iter_search = 20
rf_random_search = RandomizedSearchCV(sklearn.ensemble.RandomForestClassifier(),
                                      param_distributions=param_dist,
                                      n_iter=n_iter_search)


models = {'xgb': xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05),
          'lasso': LassoCV(cv=5,fit_intercept=True),
          'logreg': LogisticRegression(fit_intercept=True),
          'rf': sklearn.ensemble.RandomForestClassifier(),
          #'svm': GridSearchCV(sklearn.svm.SVC(kernel='rbf',class_weight='balanced',probability=False),
          #                   svm_parameters, cv=5, scoring='roc_auc')
         }

results = dict()

np.random.seed(seed=seeds[data_ext])

# load the data into a numpy array
X, y, X_header = mp.load_design_matrix(co,
                                       df_additional_data=df_static[vars_static],
                                       data_ext='_' + data_ext)

print('{} - ========= {} ========='.format(dt.datetime.now(), data_ext))

scores = list()
for i, mdl in enumerate(models):
    if mdl == 'xgb':
        # no pre-processing of data necessary for xgb
        estimator = Pipeline([(mdl, models[mdl])])

    else:
        estimator = Pipeline([("imputer", Imputer(missing_values='NaN',
                                          strategy="mean",
                                          axis=0)),
                      ("scaler", StandardScaler()),
                      (mdl, models[mdl])])


    curr_score = cross_val_score(estimator, X, y, scoring='roc_auc',cv=5)

    print('{} - {:10s} {:0.4f} [{:0.4f}, {:0.4f}]'.format(dt.datetime.now(), mdl,
                                                          np.mean(curr_score),
                                                          np.min(curr_score), np.max(curr_score)))

    # save the score to a dictionary
    results[mdl] = curr_score

#analyses = ['base', 'base_nodeathfix', '00', '04', '08','16',
#            '24','fixed', 'wt8', 'wt16', 'wt24',
#            'wt8_00', 'wt8_08', 'wt8_16', 'wt8_24']

seeds = {'base': 473010,
        'base_nodeathfix': 217632,
        '00': 724311,
        '04': 952227,
        '08': 721297,
        '16': 968879,
        '24': 608972,
        'fixed': 585794,
        'wt8': 176381,
        'wt16': 658229,
        'wt24': 635170,
        'wt8_00': 34741,
        'wt8_08': 95467,
        'wt8_16': 85349,
        'wt8_24': 89642,
        'wt24_fixed': 761456}

data_ext = 'wt24_fixed'

# SVM parameters tuned by cross-validation
#svm_parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10]}

# use a full grid over all parameters
# specify parameters and distributions to sample from
N_FEAT = X.shape[1]
param_dist = {"max_depth": [3, 7, None],
              "max_features": sp.stats.randint(1, N_FEAT),
              "min_samples_split": sp.stats.randint(1, N_FEAT),
              "min_samples_leaf": sp.stats.randint(1, N_FEAT),
              "n_estimators": sp.stats.randint(50, 500),
              "criterion": ["gini", "entropy"]}

# set up randomized search for RF
n_iter_search = 20
rf_random_search = RandomizedSearchCV(sklearn.ensemble.RandomForestClassifier(),
                                      param_distributions=param_dist,
                                      n_iter=n_iter_search)


models = {'xgb': xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05),
          'lasso': LassoCV(cv=5,fit_intercept=True),
          'logreg': LogisticRegression(fit_intercept=True),
          'rf': sklearn.ensemble.RandomForestClassifier(),
          #'svm': GridSearchCV(sklearn.svm.SVC(kernel='rbf',class_weight='balanced',probability=False),
          #                   svm_parameters, cv=5, scoring='roc_auc')
         }

results = dict()

np.random.seed(seed=seeds[data_ext])

# load the data into a numpy array
X, y, X_header = mp.load_design_matrix(co,
                                       df_additional_data=df_static[vars_static],
                                       data_ext='_' + data_ext)

print('{} - ========= {} ========='.format(dt.datetime.now(), data_ext))

scores = list()
for i, mdl in enumerate(models):
    if mdl == 'xgb':
        # no pre-processing of data necessary for xgb
        estimator = Pipeline([(mdl, models[mdl])])

    else:
        estimator = Pipeline([("imputer", Imputer(missing_values='NaN',
                                          strategy="mean",
                                          axis=0)),
                      ("scaler", StandardScaler()),
                      (mdl, models[mdl])])


    curr_score = cross_val_score(estimator, X, y, scoring='roc_auc',cv=5)

    print('{} - {:10s} {:0.4f} [{:0.4f}, {:0.4f}]'.format(dt.datetime.now(), mdl,
                                                          np.mean(curr_score),
                                                          np.min(curr_score), np.max(curr_score)))

    # save the score to a dictionary
    results[mdl] = curr_score

# compare to severity of illness scores
df = co

# merge in the various severity scores
df = df.merge(oa, how='left', left_index=True,right_index=True,suffixes=('','_oasis'))
df = df.merge(sofa, how='left', left_index=True,right_index=True,suffixes=('','_sofa'))
df = df.merge(saps, how='left', left_index=True,right_index=True,suffixes=('','_saps'))
df = df.merge(sapsii, how='left', left_index=True,right_index=True,suffixes=('','_sapsii'))
df = df.merge(apsiii, how='left', left_index=True,right_index=True,suffixes=('','_apsiii'))


for v in df.columns:
    if v != 'hospital_expire_flag':
        print('{:8s} - {:0.4f}'.format(v,metrics.roc_auc_score(df['hospital_expire_flag'],df[v])))

# print the results
mdl = 'xgb'

print('=================== {} ==================='.format(mdl))

for data_ext in np.sort(results_val.keys()):
    curr_score = results[mdl][data_ext]
    print('{:15s} - {:0.4f} [{:0.4f} - {:0.4f}]'.format(data_ext, np.mean(curr_score), np.min(curr_score), np.max(curr_score)))

# extract the data used to train the model
data_ext = 'base'
np.random.seed(seed=seeds[data_ext])

# load the data into a numpy array
X, y, X_header = mp.load_design_matrix(co,
                                       df_additional_data=df_static[vars_static],
                                       data_ext=data_ext)


    
# load into a dictionary the other various datasets/models
X_val = dict()
y_val = dict()
X_header_val = dict()
results_val = dict() # stores AUROCs across datasets
mdl_val = dict() # stores the model trained across k-folds

for i, data_ext in enumerate(analyses):

    # load the data into a numpy array
    X_val[data_ext], y_val[data_ext], X_header_val[data_ext] = mp.load_design_matrix(co,
                                           df_additional_data=df_static[vars_static],
                                           data_ext=data_ext)
    results_val[data_ext] = dict()
    
print('{} - Finished loading data'.format(dt.datetime.now()))

np.random.seed(seed=seeds[data_ext])

# create k-fold indices
K = 5 # number of folds
idxK = np.random.permutation(X.shape[0])
idxK = np.mod(idxK,K)

mdl = 'xgb'
mdl_val[mdl] = list()


for data_ext in X_val:
    results_val[data_ext][mdl] = list() # initialize list for scores

# no pre-processing of data necessary for xgb
estimator = Pipeline([(mdl, models[mdl])])    

for k in range(K):
    # train the model using all but the kth fold
    curr_mdl = estimator.fit(X[idxK != k, :],y[idxK != k])

    for data_ext in X_val:
        # get prediction on this dataset
        curr_prob = curr_mdl.predict_proba(X_val[data_ext][idxK == k, :])
        curr_prob = curr_prob[:,1]

        # calculate score (AUROC)
        curr_score = metrics.roc_auc_score(y_val[data_ext][idxK == k], curr_prob)

        # add score to list of scores
        results_val[data_ext][mdl].append(curr_score)

        # save the current model
        mdl_val[mdl].append(curr_mdl)
    
    print('{} - Finished fold {} of {}.'.format(dt.datetime.now(), k+1, K))

# print the results
mdl = 'xgb'

print('=================== {} ==================='.format(mdl))

for data_ext in np.sort(results_val.keys()):
    curr_score = results_val[data_ext][mdl]
    print('{:15s} - {:0.4f} [{:0.4f} - {:0.4f}]'.format(data_ext, np.mean(curr_score), np.min(curr_score), np.max(curr_score)))

# extract the data
np.random.seed(seed=seeds[data_ext])
data_ext = 'base'

# load the data into a numpy array
X, y, X_header = mp.load_design_matrix(co,
                                       df_additional_data=df_static[vars_static],
                                       data_ext=data_ext,
                                       diedWithin=24)

# load into a dictionary the other various datasets/models
X_val = dict()
y_val = dict()
X_header_val = dict()
results_val = dict() # stores AUROCs across datasets
mdl_val = dict() # stores the model trained across k-folds

for i, data_ext in enumerate(analyses):

    # load the data into a numpy array
    X_val[data_ext], y_val[data_ext], X_header_val[data_ext] = mp.load_design_matrix(co,
                                           df_additional_data=df_static[vars_static],
                                           data_ext='_' + data_ext)
    results_val[data_ext] = dict()
    
print('{} - Finished loading data'.format(dt.datetime.now()))

np.random.seed(seed=seeds[data_ext])

# create k-fold indices
K = 5 # number of folds
idxK = np.random.permutation(X.shape[0])
idxK = np.mod(idxK,K)

mdl = 'xgb'
mdl_val[mdl] = list()


for data_ext in X_val:
    results_val[data_ext][mdl] = list() # initialize list for scores

# no pre-processing of data necessary for xgb
estimator = Pipeline([(mdl, models[mdl])])    

for k in range(K):
    # train the model using all but the kth fold
    curr_mdl = estimator.fit(X[idxK != k, :],y[idxK != k])

    for data_ext in X_val:
        # get prediction on this dataset
        curr_prob = curr_mdl.predict_proba(X_val[data_ext][idxK == k, :])
        curr_prob = curr_prob[:,1]

        # calculate score (AUROC)
        curr_score = metrics.roc_auc_score(y_val[data_ext][idxK == k], curr_prob)

        # add score to list of scores
        results_val[data_ext][mdl].append(curr_score)

        # save the current model
        mdl_val[mdl].append(curr_mdl)
    
    print('{} - Finished fold {} of {}.'.format(dt.datetime.now(), k+1, K))

# print the results
mdl = 'xgb'

print('=================== {} ==================='.format(mdl))

for data_ext in np.sort(results_val.keys()):
    curr_score = results_val[data_ext][mdl]
    print('{:15s} - {:0.4f} [{:0.4f} - {:0.4f}]'.format(data_ext, np.mean(curr_score), np.min(curr_score), np.max(curr_score)))

# create training / test sets
np.random.seed(seed=324875)
icustay_id = co.index.values
idxTest = np.random.rand(X.shape[0]) > 0.20
X_train = X[~idxTest,:]
y_train = y[~idxTest]
iid_train = icustay_id[~idxTest]

X_test = X[idxTest,:]
y_test = y[idxTest]
iid_test = icustay_id[~idxTest]

# optimize hyperparameters of a model using only the training set
# takes ~20 minutes

# first train it w/o grid search
xgb_nopreproc = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
xgb_nopreproc = xgb_nopreproc.fit(X_train, y_train)

# parameters with multiple values will be used in the grid search
grid_params = {
         'max_depth': [4,7], # max depth of the tree
         'learning_rate': [0.05, 0.3], # step size shrinkage, makes earlier trees less important over time
         'n_estimators': [300, 1000], # number of trees built
         'subsample': [0.3, 0.8] # subsample the data when fitting each tree (prevent overfitting)
         }

default_params = {'colsample_bytree': 1,
                  'colsample_bylevel':1,
                  'silent':1,
                  'reg_lambda':1, # L2 regularization on weights
                  'reg_alpha':0, # L1 regularization on weights
                  'objective':'binary:logistic'}

init_model = xgb.XGBClassifier(**default_params)

# the pipeline here is redundant - but could be useful if you want to add any custom preprocessing
# for example, creating binary features from categories, etc...
# the custom function only has to implement 'fit' and 'transform'
estimator = Pipeline([("xgb", GridSearchCV(init_model, grid_params, verbose=1))])

xgb_model_cv = estimator.fit(X_train,y_train)

# generate class probabilities
y_prob = xgb_model_cv.predict_proba(X_test)
y_prob = y_prob[:, 1]

# predict class labels for the test set
y_pred = (y_prob > 0.5).astype(int)

# get the original xgb predictions without cross-validation
# gives us a rough idea of the improvement of selecting some of the parameters
y_prob_nocv = xgb_nopreproc.predict_proba(X_test)[:,1]

print('\n --- Performance on 20% held out test set --- \n')
# generate evaluation metrics
print('Accuracy = {:0.3f}'.format(metrics.accuracy_score(y_test, y_pred)))            
print('AUROC = {:0.3f} (unoptimized model was {:0.3f})'.format(metrics.roc_auc_score(y_test, y_prob),
                                                               metrics.roc_auc_score(y_test, y_prob_nocv)))


mp.print_cm(y_test, y_pred)

#best_params = xgb_model_cv.get_params()['xgb'].best_params_
xgb_model = xgb.XGBClassifier(**default_params)
#xgb_model = xgb_model.set_params(**best_params)
xgb_model = xgb_model.fit(X_train, y_train)

# feature importance!
plt.figure(figsize=[14,40])
ax = plt.gca()
mp.plot_xgb_importance_fmap(xgb_model, X_header=X_header, ax=ax)
plt.show()

