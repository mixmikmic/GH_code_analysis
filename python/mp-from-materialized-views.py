# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import sys
import datetime as dt

from sklearn.pipeline import Pipeline

# used for train/test splits and cross validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

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

# pretty confusion matrices!
def print_cm(y, yhat):
    print('\nConfusion matrix')
    cm = metrics.confusion_matrix(y, yhat)
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    TP = cm[1,1]
    N = TN+FP+FN+TP
    print('   \t{:6s}\t{:6s}').format('yhat=0','yhat=1')
    print('y=0\t{:6g}\t{:6g}\tNPV={:2.2f}').format(cm[0,0],cm[0,1], 100.0*TN / (TN+FN)) # NPV
    print('y=1\t{:6g}\t{:6g}\tPPV={:2.2f}').format(cm[1,0],cm[1,1], 100.0*TP / (TP+FP)) # PPV
    # add sensitivity/specificity as the bottom line
    print('   \t{:2.2f}\t{:2.2f}\tAcc={:2.2f}').format(100.0*TN/(TN+FP), 100.0*TP/(TP+FN), 100.0*(TP+TN)/N)
    print('   \tSpec\tSens')
    
    
get_ipython().run_line_magic('matplotlib', 'inline')

# below config used on pc70
sqluser = 'alistairewj'
dbname = 'mimic'
schema_name = 'mimiciii'

# Connect to local postgres version of mimic
con = psycopg2.connect(dbname=dbname, user=sqluser)
cur = con.cursor()
cur.execute('SET search_path to ' + schema_name)

# extract data from each materialized view
mp_bloodgasarterial = pd.read_sql_query("select * from mp_bloodgasarterial",con)
mp_gcs = pd.read_sql_query("select * from mp_gcs",con)
mp_height = pd.read_sql_query("select * from mp_height",con)
mp_labs = pd.read_sql_query("select * from mp_labs",con)
mp_rass = pd.read_sql_query("select * from mp_rass",con)
mp_rrt = pd.read_sql_query("select * from mp_rrt",con)
#mp_service = pd.read_sql_query("select * from mp_service",con)
mp_uo = pd.read_sql_query("select * from mp_uo",con)
mp_vasopressor = pd.read_sql_query("select * from mp_vasopressor",con)
mp_vent = pd.read_sql_query("select * from mp_vent",con)
mp_vitals = pd.read_sql_query("select * from mp_vitals",con)
mp_weight = pd.read_sql_query("select * from mp_weight",con)

# define our cohort
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
    and pat.dob < ie.intime - interval '16' year
where adm.HAS_CHARTEVENTS_DATA = 1
and (ie.outtime - ie.intime) >= interval '4' hour
)
select 
    icustay_id
    , HOSPITAL_EXPIRE_FLAG
from t1
"""
co = pd.read_sql_query(query,con)

#TODO: delete subject_id // hadm_id from one of the data tables

# merge in the data into co from the various other tables
# note we start off with co then merge into the created dataframe
df = co.merge(mp_bloodgasarterial, how='inner', on='icustay_id',suffixes=('','_bloodgasarterial'))
df = df.merge(mp_gcs, how='inner', on='icustay_id',suffixes=('','_gcs'))
df = df.merge(mp_height, how='inner', on='icustay_id',suffixes=('','_height'))
df = df.merge(mp_labs, how='inner', on='icustay_id',suffixes=('','_labs'))
df = df.merge(mp_rass, how='inner', on='icustay_id',suffixes=('','_rass'))
df = df.merge(mp_rrt, how='inner', on='icustay_id',suffixes=('','_rrt'))
df = df.merge(mp_uo, how='inner', on='icustay_id',suffixes=('','_uo'))
df = df.merge(mp_vasopressor, how='inner', on='icustay_id',suffixes=('','_vasopressor'))
df = df.merge(mp_vent, how='inner', on='icustay_id',suffixes=('','_vent'))
df = df.merge(mp_vitals, how='inner', on='icustay_id',suffixes=('','_vitals'))
df = df.merge(mp_weight, how='inner', on='icustay_id',suffixes=('','_weight'))

# also get some static vars to compare model w/ and w/o them
mp_static_vars = pd.read_sql_query("select * from mp_static_vars",con)

cur.close()
con.close()

# write the data out to file
df.set_index('icustay_id').to_csv('design_matrix_from_view.csv')

np.random.seed(seed=72397)
# move from a data frame into a numpy array
X = df.values.astype(float)
y = X[:,1].astype(float)

icustay_id = X[:,0]

# delete first 2 columns: the ID and the outcome
X = np.delete(X,0,axis=1)
X = np.delete(X,0,axis=1)

# get a header row
X_header = [xval for x, xval in enumerate(df.columns) if x > 1]

X_orig = X
X_header_orig = X_header

models = {'l2logreg': LogisticRegressionCV(penalty='l2',cv=5,fit_intercept=True),
         'lasso': LassoCV(cv=5,fit_intercept=True),
         'xgb': xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05),
         'logreg': LogisticRegression(fit_intercept=True)}

scores = list()
for i, mdl in enumerate(['logreg','l2logreg','lasso','xgb']):
    if mdl == 'xgb':
        # no pre-processing of data necessary
        estimator = Pipeline([(mdl, models[mdl])])
        
    else:
        estimator = Pipeline([("imputer", Imputer(missing_values='NaN',
                                          strategy="mean",
                                          axis=0)),
                      ("scaler", StandardScaler()),
                      (mdl, models[mdl])])
    
    scores.append(cross_val_score(estimator, X, y, scoring='roc_auc',cv=5))
    
    print('{:10s} - {:0.4f} [{:0.4f}, {:0.4f}]').format(mdl, np.mean(scores[i]), np.min(scores[i]), np.max(scores[i]) )

df = df.merge(mp_static_vars, how='inner', on='icustay_id',suffixes=('','_static'))

# move from a data frame into a numpy array
X = df.values.astype(float)
y = X[:,1].astype(float)

# delete first 2 columns: the ID and the outcome
X = np.delete(X,0,axis=1)
X = np.delete(X,0,axis=1)

# get a header row
X_header = [xval for x, xval in enumerate(df.columns) if x > 1]

models = {'l2logreg': LogisticRegressionCV(penalty='l2',cv=5,fit_intercept=True),
         'lasso': LassoCV(cv=5,fit_intercept=True),
         'xgb': xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05),
         'logreg': LogisticRegression(fit_intercept=True)}

scores = list()
for i, mdl in enumerate(['logreg','l2logreg','lasso','xgb']):
    if mdl == 'xgb':
        # no pre-processing of data necessary
        estimator = Pipeline([(mdl, models[mdl])])
        
    else:
        estimator = Pipeline([("imputer", Imputer(missing_values='NaN',
                                          strategy="mean",
                                          axis=0)),
                      ("scaler", StandardScaler()),
                      (mdl, models[mdl])])
    
    scores.append(cross_val_score(estimator, X, y, scoring='roc_auc',cv=5))
    
    print('{:10s} - {:0.4f} [{:0.4f}, {:0.4f}]').format(mdl, np.mean(scores[i]), np.min(scores[i]), np.max(scores[i]) )

