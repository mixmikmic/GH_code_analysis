from __future__ import print_function

# Import libraries
import numpy as np
import pandas as pd
import matplotlib
import sklearn
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties # for unicode fonts
import psycopg2
import sys
import datetime as dt
import mp_utils as mp

from collections import OrderedDict

# used to print out pretty pandas dataframes
from IPython.display import display, HTML

from sklearn.pipeline import Pipeline

# used to impute mean for data and standardize for computational stability
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

# logistic regression is our favourite model ever
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV # l2 regularized regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

# used to calculate AUROC/accuracy
from sklearn import metrics

# used to create confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

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

# below config used on pc70
sqluser = 'alistairewj'
dbname = 'mimic'
schema_name = 'mimiciii'
query_schema = 'SET search_path to public,' + schema_name + ';'

# Connect to local postgres version of mimic
con = psycopg2.connect(dbname=dbname, user=sqluser)

# exclusion criteria:
#   - less than 16 years old
#   - stayed in the ICU less than 4 hours
#   - never have any chartevents data (i.e. likely administrative error)
query = query_schema + """
select 
    subject_id, hadm_id, icustay_id
from mp_cohort
where excluded = 0
"""
co = pd.read_sql_query(query,con)

# extract static vars into a separate dataframe
df_static = pd.read_sql_query(query_schema + 'select * from mp_static_data', con)
#for dtvar in ['intime','outtime','deathtime']:
#    df_static[dtvar] = pd.to_datetime(df_static[dtvar])

vars_static = [u'is_male', u'emergency_admission', u'age',
               # services
               u'service_any_noncard_surg',
               u'service_any_card_surg',
               u'service_cmed',
               u'service_traum',
               u'service_nmed',
               # ethnicities
               u'race_black',u'race_hispanic',u'race_asian',u'race_other',
               # phatness
               u'height', u'weight', u'bmi']

# get ~5 million rows containing data from errbody
# this takes a little bit of time to load into memory (~2 minutes)

# %%time results
# CPU times: user 42.8 s, sys: 1min 3s, total: 1min 46s
# Wall time: 2min 7s

df = pd.read_sql_query(query_schema + 'select * from mp_data', con)
df.drop('subject_id',axis=1,inplace=True)
df.drop('hadm_id',axis=1,inplace=True)
df.sort_values(['icustay_id','hr'],axis=0,ascending=True,inplace=True)
print(df.shape)

# get death information
df_death = pd.read_sql_query(query_schema + """
select 
co.subject_id, co.hadm_id, co.icustay_id
, ceil(extract(epoch from (co.outtime - co.intime))/60.0/60.0) as dischtime_hours
, ceil(extract(epoch from (adm.deathtime - co.intime))/60.0/60.0) as deathtime_hours
, case when adm.deathtime is null then 0 else 1 end as death
from mp_cohort co
inner join admissions adm
on co.hadm_id = adm.hadm_id
where co.excluded = 0
""", con)

# get censoring information
df_censor = pd.read_sql_query(query_schema + """
select co.icustay_id, min(cs.charttime) as censortime
, ceil(extract(epoch from min(cs.charttime-co.intime) )/60.0/60.0) as censortime_hours
from mp_cohort co 
inner join mp_code_status cs
on co.icustay_id = cs.icustay_id
where cmo+dnr+dni+dncpr+cmo_notes>0
and co.excluded = 0
group by co.icustay_id
""", con)

time_dict = mp.generate_times_before_death(df_death, T=4, seed=111)
df_data = mp.get_design_matrix(df, time_dict, W=8, W_extra=24)

# load the data into a numpy array

# first, the data from static vars from df_static
X = df_data.merge(df_static.set_index('icustay_id')[vars_static], how='left', left_index=True, right_index=True)
# next, add in the outcome: death in hospital
X = X.merge(df_death.set_index('icustay_id')[['death']], left_index=True, right_index=True)

# generate K-fold indices
K = 5 # number of folds
X = X.merge(df_death.set_index('icustay_id')[['subject_id']], left_index=True, right_index=True)

# get unique subject_id
sid = np.sort(np.unique(X['subject_id'].values))

# assign k-fold
idxK_sid = np.random.permutation(sid.shape[0])
idxK_sid = np.mod(idxK_sid,K)

# get indices which map subject_ids in sid to the X dataframe
idxMap = np.searchsorted(sid, X['subject_id'].values)

# use these indices to map the k-fold integers
idxK = idxK_sid[idxMap]

# drop the subject_id column
X.drop('subject_id',axis=1,inplace=True)
# convert to numpy data (assumes target, death, is the last column)
X = X.values
y = X[:,-1]
X = X[:,0:-1]
X_header = vars_static + [x for x in df_data.columns.values]

# Rough timing info:
#     rf - 3 seconds per fold
#    xgb - 30 seconds per fold
# logreg - 4 seconds per fold
#  lasso - 8 seconds per fold
models = {'xgb': xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05),
          'lasso': LassoCV(cv=5,fit_intercept=True,normalize=True,max_iter=10000),
          'logreg': LogisticRegression(fit_intercept=True),
          'rf': RandomForestClassifier()
         }

mdl_val = dict()
results_val = dict()

for mdl in models:
    print('=============== {} ==============='.format(mdl))
    mdl_val[mdl] = list()
    results_val[mdl] = list() # initialize list for scores

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
        curr_mdl = estimator.fit(X[idxK != k, :],y[idxK != k])

        # get prediction on this dataset
        if mdl == 'lasso':
            curr_prob = curr_mdl.predict(X[idxK == k, :])
        else:
            curr_prob = curr_mdl.predict_proba(X[idxK == k, :])
            curr_prob = curr_prob[:,1]

        # calculate score (AUROC)
        curr_score = metrics.roc_auc_score(y[idxK == k], curr_prob)

        # add score to list of scores
        results_val[mdl].append(curr_score)

        # save the current model
        mdl_val[mdl].append(curr_mdl)

        print('{} - Finished fold {} of {}. AUROC {:0.3f}.'.format(dt.datetime.now(), k+1, K, curr_score))

mp.plot_model_results(results_val)

# creates datasets in X_all for evaluation

# experiment elements contain a list: [seed, W (window size), T_to_death]
experiments = OrderedDict([['base', [473010,8,None]],
               ['24hr', [585794,24,None]],
               ['Td=00', [724311,8,0]],
               ['Td=04', [952227,8,4]],
               ['Td=08', [721297,8,8]],
               ['Td=16', [968879,8,16]],
               ['Td=24', [608972,8,24]],
               ['24hr Td=00', [34741,24,0]],
               ['24hr Td=04', [34319,24,4]],
               ['24hr Td=08', [95467,24,8]],
               ['24hr Td=16', [85349,24,16]],
               ['24hr Td=24', [89642,24,24]]
                          ])

# fuzzyness to allow deathtime to be a little bit after discharge time
death_epsilon=2
X_all = dict()
y_all = dict()
iid_all = dict()
pred_all = dict()
time_all = dict()
X_header_all = dict()

for e in experiments:
    params = experiments[e]
    time_all[e] = mp.generate_times_before_death(df_death, seed=params[0], T=None, T_to_death=params[2])
    df_data = mp.get_design_matrix(df, time_all[e], W=params[1], W_extra=24)
    
    # load the data into a numpy array
        
    # Add in static vars from df_static
    X = df_data.merge(df_static.set_index('icustay_id')[vars_static],
                      how='left', left_index=True, right_index=True)
    
    
    if params[2] is not None:
        df_tmp = df_death[['icustay_id','death','dischtime_hours', 'deathtime_hours']].copy()
        df_tmp['death_in_icu'] = (df_tmp['deathtime_hours']<=(df_tmp['dischtime_hours']+params[2]+death_epsilon)).astype(float)
        X = X.merge(df_tmp[['icustay_id','death_in_icu']].set_index('icustay_id'),
                          left_index=True, right_index=True)
    else:
        X = X.merge(df_death[['icustay_id','death']].set_index('icustay_id'),
                          left_index=True, right_index=True)

    iid_all[e] = X.index.values
    X = X.values
    y_all[e] = X[:,-1]
    X_all[e] = X[:,0:-1]
    
    X_header_all[e] = df_data.columns

# evaluate the models on various datasets
K = 5 # number of folds
results_all = dict()
mdl_base = dict()
base_exp = 'base'

# train the base model
e = base_exp

# get unique subject_id
sid = np.sort(np.unique(df_death['subject_id'].values))

# assign k-fold
idxK_sid = np.random.permutation(sid.shape[0])
idxK_sid = np.mod(idxK_sid,K)

# get indices which map subject_ids in sid to the X dataframe
idxMap = np.searchsorted(sid, df_death['subject_id'].values)

# use these indices to map the k-fold integers
idxK_all = idxK_sid[idxMap]

# get the data for the dataset which the model is developed on
X = X_all[e]
y = y_all[e]
iid_curr = iid_all[e]

# map the k-fold indices from all IID to the subset included in this data
iid = df_death['icustay_id'].values
idxMap = np.nonzero(np.in1d(iid,iid_curr))[0]
idxK = idxK_all[idxMap]

results_all[e] = dict()

idxMap = np.nonzero(np.in1d(iid,iid_curr))[0]

for mdl in models:
    # train the model for the fixed dataset
    print('=============== {} ==============='.format(mdl))
    
    if mdl == 'xgb':
        # no pre-processing of data necessary for xgb
        estimator = Pipeline([(mdl, models[mdl])])

    else:
        estimator = Pipeline([("imputer", Imputer(missing_values='NaN',
                                          strategy="mean",
                                          axis=0)),
                      ("scaler", StandardScaler()),
                      (mdl, models[mdl])])
    print('Training 5-fold model for application to various datasets...'.format(K))
    
    results_all[e][mdl] = list()
    mdl_base[mdl] = list()
    
    for k in range(K):
        # train the model using all but the kth fold
        curr_mdl = estimator.fit(X[idxK != k, :], y[idxK != k])

        # get prediction on this dataset
        if mdl == 'lasso':
            curr_prob = curr_mdl.predict(X[idxK == k, :])
        else:
            curr_prob = curr_mdl.predict_proba(X[idxK == k, :])
            curr_prob = curr_prob[:,1]

        # calculate score (AUROC)
        curr_score = metrics.roc_auc_score(y[idxK == k], curr_prob)

        # add score to list of scores
        results_all[e][mdl].append(curr_score)

        # save the current model
        mdl_base[mdl].append(curr_mdl)

        print('{} - Finished fold {} of {}. AUROC {:0.3f}.'.format(dt.datetime.now(), k+1, K, curr_score))

    # apply the trained model to each dataset in experiments
    for e in experiments:
        if e == base_exp:
            continue
            
        if e not in results_all:
            results_all[e] = dict()
        results_all[e][mdl] = list()

        X = X_all[e]
        y = y_all[e]
        iid_curr = iid_all[e]
        
        # map the k-fold indices from all IID to the subset included in this data
        idxMap = np.nonzero(np.in1d(iid,iid_curr))[0]
        idxK = idxK_all[idxMap]
        
        
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
            curr_mdl = mdl_base[mdl][k]

            # get prediction on this dataset
            if mdl == 'lasso':
                curr_prob = curr_mdl.predict(X[idxK == k, :])
            else:
                curr_prob = curr_mdl.predict_proba(X[idxK == k, :])
                curr_prob = curr_prob[:,1]

            # calculate score (AUROC)
            curr_score = metrics.roc_auc_score(y[idxK == k], curr_prob)

            # add score to list of scores
            results_all[e][mdl].append(curr_score)

        print('{} - {:10s} - AUROC {:0.3f} [{:0.3f}, {:0.3f}]'.format(dt.datetime.now(), e,
                                                                    np.mean(results_all[e][mdl]),
                                                                    np.min(results_all[e][mdl]),
                                                                    np.max(results_all[e][mdl])))

# plot a figure of the results
marker = ['o','s','x','d']
xi_str = ['Td=00','Td=04','Td=08','Td=16','Td=24']
xi = [int(x[-2:]) for x in xi_str]

plt.figure(figsize=[14,10])
for m, mdl in enumerate(models):
    all_score = list()
    for i, x in enumerate(xi_str):
        curr_score = results_all[x][mdl]

        plt.plot(xi[i] * np.ones(len(curr_score)), curr_score,
                marker=marker[m], color=col[m],
                markersize=10, linewidth=2, linestyle=':')

        all_score.append(np.median(curr_score))
        
    # plot a line through the mean across all evaluations

    plt.plot(xi, all_score,
            marker=marker[m], color=col[m],
            markersize=10, linewidth=2, linestyle='-',
            label=mdl)

plt.gca().set_xticks(np.linspace(0,24,7))
plt.gca().set_xlim([-1,25])
plt.gca().invert_xaxis()
plt.legend(loc='lower center',fontsize=16)
plt.xlabel('Lead time (hours)',fontsize=18)
plt.ylabel('AUROC',fontsize=18)

ax = plt.gca()

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(16) 
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(16) 

plt.grid()
#plt.savefig('auroc_over_time_dw24.pdf')
plt.show()

