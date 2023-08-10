get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings('ignore')

#load dataset
df=pd.read_csv('dengue_features_train.csv')
labels = pd.read_csv('dengue_labels_train.csv')
test = pd.read_csv('dengue_features_test.csv')
#fill NaNs
df.fillna(method='ffill', inplace=True)
test.fillna(method='ffill', inplace=True)
#to datetime
df['week_start_date']=pd.to_datetime(df['week_start_date'])
test['week_start_date']=pd.to_datetime(test['week_start_date'])
#extract month to new column
df['month']=df.week_start_date.dt.month
test['month']=test.week_start_date.dt.month

df=pd.merge(df, labels, on=['city', 'year', 'weekofyear'])

#idea historical average dengue counts and climate for that week in the year

#for each city, on that week, what is the avg num cases over the years?
df=df.join(df.groupby(['city','weekofyear'])['total_cases'].mean(), on=['city','weekofyear'], rsuffix='_avg')
test=test.join(df.groupby(['city','weekofyear'])['total_cases'].mean(), on=['city','weekofyear'], rsuffix='_avg')

#quick column fix
test.rename(columns={'total_cases': 'total_cases_avg'}, inplace=True)
test.head()

# #plan to lag these columns 3 weeks 
# #this lag did not greatly improve model performance, the rolling avg was better
# cols_to_lag=[
#  'precipitation_amt_mm',
#  'reanalysis_air_temp_k',
#  'reanalysis_avg_temp_k',
#  'reanalysis_dew_point_temp_k',
#  'reanalysis_max_air_temp_k',
#  'reanalysis_min_air_temp_k',
#  'reanalysis_precip_amt_kg_per_m2',
#  'reanalysis_relative_humidity_percent',
#  'reanalysis_sat_precip_amt_mm',
#  'reanalysis_specific_humidity_g_per_kg',
#  'reanalysis_tdtr_k',
#  'station_precip_mm',
#  ]

# for col in cols_to_lag:
#     df['lagged_'+col] = df[col].shift(2)

# for col in cols_to_lag:
#     test['lagged_'+col] = test[col].shift(2)

# lagged_cols=[

#  'lagged_precipitation_amt_mm',
#  'lagged_reanalysis_air_temp_k',
#  'lagged_reanalysis_avg_temp_k',
#  'lagged_reanalysis_dew_point_temp_k',
#  'lagged_reanalysis_max_air_temp_k',
#  'lagged_reanalysis_min_air_temp_k',
#  'lagged_reanalysis_precip_amt_kg_per_m2',
#  'lagged_reanalysis_relative_humidity_percent',
#  'lagged_reanalysis_sat_precip_amt_mm',
#  'lagged_reanalysis_specific_humidity_g_per_kg',
#  'lagged_reanalysis_tdtr_k',
#  'lagged_station_precip_mm',
# ]

# #fill NaNs after the lag
# df=df.fillna(df.groupby("month").transform(lambda x: x.fillna(x.mean())))
# test=test.fillna(test.groupby("month").transform(lambda x: x.fillna(x.mean())))

rolling_cols_sum=[
 'precipitation_amt_mm',
 'reanalysis_sat_precip_amt_mm',
 'station_precip_mm'
]

rolling_cols_avg=[
 'ndvi_ne',
 'ndvi_nw',
 'ndvi_se',
 'ndvi_sw',
 'reanalysis_air_temp_k',
 'reanalysis_avg_temp_k',
 'reanalysis_dew_point_temp_k',
 'reanalysis_max_air_temp_k',
 'reanalysis_min_air_temp_k',
 'reanalysis_precip_amt_kg_per_m2',
 'reanalysis_relative_humidity_percent',
 'reanalysis_specific_humidity_g_per_kg',
 'reanalysis_tdtr_k',
 'station_avg_temp_c',
 'station_diur_temp_rng_c',
 'station_max_temp_c',
 'station_min_temp_c'
]

#loop to make the columns with rolling averages on independent vars
#takes the avg of prior 3 or 4weeks
for col in rolling_cols_sum:
    df['rolling_sum_'+col] = pd.rolling_sum(df[col], 3)
    test['rolling_sum_'+col] = pd.rolling_sum(test[col], 3)
#loop to make the columns with rolling averages on independent vars
#takes the avg of prior 3 weeks
for col in rolling_cols_avg:
    df['rolling_avg_'+col] = pd.rolling_mean(df[col], 3)
    test['rolling_avg_'+col] = pd.rolling_mean(test[col], 3)

# # #engineer column to identify worst months for dengue in each location
# #this didn't help much either so am commenting it out here
# def bad_mo_sj (x):
#     if x == 10: return 1
#     if x == 11: return 1
#     if x == 9: return 1
#     return 0

# def bad_mo_iq (x):
#     if x == 1: return 1
#     if x == 12: return 1
#     if x == 2: return 1
#     return 0

# #create the new column
# sj['key_months'] = sj.month.apply(bad_mo_sj)
# iq['key_months'] = iq.month.apply(bad_mo_iq)
# sj_test['key_months'] = sj_test.month.apply(bad_mo_sj)
# iq_test['key_months'] = iq_test.month.apply(bad_mo_iq)

features=[
'total_cases_avg',                                      
'rolling_avg_reanalysis_specific_humidity_g_per_kg',    
'rolling_avg_station_avg_temp_c',                       
'rolling_avg_reanalysis_dew_point_temp_k',              
'rolling_avg_station_min_temp_c',                       
'rolling_avg_station_max_temp_c',                       
'rolling_avg_reanalysis_min_air_temp_k',                
'rolling_avg_reanalysis_max_air_temp_k',                
'rolling_avg_reanalysis_air_temp_k',                    
'rolling_avg_reanalysis_avg_temp_k',            
'reanalysis_specific_humidity_g_per_kg',               
'reanalysis_dew_point_temp_k',                          
'reanalysis_min_air_temp_k',                           
'station_min_temp_c'     
    ]

#fill resulting NaNs from the lag functions
df.fillna(method='bfill', inplace=True)
test.fillna(method='bfill', inplace=True)

df.to_csv('train_edited.csv')

# separate san juan and iquitos
sj = df[df['city']=='sj']
iq = df[df['city']=='iq']

sj_test=test[test['city']=='sj']
iq_test=test[test['city']=='iq']

from sklearn import model_selection
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

sj_train_subtrain = sj.head(800)
sj_train_subtest = sj.tail(sj.shape[0] - 800)

iq_train_subtrain = iq.head(400)
iq_train_subtest = iq.tail(iq.shape[0] - 400)

#code reference: Machine Learning Mastery - http://machinelearningmastery.com/
#set x and y

X= sj[features]
Y = sj['total_cases']

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', linear_model.LinearRegression()))
models.append(('DT', DecisionTreeRegressor()))
models.append(('RFR', RandomForestRegressor()))
models.append(('SVR', SVR()))
# evaluate each model in turn
results = []
names = []
scoring = 'neg_mean_absolute_error'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='neg_mean_absolute_error')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

## IQUITOS
X= iq[features]
Y = iq['total_cases']

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', linear_model.LinearRegression()))
models.append(('DT', DecisionTreeRegressor()))
models.append(('RFR', RandomForestRegressor()))
models.append(('SVR', SVR()))
# evaluate each model in turn
results = []
names = []
scoring = 'neg_mean_absolute_error'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='neg_mean_absolute_error')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

X= sj[features]
Y = sj['total_cases']

model =DecisionTreeRegressor()
model.fit(X,Y)
model.predict(sj_test[features])

importances=model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

feature_names = X.columns

f, ax = plt.subplots(figsize=(7, 4))
plt.title("Feature ranking", fontsize = 14)
plt.bar(range(X.shape[1]), importances[indices],
    color="b", 
    align="center")
plt.xticks(range(X.shape[1]), feature_names, rotation=90 )
plt.xlim([-1, X.shape[1]])
plt.ylabel("importance", fontsize = 10)
plt.xlabel("index of the feature", fontsize = 10)

X= iq[features]
Y = iq['total_cases']

model =DecisionTreeRegressor()
model.fit(X,Y)
model.predict(iq_test[features])

importances=model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

feature_names = X.columns

f, ax = plt.subplots(figsize=(7, 4))
plt.title("Feature ranking", fontsize = 14)
plt.bar(range(X.shape[1]), importances[indices],
    color="b", 
    align="center")
plt.xticks(range(X.shape[1]), feature_names, rotation=90 )
plt.xlim([-1, X.shape[1]])
plt.ylabel("importance", fontsize = 10)
plt.xlabel("index of the feature", fontsize = 10)

from sklearn.model_selection import GridSearchCV
import time as time

X_sj= sj[features]
Y_sj = sj['total_cases']

X_iq= iq[features]
Y_iq = iq['total_cases']

train_size = 100
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

t0 = time.time()
svr.fit(X_sj,Y_sj)
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)

model_svr_sj=svr.best_estimator_
model_svr_sj

train_size = 100
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

t0 = time.time()
svr.fit(X_iq,Y_iq)
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)

model_svr_iq=svr.best_estimator_
model_svr_iq

model_svr_sj.fit(X_sj,Y_sj)
model_svr_iq.fit(X_iq,Y_iq)

# test_model=SVR(C=10.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
#   gamma=0.10000000000000001, kernel='rbf', max_iter=-1, shrinking=True,
#   tol=0.001, verbose=False)

# test_model_iq=SVR(C=10.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
#   gamma=0.10000000000000001, kernel='rbf', max_iter=-1, shrinking=True,
#   tol=0.001, verbose=False)

# test_model.fit(sj_train_subtrain[features], sj_train_subtrain['total_cases'])

# test_model_iq.fit(iq_train_subtrain[features], iq_train_subtrain['total_cases'])

preds_sj_svr= model_svr_sj.predict(sj_train_subtest[features]).astype(int)
preds_iq_svr=model_svr_iq.predict(iq_train_subtest[features]).astype(int)

sj_train_subtest['fitted'] = preds_sj_svr
iq_train_subtest['fitted'] = preds_iq_svr

sj_train_subtest['fitted'] = preds_sj_svr
iq_train_subtest['fitted'] = preds_iq_svr
### reset axis
sj_train_subtest.index = sj_train_subtest['week_start_date']
iq_train_subtest.index = iq_train_subtest['week_start_date']

sj_train_subtest['fitted'] = preds_sj_svr
iq_train_subtest['fitted'] = preds_iq_svr
### reset axis
sj_train_subtest.index = sj_train_subtest['week_start_date']
iq_train_subtest.index = iq_train_subtest['week_start_date']

figs, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 9))
sj_train_subtest.total_cases.plot(ax=axes[0], label="Actual")
sj_train_subtest.fitted.plot(ax=axes[0], label="Predictions")

iq_train_subtest.total_cases.plot(ax=axes[1], label="Actual")
iq_train_subtest.fitted.plot(ax=axes[1], label="Predictions")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()

preds_sj_svr= model_svr_sj.predict(sj_test[features]).astype(float)
preds_iq_svr=model_svr_iq.predict(iq_test[features]).astype(float)

submission = pd.read_csv("submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([preds_sj_svr, preds_iq_svr])
#add a total cases column to the test df
test['total_cases'] = np.concatenate([preds_sj_svr, preds_iq_svr])
submission.to_csv("svr.csv")

test['total_cases']=test['total_cases']+test['total_cases_avg']

test['total_cases']=test['total_cases']+test['total_cases_avg']
test['random']=np.random.uniform(low=0.8, high=1.5, size=len(test))
#add some randomness
test['total_cases']=test['total_cases_avg']*test['random']
#save file
test.to_csv("test_preds_added.csv")

