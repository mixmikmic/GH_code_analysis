#standard imports
get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import time as time

from warnings import filterwarnings
filterwarnings('ignore')

#load dataset
df=pd.read_csv('train_edited.csv') #training set with preprocessing already implemented
#labels = pd.read_csv('dengue_labels_train.csv')
test = pd.read_csv('test_preds_added.csv')# this is the testing data with one iteration of predictions
#fill NaNs - ffill bc it is a timeseries
df.fillna(method='ffill', inplace=True)
test.fillna(method='ffill', inplace=True)
#to datetime
df['week_start_date']=pd.to_datetime(df['week_start_date'])
test['week_start_date']=pd.to_datetime(test['week_start_date'])
#extract month to new column
df['month']=df.week_start_date.dt.month
test['month']=test.week_start_date.dt.month
#merge labels
#df=pd.merge(df, labels, on=['city', 'year', 'weekofyear'])

df.rename(columns={'total_cases_x': 'total_cases'}, inplace=True)

# separate san juan and iquitos
sj = df[df['city']=='sj']
iq = df[df['city']=='iq']

sj_test=test[test['city']=='sj']
iq_test=test[test['city']=='iq']

#value previous week
#train
sj['cases_prev_wk'] = sj['total_cases'].shift(1)
iq['cases_prev_wk'] = iq['total_cases'].shift(1)

#test
sj_test['cases_prev_wk'] = sj_test['total_cases'].shift(1)
iq_test['cases_prev_wk'] = iq_test['total_cases'].shift(1)

#need to make sure no NaNs added when creating moving avg or getting previous week values
sj.fillna(method='bfill', inplace=True)
iq.fillna(method='bfill', inplace=True)

sj_test.fillna(method='bfill', inplace=True)
sj_test.fillna(method='bfill', inplace=True)

#these features performed better
features2=[
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
'station_min_temp_c',
'cases_prev_wk'
    ]      

#TRAIN
X_sj= sj[features2]
Y_sj = sj['total_cases']

X_iq= iq[features2]
Y_iq = iq['total_cases']

#TEST
X_sj_t= sj_test[features2]

X_iq_t= iq_test[features2]


#need to make sure no NaNs added when creating moving avg or getting previous week values
X_sj.fillna(method='bfill', inplace=True)
X_iq.fillna(method='bfill', inplace=True)

X_sj_t.fillna(method='bfill', inplace=True)
X_iq_t.fillna(method='bfill', inplace=True)

##SAN JUAN
train_size = 100
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

t0 = time.time()
svr.fit(X_sj,Y_sj)
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)

model_sj=svr.best_estimator_
model_sj

##IQUITOS
train_size = 100
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

t0 = time.time()
svr.fit(X_iq,Y_iq)
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)

model_iq=svr.best_estimator_
model_iq

#further divide the training set to create a lineplot
sj_train_subtrain = sj.head(800)
sj_train_subtest = sj.tail(sj.shape[0] - 800)

iq_train_subtrain = iq.head(400)
iq_train_subtest = iq.tail(iq.shape[0] - 400)

#create preds
preds_sj= model_sj.predict(sj_train_subtest[features2]).astype(int)
preds_iq=model_iq.predict(iq_train_subtest[features2]).astype(int)
#add to the dataframes
sj_train_subtest['fitted'] = preds_sj
iq_train_subtest['fitted'] = preds_iq
### reset axis
sj_train_subtest.index = sj_train_subtest['week_start_date']
iq_train_subtest.index = iq_train_subtest['week_start_date']
## make plot
figs, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 9))
sj_train_subtest.total_cases.plot(ax=axes[0], label="Actual")
sj_train_subtest.fitted.plot(ax=axes[0], label="Predictions")

iq_train_subtest.total_cases.plot(ax=axes[1], label="Actual")
iq_train_subtest.fitted.plot(ax=axes[1], label="Predictions")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()

## FIT THE MODELS
model_sj.fit(X_sj,Y_sj)
model_iq.fit(X_iq,Y_iq)

#predict for each city using test set
sj_predictions = model_sj.predict(X_sj_t).astype(int)
iq_predictions = model_iq.predict(X_iq_t).astype(int)

#take a look at predictions
iq_predictions

#read in the driven data submission example and add the predictions
submission = pd.read_csv("submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])

submission.to_csv("submissions/svr_predictions5.csv")

#these features performed better
features2=[
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
'station_min_temp_c',
'cases_prev_wk',
'total_cases'
    ]      

#TRAIN
X_sj= sj[features2]
Y_sj = sj['total_cases']

X_iq= iq[features2]
Y_iq = iq['total_cases']

#TEST
X_sj_t= sj_test[features2]

X_iq_t= iq_test[features2]


#need to make sure no NaNs
X_sj.fillna(method='bfill', inplace=True)
X_iq.fillna(method='bfill', inplace=True)

X_sj_t.fillna(method='bfill', inplace=True)
X_iq_t.fillna(method='bfill', inplace=True)

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools import eval_measures

def get_best_model(train, test):
    # Step 1: specify the form of the model
    
  

                    
    model_formula = "total_cases ~ 1 + "                     "rolling_avg_reanalysis_specific_humidity_g_per_kg + "                     "rolling_avg_reanalysis_dew_point_temp_k + "                     "rolling_avg_station_min_temp_c + "                     "rolling_avg_station_avg_temp_c + "                     "station_min_temp_c + "                    "reanalysis_dew_point_temp_k +"                     "cases_prev_wk"
                   
    
    grid = 10 ** np.arange(-8, -3, dtype=np.float64)
                    
    best_alpha = []
    best_score = 1000
        
    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)
            
    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model

sj_best_model = get_best_model(X_sj,X_sj_t)
iq_best_model = get_best_model(X_iq,X_iq_t)

figs, axes = plt.subplots(nrows=2, ncols=1,figsize=(16, 12))

# plot sj
sj['fitted'] = sj_best_model.predict(X_sj)
sj.fitted.plot(ax=axes[0], label="Predictions")
sj.total_cases.plot(ax=axes[0], label="Actual", title="SAN JUAN")

# plot iq
iq['fitted'] = iq_best_model.predict(X_iq)
iq.fitted.plot(ax=axes[1], label="Predictions")
iq.total_cases.plot(ax=axes[1], label="Actual", title="IQUITOS")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()

from sklearn.metrics import mean_absolute_error, mean_squared_error
print("MAE San Juan:", mean_absolute_error(sj['total_cases'], sj['fitted']))
print("MSE San Juan:", mean_squared_error(sj['total_cases'], sj['fitted']))
print("MAE Iquitos:", mean_absolute_error(iq['total_cases'], iq['fitted']))
print("MSE Iquitios:", mean_squared_error(iq['total_cases'], iq['fitted']))

sj_predictions = sj_best_model.predict(X_sj_t).astype(int)
iq_predictions = iq_best_model.predict(X_iq_t).astype(int)

submission = pd.read_csv("submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("glm_improved2.csv")

print(sj_best_model.summary())

