from get_train_test_for_modeling import *
from get_prediction_data import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score
from format_train_test import *
import pickle
import sklearn.cross_validation as cv
from sklearn.ensemble import (GradientBoostingRegressor, 
                              GradientBoostingClassifier, 
                              AdaBoostClassifier,
                              RandomForestClassifier)
import datetime
plt.style.use('ggplot')


get_ipython().magic('matplotlib inline')

# making the training and testing data
filename = 'data_X_y_46059_24hr.csv'
Xy_df = get_Xy_data(filename)

cols_to_keep = ['YY_x', 'MM_x', 'DD_x', 'hh_x', 'WD_x', 'WSPD_x',
                'GST_x', 'WVHT_x', 'DPD_x', 'APD_x', 'BAR_x', 'ATMP_x',
                'WTMP_x', 'DEWP_x', 'ID_x', 't_arrive','WVHT_y']

train_yrs = [1995, 1996, 1997, 1998, 1999, 2000, 2003, 2004, 2006, 2007]
test_yrs  = [2008]

X_train, X_test, y_train, y_test = get_train_test(Xy_df, cols_to_keep, train_yrs, test_yrs)

# fitting a gradient booster model
n_estimators = 60000
params = {'n_estimators': n_estimators, 'max_depth': 3, 'min_samples_split': 4,
          'learning_rate': 0.01, 'loss': 'ls' }
gbr_24 = ensemble.GradientBoostingRegressor(**params)
gbr_24.fit(X_train, y_train)

y_hat_24 = gbr_24.predict(X_test)

mse_train_2 = mean_squared_error(y_train, gbr_24.predict(X_train))
mse_train_2

mse_test_2  = mean_squared_error(y_test, y_hat_24)
mse_test_2  

fig, ax = plt.subplots(figsize=(14,6))
ax.plot(y_test[0::2],':', color='b', label = 'Test Data')
ax.plot(y_hat_24[0::2], ':', label = 'Model Prediction')
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Wave Height (m)', fontsize=14)
#ax.set_title('Test Data Comparison with Model Predictions for Source 46005')
plt.legend(prop={'size': 14})

# making the training and testing data
filename = 'data_X_y_46059_48hr.csv'
Xy_df = get_Xy_data(filename)

cols_to_keep = ['YY_x', 'MM_x', 'DD_x', 'hh_x', 'WD_x', 'WSPD_x',
                'GST_x', 'WVHT_x', 'DPD_x', 'APD_x', 'BAR_x', 'ATMP_x',
                'WTMP_x', 'DEWP_x', 'ID_x', 't_arrive','BAR_y','WVHT_y']

train_yrs = [1995, 1996, 1997, 1998, 1999, 2000, 2003, 2004, 2006, 2007]
test_yrs  = [2008]

X_train48, X_test48, y_train48, y_test48 = get_train_test(Xy_df, cols_to_keep, train_yrs, test_yrs)

## fitting the model for 48 hrs

# fitting a gradient booster model
n_estimators = 6000
params = {'n_estimators': n_estimators, 'max_depth': 3, 'min_samples_split': 4,
          'learning_rate': 0.01, 'loss': 'ls'}
gbr_48 = ensemble.GradientBoostingRegressor(**params)
gbr_48.fit(X_train48, y_train48)

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(gbr_24.staged_predict(X_test)):
    if i%100 == 0:
        print('Predicting for {}'.format(i))
    test_score[i] = gbr_24.loss_(y_test, y_pred)

test_score[:]

filename = '../data/NDBC_all_data_all_years.csv'
buoyID_train = [46059]
buoyID_test = [46026]

print('Processing the data for training and testing')

# getting the testing and traing data
#data_train_46005 = get_train_bouys(filename, buoyID_train[0])
data_train_46059 = get_train_bouys(filename, buoyID_train[0])
data_labels_46026  = get_train_bouys(filename, buoyID_test[0])

yr_lst = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002,
          2003, 2004, 2006, 2007, 2008]

data_train_46059_hr = join_all_hourly_data(data_train_46059, yr_lst)
data_labels_46026_hr  = join_all_hourly_data(data_labels_46026, yr_lst)

# adding time delta for the data frames
data_train_46059_t = add_time_delta(data_train_46059_t)

# adding time to time_delta
data_train_46059_t = add_time_y(data_train_46059_t)

#rounding time
data_train_46059_t = round_time_y(data_train_46059_t)

predict_hrs = [hr for hr in range(24,166,24)]
for hr in predict_hrs:
    data_train_46059_t['time_delta_{}'.format(str(hr))] = data_train_46059_t['t_arrive'].apply(lambda x: datetime.timedelta((x+hr)/24))
    data_train_46059_t['time_y_{}'.format(str(hr))] = data_train_46059_t.index + data_train_46059_t['time_delta_{}'.format(hr)]
    data_train_46059_t['time_y_hr_{}'.format(hr)]  = data_train_46059_t['time_y_{}'.format(hr)].apply(lambda dt: datetime.datetime(dt.year,
                                                                                             dt.month,
                                                                                             dt.day,
                                                                                             dt.hour,
                                                                                             0,0))

lst_merge = ['time_yr_hr', 'time_yr_hr_24', 'time_yr_hr_48', 'time_yr_hr_72',
             'time_yr_hr_96', 'time_yr_hr_120', 'time_yr_hr_144']

lst_merge[0]

data_X_y_46059 = pd.merge(data_train_46059_t,
                          data_labels_46026_hr,
                          how='left', left_on=lst_merge[0], right_on='id')

for item in lst_merge:
    #data_X_y_46005 = pd.merge(data_train_46005_t, data_test_46026, left_on='time_y_hr', right_index=True)
    merged_df = pd.merge(data_train_46059_t,
                         data_labels_46026_hr,
                         how='left', left_on=item, right_on='id')
    merge_df.to_csv('data_X_y_46059_train_{}.csv'.format(item.split('_')[-1]), index=False)













