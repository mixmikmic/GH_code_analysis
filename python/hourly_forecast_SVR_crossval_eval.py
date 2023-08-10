get_ipython().magic('pylab inline')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import history
import errors

# some nice colors - like Dad always said, it's better to look good than to be good
gray_light = '#d4d4d2'
gray_med = '#737373'
red_orange = '#ff3700'
blue_light = '#1ebed7'

# Import the BGE hourly electricity data and weather data using import_funcs.py class
elec = pd.read_csv('data/elec_hourly_oldApt_2014-04-30.csv', parse_dates=True, index_col=0)
weather = pd.read_csv('data/weather_2015-02-01.csv', parse_dates=True, index_col=0)

# Merge into one Pandas dataframe
elec_and_weather = pd.merge(weather, elec, left_index=True, right_index=True)

# Remove unnecessary fields from dataframe
del elec_and_weather['tempm'], elec_and_weather['COST'], elec_and_weather['UNITS']
del elec_and_weather['precipm']

# Convert windspeed to MPH for my feeble brain to interpret
elec_and_weather['wspdMPH'] = elec_and_weather['wspdm'] * 0.62
del elec_and_weather['wspdm']

elec_and_weather.head()

# Set weekends and holidays to 1, otherwise 0
elec['Atypical_Day'] = np.zeros(len(elec))

# Weekends
elec['Atypical_Day'][(elec.index.dayofweek==5)|(elec.index.dayofweek==6)] = 1

# Holidays, days I worked from home
holidays = ['2014-01-01','2014-01-20']
work_from_home = ['2014-01-21','2014-02-13','2014-03-03','2014-04-04']

for i in range(len(holidays)):
    elec['Atypical_Day'][elec.index.date==np.datetime64(holidays[i])] = 1

for i in range(len(work_from_home)):
    elec['Atypical_Day'][elec.index.date==np.datetime64(work_from_home[i])] = 1
    

# Create new column for each hour of day, assign 1 if index.hour is corresponding hour of column, 0 otherwise

for i in range(0,24):
    elec[i] = np.zeros(len(elec['USAGE']))
    elec[i][elec.index.hour==i] = 1
    
# Example 3am
elec[3][:5]

# MOVED TO history.py

# Set number of hours prediction is in advance
n_hours_advance = 1
# Set number of historic hours used
n_hours_window = 1

elec = history.append_history(elec,'USAGE',n_hours_advance,n_hours_window)

# Define training and testing periods

gridsearch_start = '18-jan-2014'
gridsearch_end = '25-jan-2014'
train_start = '26-jan-2014'
train_end = '24-march-2014'
test_start = '25-march-2014'
test_end = '31-march-2014'

from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing as pre
from sklearn import grid_search

# Set up dataframes for building SVR model

# Need to keep only the t-1 variables for predicting the next hour
X_gridsearch_df = elec[gridsearch_start:gridsearch_end]
del X_gridsearch_df['USAGE']

X_scaling_df = elec[gridsearch_start:train_end]
del X_scaling_df['USAGE']

X_train_df = elec[train_start:train_end]
del X_train_df['USAGE']

X_test_df = elec[test_start:test_end]
del X_test_df['USAGE']

y_gridsearch_df = elec['USAGE'][gridsearch_start:gridsearch_end]
y_train_df = elec['USAGE'][train_start:train_end]
y_test_df = elec['USAGE'][test_start:test_end]





# Numpy arrays for sklearn

X_gridsearch = np.array(X_gridsearch_df)
X_scaling = np.array(X_scaling_df)
X_train = np.array(X_train_df)
X_test = np.array(X_test_df)

y_train = np.array(y_train_df)
y_test = np.array(y_test_df)
y_gridsearch = np.array(y_gridsearch_df)



from sklearn import preprocessing as pre

scaler = pre.StandardScaler().fit(X_scaling)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

get_ipython().magic('time')

parameters = {'C':[.001, .01, .1, 1, 10, 100], 'gamma':[.0001, .001, .01, .1, 1, 10, 100]}

SVR_model = svm.SVR(kernel='rbf')

SVR_model_OptParams = grid_search.GridSearchCV(SVR_model, parameters)
SVR_model_OptParams.fit(X_gridsearch,y_gridsearch)

print 'Optimum hyperparameters C and gamma: ',SVR_model_OptParams.best_params_
print 'N samples for grid search of hyperparameters C and gamma: ',len(X_gridsearch)

get_ipython().magic('time')
scores = cross_validation.cross_val_score(SVR_model_OptParams, X_train, y_train, cv=10)
print(scores.min(), scores.mean(), scores.max())

scores

# Use SVR model to calculate predicted next-hour usage
predict_y_array = SVR_model_OptParams.best_estimator_.predict(X_test_scaled)

# Put it in a Pandas dataframe for ease of use
predict_y = pd.DataFrame(predict_y_array,columns=['USAGE'])
predict_y.index = X_test_df.index

# Plot the predicted values and actual
import matplotlib.dates as dates

plot_start = test_start
plot_end = test_end

fig = plt.figure(figsize=[12,5])
ax = fig.add_subplot(111)
plt.plot(y_test_df.index,y_test_df,color=blue_light,linewidth=1)
plt.plot(predict_y.index,predict_y,color=red_orange,linewidth=1)
plt.ylabel('Electricity Usage (kWh)')
plt.ylim([0,2])
plt.legend(['Actual','Predicted'],loc='best')
ax.xaxis.set_major_formatter(dates.DateFormatter('%b %d'))

#fig.savefig('output_plots/SVM_predict_TS.png')

fig = plt.figure(figsize=(4,4))
plot = plt.plot(y_test_df,predict_y,color=red_orange,marker='.',linewidth=0,markersize=10,alpha=.4)
plot45 = plt.plot([0,2],[0,2],'k')
plt.xlim([0,2])
plt.ylim([0,2])
plt.xlabel('Actual Hourly Elec. Usage (kWh)')
plt.ylabel('Predicted Hourly Elec. Usage (kWh)')

#fig.savefig('output_plots/SVM_plot_errors.png')

#plot_start = test_start
#plot_end = test_end
plot_start = '25-mar-2014 00:00:00'
plot_end = '26-mar-2014 23:00:00'
xticks = pd.date_range(start=plot_start, end=plot_end, freq='6H')

fig = plt.figure(figsize=[10,4])
ax = fig.add_subplot(111)
plot1 = plt.plot(predict_y[plot_start:plot_end].index,predict_y[plot_start:plot_end],color=red_orange,linewidth=2)
plot2 = plt.plot(y_test_df[plot_start:plot_end].index,y_test_df[plot_start:plot_end],color=blue_light,linewidth=2)
#plt_predict = plt.plot(predict_y[plot_start:plot_end].index,predict_y[plot_start:plot_end])
#plt_actual = plt.plot(y_test_df[plot_start:plot_end].index,y_test_df[plot_start:plot_end])
plt.ylabel('Electricity Usage (kWh)')
plt.xticks(xticks)
plt.title('Support Vector Regression')
plt.legend(['Predicted','Actual'],loc='best')
#ax.xaxis.set_major_locator()
ax.xaxis.set_major_formatter(dates.DateFormatter('%H:00 \n%a \n%b %d \n \n'))
#dates.HourLocator(interval=12)


#fig.savefig('output_plots/SVM_predict_TS_zoom.png')

# Plot daily total kWh over testing period
y_test_barplot_df = pd.DataFrame(y_test_df,columns=['USAGE'])
y_test_barplot_df['Predicted'] = predict_y['USAGE']

fig = plt.figure()
ax = fig.add_subplot(111)
y_test_barplot_df.resample('d',how='sum').plot(kind='bar',ax=ax,color=[blue_light,red_orange])
ax.grid(False)
ax.set_ylabel('Total Daily Electricity Usage (kWh)')
ax.set_xlabel('')
# Pandas/Matplotlib bar graphs convert xaxis to floats, so need a hack to get datetimes back
ax.set_xticklabels([dt.strftime('%b %d') for dt in y_test_df.resample('d',how='sum').index.to_pydatetime()],rotation=0)

#fig.savefig('output_plots/SVM_predict_DailyTotal.png')













