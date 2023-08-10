get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from warnings import filterwarnings
filterwarnings('ignore')

df=pd.read_csv('dengue_features_train.csv')
labels = pd.read_csv('dengue_labels_train.csv')
test = pd.read_csv('dengue_features_test.csv')

df=pd.merge(data, labels, on=['city', 'year', 'weekofyear'])

df['week_start_date']=pd.to_datetime(df['week_start_date'])
df['month']=df['week_start_date'].dt.month
### reset axis
df.index = df['week_start_date']
del df['week_start_date']

iq=df[df['city']=='iq']
sj=df[df['city']=='sj']

iq.fillna(method='ffill', inplace=True)
sj.fillna(method='ffill', inplace=True)

plt.figure(figsize=(11,9))
plt.plot(sj['total_cases'],label="San Juan", color='b')
plt.plot(iq['total_cases'],label="Iquitos", color='g')
plt.legend()

#randomly plot one year only - 2004 
plt.figure(figsize=(11,9))
plt.plot(sj['2004']['total_cases'],label="San Juan", color='b')
plt.plot(iq['2004']['total_cases'],label="Iquitos", color='g')
plt.legend()

# not stationary
# no clear increase throughout the years (1990-2010)
# no clear increase throughout the months (jan - dec)
# the outbreaks in the disease dont appear to be dependent on the time stamp-
# it's more the mechanics of how the disease spreads through people and networks, and the contagiousness of dengue

#a look into rolling mean and std and the Dickey_Fuller test 

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=3)#three month rolling avg
    rolstd = pd.rolling_std(timeseries, window=3)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

plt.figure(figsize=(10,7))
test_stationarity(sj['total_cases'])

plt.figure(figsize=(10,7))
test_stationarity(iq['total_cases'])

#plot the log of the total cases
plt.figure(figsize=(10,7))
ts_log = np.log(df['total_cases'])
plt.plot(ts_log)

#plot 3 week rolling mean of the log
plt.figure(figsize=(10,7))
moving_avg = pd.rolling_mean(ts_log,3)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)

#differencing
plt.figure(figsize=(10,7))
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)

#pretty cool - hope to take a class on timeseries analytics in the future

ts_log.replace([np.inf, -np.inf], np.nan)
ts_log.dropna(inplace=True)

ts_log=ts_log+1

#print decomposition of the time series
import statsmodels.api as sm

#decomposition = seasonal_decompose(ts_log.price.values, freq=30)
# deal with missing values. see issue
sj.total_cases.interpolate(inplace=True)

res = sm.tsa.seasonal_decompose(sj.total_cases.values, freq=30)
resplot = res.plot()

#need floats for this to work
sj['total_cases']=sj['total_cases'].astype(float)
iq['total_cases']=iq['total_cases'].astype(float)

#from pandas import read_csv
#from pandas import datetime
from matplotlib import pyplot #keep consistent with the tutorial
from pandas.tools.plotting import autocorrelation_plot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = sj['total_cases']
autocorrelation_plot(series)

pyplot.xticks(np.arange(0, 50, 1.0))
pyplot.show()

from statsmodels.tsa.arima_model import ARIMA
#import pandas as pd
from pandas import DataFrame

model = ARIMA(series, order=(6,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

# df=pd.read_csv('dengue_features_train.csv')
# labels = pd.read_csv('dengue_labels_train.csv')
# test = pd.read_csv('dengue_features_test.csv')
# submission=pd.read_csv('submission_format.csv')

# df['week_start_date']=pd.to_datetime(df['week_start_date'])
# test['week_start_date']=pd.to_datetime(test['week_start_date'])

# df=pd.merge(data, labels, on=['city', 'year', 'weekofyear'])
# ### reset axis
# df.index = df['week_start_date']
# del df['week_start_date']

# test.index = test['week_start_date']
# del test['week_start_date']

# iq=df[df['city']=='iq']
# sj=df[df['city']=='sj']

# iq_test=test[test['city']=='iq']
# sj_test=test[test['city']=='sj']

# sj['total_cases']=sj['total_cases'].astype(float)
# iq['total_cases']=iq['total_cases'].astype(float)

from sklearn.metrics import mean_absolute_error

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series1 = sj['total_cases']

X = series.values
#size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_absolute_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

#check how far in the future to forecast
print(len(sj_test))
print(len(iq_test))

#get the forecast for San Juan
sj_forecast= model_fit.forecast(260)

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = iq['total_cases']

X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_absolute_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

#get forecast for Iquitos
iq_forecast= model_fit.forecast(156)
len(iq_forecast[0])

submission = pd.read_csv("submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_forecast[0], iq_forecast[0]])
submission.to_csv("submissions/arima.csv")

