import numpy as np
import matplotlib

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import pandas as pd

pd.options.display.max_rows = 10
matplotlib.rcParams['figure.figsize'] = (15, 6)

data = pd.read_csv('../data/AirPassengers.csv', parse_dates=True, index_col='Month', date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m'))
ts = data['#Passengers']
data

plt.plot(ts)
plt.show()

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

def test_stationarity(timeseries):
        
    rolling = timeseries.rolling(center=False, window=12)        
    
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolling.mean(), color='red', label='Rolling Mean')
    plt.plot(rolling.std(), color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
        
    
    df_test = adfuller(timeseries, autolag='AIC')
    output = pd.Series(df_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        
    for key, value in df_test[4].items():
        output['Critical Value (%s)' % key] = value
        
    print('Dickey-Fuller test:')    
    print(output)

test_stationarity(ts)

def decompose(ts):
    
    decomposition = seasonal_decompose(ts)
    plt.subplot(411)
    plt.plot(ts, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    return decomposition

decomposition = decompose(np.log(ts))
test_stationarity(decomposition.resid.dropna())

ts_log = np.log(ts)
ts_log_diff = (ts_log - ts_log.shift()).dropna()

test_stationarity(ts_log_diff)

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts_log_diff, order=(2, 0, 2))  
results = model.fit()

plt.plot(ts_log_diff)
plt.plot(results.fittedvalues, c='r')
plt.title('R^2: %.4f'% sum((results.fittedvalues - ts_log_diff) ** 2))
plt.show()

import datetime
from datetime import timedelta

prediction = results.predict(start=ts.index[-1], end=ts.index[-1] + timedelta(days=365 * 2), dynamic=True)

plt.plot(ts)
plt.plot(np.exp(results.fittedvalues.cumsum() + ts_log[0]), c='r')
plt.plot(np.exp(prediction.cumsum() + ts_log[-1]), c='g')
plt.show()

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

prediction = results.get_prediction(start=ts.index[-1], end=ts.index[-1] + timedelta(days=365 * 2), dynamic=True)
conf_int = prediction.conf_int(alpha=0.05)

plt.plot(ts)
plt.plot(results.fittedvalues, c='r')
plt.plot(prediction.predicted_mean, c='g')
plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='silver')
plt.show()

