import numpy as np
import matplotlib
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (15, 6)

import pandas as pd
pd.options.display.max_rows = 10

data = pd.read_csv('../data/AirPassengers.csv', parse_dates=True, index_col='Month',date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m'))
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

#TODO: test stationarity & try decompose

#TODO: try to apply logarithm & shift, then test stationarity again.

from statsmodels.tsa.arima_model import ARIMA

#TODO: instatiate ARIMA model and fit it. (use order=(2, 0, 2) or (2, 1, 2) based on used ts (shift or no shift)) Visualize fitted values.

model = ARIMA( ... )  
results = ...

#plt.plot(results.fittedvalues, c='r')

import datetime
from datetime import timedelta

start=ts.index[-1]
end=ts.index[-1] + timedelta(days=365 * 2)

prediction = results.predict(start=ts.index[-1], end=ts.index[-1] + timedelta(days=365 * 2), dynamic=True)

#TODO: inverse-transform prediciton to the original space

from statsmodels.tsa.statespace.sarimax import SARIMAX

#TODO: try to fit SARIMAX model

