import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('../vb_data/agg_stations.csv')

df.info()

df['update'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df['update']]

df.set_index('update', inplace = True)

24*4

decomposition = seasonal_decompose(df[0:960].available, freq=96)  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=96)
    rolstd = pd.rolling_std(timeseries, window=96)

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in list(dftest[4].items()):
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(df[0:960].available)

df['first_difference'] = df[0:960].available - df[0:960].available.shift(1)  
test_stationarity(df.first_difference.dropna(inplace=False))

df['seasonal_difference'] = df[0:960].available - df[0:960].available.shift(96)
test_stationarity(df.seasonal_difference.dropna(inplace=False))

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(df[0:960].available, nlags=30)
lag_pacf = pacf(df[0:960].available, nlags=10, method='ols')

#Plot ACF: 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df[0:960])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df[0:960])),linestyle='--',color='gray')
#plt.xlim([25, 26])
#plt.ylim([0.05, 0.075])
plt.title('Autocorrelation Function')

#Plot PACF:
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df[0:960])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df[0:960])),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.xlim([4, 6])
plt.ylim([0.0, 0.1])
plt.tight_layout()

mod = sm.tsa.statespace.SARIMAX(df[0:960].available, trend='n', order=(6,0,26), seasonal_order=(1,1,7,24), enforce_stationarity = False, enforce_invertibility = False)
results = mod.fit()
results.summary()

data = df.reset_index()
data['forecast'] = results.predict(start = 900, end= 1006, dynamic=True)
data.available.iloc[:960].plot()
data.forecast.plot(figsize=(20,8))
data[['available', 'forecast']].plot(figsize=(20, 8)) 

mod = sm.tsa.statespace.SARIMAX(df[0:960].available, trend='n', order=(1,0,1), seasonal_order=(1,0,1,4))
results = mod.fit()
results.summary()

data = df.reset_index()
data['forecast'] = results.predict(start = 900, end= 1006, dynamic=True)
data.available.iloc[:960].plot()
data.forecast.plot(figsize=(20,8))
data[['available', 'forecast']].plot(figsize=(20, 8)) 

mod = sm.tsa.statespace.SARIMAX(df[0:960].available, trend='n', order=(1,0,0), seasonal_order=(1,1,7,24), enforce_invertibility = False)
results = mod.fit()
results.summary()

data = df.reset_index()

data['forecast'] = results.predict(start = 900, end= 1006, dynamic=True)
data.available.iloc[:960].plot()
data.forecast.plot(figsize=(20,8))
data[['available', 'forecast']].plot(figsize=(20, 8)) 

mod = sm.tsa.statespace.SARIMAX(df[0:960].available, trend='n', order=(1,0,1), seasonal_order=(1,1,7,24), enforce_invertibility = False)
results = mod.fit()
results.summary()

data = df.reset_index()
data['forecast'] = results.predict(start = 900, end= 1006, dynamic=True)
data.available.iloc[:960].plot()
data.forecast.plot(figsize=(20,8))
data[['available', 'forecast']].plot(figsize=(20, 8)) 

mod = sm.tsa.statespace.SARIMAX(df[0:960].available, trend='n', order=(4,0,4), seasonal_order=(1,0,1,4), 
                                enforce_stationarity =False, enforce_invertibility = False)
results = mod.fit()
results.summary()

data = df.reset_index()
data['forecast'] = results.predict(start = 900, end= 1006, dynamic=True)
data.available.iloc[:960].plot()
data.forecast.plot(figsize=(20,8))
data[['available', 'forecast']].plot(figsize=(20, 8)) 

mod = sm.tsa.statespace.SARIMAX(df[0:960].available, trend='n', order=(1,0,1), seasonal_order=(4,0,7,24), 
                                enforce_stationarity =False, enforce_invertibility = False)
results = mod.fit()
results.summary()

predictions = pd.DataFrame(results.predict(start = 900, end= 1006, dynamic=True))
predictions.columns = ["forecast"]
data = df.reset_index()
data.drop('forecast', axis =1, inplace=True)
data = data.join(predictions)

data['forecast'] = results.predict(start = 900, end= 1006, dynamic=True)
data.available.iloc[:960].plot()
data.forecast.plot(figsize=(20,8))
data[['available', 'forecast']].plot(figsize=(20, 8)) 
#plt.savefig('ts_df_predict.png', bbox_inches='tight')

mod = sm.tsa.statespace.SARIMAX(df[0:960].available, trend='n', order=(1,0,0), seasonal_order=(4,0,7,24), enforce_invertibility = False)
results = mod.fit()
results.summary()

data = df.reset_index()
data['forecast'] = results.predict(start = 900, end= 1006, dynamic=True)
data.available.iloc[:960].plot()
data.forecast.plot(figsize=(20,8))
data[['available', 'forecast']].plot(figsize=(20, 8)) 

mod = sm.tsa.statespace.SARIMAX(df[0:960].available, trend='n', order=(1,0,1), seasonal_order=(4,0,8,24), 
                                enforce_stationarity =False, enforce_invertibility = False)
results = mod.fit()
results.summary()

data = df.reset_index()
data['forecast'] = results.predict(start = 900, end= 1006, dynamic=True)
data.available.iloc[:960].plot()
data.forecast.plot(figsize=(20,8))
data[['available', 'forecast']].plot(figsize=(20, 8)) 



