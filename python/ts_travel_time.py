import csv
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.tsa.arima_model 
from datetime import datetime, timedelta
from statsmodels.graphics.api import qqplot

get_ipython().magic('matplotlib notebook')

df_data = pd.read_csv('data/preprocessed_input.csv')

df_data

df_travel_time_mean = df_data.filter(regex='A2_routetime_median').mean()
df_travel_time_mean

df_travel_time = df_data.filter(regex='A2_routetime_median').fillna(df_travel_time_mean)
df_travel_time

time_len = len(df_travel_time)*len(df_travel_time.iloc[0,0:9])
travel_time_index = [pd.to_datetime(df_data.iloc[0,0])]*time_len + np.arange(time_len)*timedelta(minutes=20)
travel_time_index

travel_time = pd.Series(pd.DataFrame.as_matrix(df_travel_time.iloc[:,0:9]).reshape(-1))
travel_time.index = pd.Index(travel_time_index)
travel_time    

travel_time.plot()
plt.show()

sm.tsa.stattools.adfuller(travel_time)[:2]

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(travel_time.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(travel_time, lags=40, ax=ax2)
fig

travel_ts=statsmodels.tsa.arima_model.ARIMA(travel_time, (1,0,0)).fit()

print travel_ts.params

print travel_ts.summary()

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(travel_ts.resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(travel_ts.resid, lags=40, ax=ax2)
fig

resid = travel_ts.resid
qqplot(resid,line='q',fit=True)

resid

travel_ts.aic

fig = plt.figure(figsize=(8, 8))
plt.plot(travel_time)
plt.plot(travel_ts.fittedvalues)
plt.show()

df_travel_time_median = df_data.filter(regex='A2_routetime_median').median()
df_travel_time_median

df_travel_time = df_data.filter(regex='A2_routetime_median').fillna(df_travel_time_median)

time_len = len(df_travel_time)*len(df_travel_time.iloc[0,0:9])
travel_time_index = [pd.to_datetime(df_data.iloc[0,0])]*time_len + np.arange(time_len)*timedelta(minutes=20)

travel_time = pd.Series(pd.DataFrame.as_matrix(df_travel_time.iloc[:,0:9]).reshape(-1))
travel_time.index = pd.Index(travel_time_index)

sm.tsa.stattools.adfuller(travel_time)[:2]

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(travel_time.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(travel_time, lags=40, ax=ax2)
fig

travel_ts=statsmodels.tsa.arima_model.ARIMA(travel_time, (11,0,1)).fit()

print travel_ts.summary()

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(travel_ts.resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(travel_ts.resid, lags=40, ax=ax2)
fig

travel_ts.aic

fig = plt.figure(figsize=(8, 8))
plt.plot(travel_time)
plt.plot(travel_ts.fittedvalues)
t_index = 109
plt.xlim(travel_time_index[t_index], travel_time_index[t_index+80])
plt.show()

travel_time

travel_ts.params

travel_ts.predict()

travel_ts.predict(travel_time_index[-1]+timedelta(minutes=20),travel_time_index[-1]+6*timedelta(minutes=20))




