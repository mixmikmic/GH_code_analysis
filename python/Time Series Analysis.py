get_ipython().magic('matplotlib inline')
import os
import pandas as pd
import numpy as np
import datetime
from datetime import date
from fbprophet import Prophet
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from scipy.signal import savgol_filter

# choose file
filename = 'data/Daylight_saving_time.csv'

# read in data
filepath = os.path.join(os.path.dirname(os.path.realpath('__file__')), filename)
df = pd.read_csv(filepath)

# recode zeros as NaN
df = df.replace('0', np.nan)

# format datestamp column
df.ds = pd.to_datetime(df.ds)

# log values
df.yOri = df.y
df.y = np.log(df.y)

# special events ("holidays")

# clocks move forward
springforward = pd.DataFrame({
  'holiday': 'springforward',
  'ds': pd.to_datetime(['2008-03-09', '2009-03-08',
                        '2010-03-14', '2011-03-13',
                        '2012-03-11', '2013-03-10',
                        '2014-03-09', '2015-03-08']),
  'lower_window': -14,
  'upper_window': 14,
})

# clocks move backward
fallback = pd.DataFrame({
  'holiday': 'fallback',
  'ds': pd.to_datetime(['2008-11-02', '2009-11-01',
                        '2010-11-07', '2011-11-06',
                        '2012-11-03', '2013-11-03',
                        '2014-11-02', '2015-11-01']),
  'lower_window': -14,
  'upper_window': 0
})

# dates when major changes to DST were discussed/implemented/dismissed
news = pd.DataFrame({
  'holiday': 'news',
  'ds': pd.to_datetime(['2010-04-14', '2010-09-12',
                        '2011-02-08', '2011-06-15',
                        '2011-09-15', '2011-09-20',
                        '2012-11-05', '2013-07-08',
                        '2014-07-22', '2015-10-25']),
  'lower_window': -7,
  'upper_window': 21
})

holidays = pd.concat((springforward, fallback, news))
print holidays

# forecast horizon
H = 365

# frequency of simulated forecasts
h = H/2

# tuning and validation: simulated historical forecast

# for storing forecast results and cutoffdates
results = pd.DataFrame()
cutoff = []

# run forecast simulations
i = 0
while (len(df)-i > 3*H): 

    # define training data
    train = df[i:(i+(3*H))] # use 3 periods of data for training

    # fit time series model
    m = Prophet(interval_width=0.95,
                changepoint_prior_scale=0.01, # default is 0.05, decreasing it makes trend more generalizable
                holidays=holidays,
                holidays_prior_scale=10, # default is 10, strength of the holiday components
                yearly_seasonality=True, # since daylight savings occur annually
                weekly_seasonality=True, # people conduct more searches on weekdays
                seasonality_prior_scale=10, # default is 10, larger values allow larger seasonal fluctuations
                # mcmc_samples=500 # to generate confidence intervals for seasonality and holiday components
               )
    m.fit(train);
    
    # future dates for which to make forecasts
    future = m.make_future_dataframe(periods=H)
    
    # make forecast
    forecast = m.predict(future)
    resultsH = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(H)
    
    # get actual values to compare with predicted values
    resultsH = df.merge(resultsH, how='right')
        
    # sort by increasing date
    resultsH = resultsH.sort_values(by='ds')
    
    # record cutoff dates
    cutoffDate = resultsH['ds'].iloc[0]
    cutoffDate = cutoffDate.strftime('%Y %b')
    cutoff = cutoff + [cutoffDate]
    
    # compile results
    results = pd.concat((results, resultsH))
    
    print 'Counting the days...', i
    i = i + h

# number of simulated forecasts
ns = len(cutoff)

# color-code each simulation run
colors = cm.rainbow(np.linspace(1, 0, ns))

# plot simulated forecast results
plt.plot_date(df.ds, df.y, fmt='.', ms=1, c='k', label='')
i = 0
for s in range(ns):
    plt.fill_between(results.ds.values[i:(i+H)],
                     results.yhat_upper[i:(i+H)], results.yhat_lower[i:(i+H)],
                     facecolor=colors[s], alpha=0.5,
                     linewidth=0.0,
                     label=cutoff[s])
    i = i + H
# legend showing the start date of each forecast horizon
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
           title='Forecast Start Date')
plt.savefig('images/time-forecast.png', format='png', bbox_inches='tight', dpi=1000)
plt.show()

# prediction error
error = abs(results.yhat - results.y)
error = error.values.reshape(ns,H)

# average error with respect to forecast horizon
errorMean = np.nanmean(error, axis=0)

# smooth error trend
errorMeanSmooth = savgol_filter(errorMean, 365, 3)

# plot error along forecast horizon
plt.xlim([1,H])
plt.plot(range(H), errorMeanSmooth, c='k', lw=2)
plt.plot(range(H), errorMean, c='r', alpha=0.5)
plt.xlabel('Forecast Horizon (days)')
plt.ylabel('Mean Absolute Prediction Error')
plt.savefig('images/time-error.png', format='png', bbox_inches='tight', dpi=1000)
plt.show()

# component plots
m.plot_components(forecast)
compPlot = m.plot_components(forecast)
compPlot.savefig('images/time-components.png', format='png', bbox_inches='tight', dpi=1000)

# tuning and validation: simulated historical forecast

# for storing forecast results and cutoffdates
results2 = pd.DataFrame()
cutoff = []

# run forecast simulations
i = 0
while (len(df)-i > 3*H): 

    # define training data
    train = df[i:(i+(3*H))] # use 3 periods of data for training

    # fit time series model
    m = Prophet(interval_width=0.95,
                changepoint_prior_scale=10, # default is 0.05, decreasing it makes trend more generalizable
                holidays=holidays,
                holidays_prior_scale=10, # default is 10, strength of the holiday components
                yearly_seasonality=True, # since daylight savings occur annually
                weekly_seasonality=True, # people conduct more searches on weekdays
                seasonality_prior_scale=10, # default is 10, larger values allow larger seasonal fluctuations
                # mcmc_samples=500 # to generate confidence intervals for seasonality and holiday components
               )
    m.fit(train);
    
    # future dates for which to make forecasts
    future = m.make_future_dataframe(periods=H)
    
    # make forecast
    forecast = m.predict(future)
    results2H = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(H)
    
    # get actual values to compare with predicted values
    results2H = df.merge(results2H, how='right')
        
    # sort by increasing date
    results2H = results2H.sort_values(by='ds')
    
    # record cutoff dates
    cutoffDate = results2H['ds'].iloc[0]
    cutoffDate = cutoffDate.strftime('%Y %b')
    cutoff = cutoff + [cutoffDate]
    
    # compile results
    results2 = pd.concat((results2, results2H))
    
    print 'Counting the days...', i
    i = i + h
    

# prediction error
error2 = abs(results2.yhat - results2.y)
error2 = error2.values.reshape(ns,H)

# average error with respect to forecast horizon
errorMean2 = np.nanmean(error2, axis=0)

# smooth error trend
errorMeanSmooth2 = savgol_filter(errorMean2, 365, 3)

# plot error along forecast horizon
plt.xlim([1,H])
plt.plot(range(H), errorMeanSmooth2, c='r', lw=2, label='scale = 10')
plt.plot(range(H), errorMeanSmooth, c='k', lw=2, label='scale = 0.01')
plt.xlabel('Forecast Horizon (days)')
plt.ylabel('Mean Absolute Prediction Error')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
           title='Change Point Prior')
plt.savefig('images/time-errorCompare.png', format='png', bbox_inches='tight', dpi=1000)
plt.show()

