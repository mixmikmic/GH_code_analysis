# Some data, in a list
my_data = [12, 5, 17, 8, 9, 11, 21]

# Function for calulating the mean of some data
def mean(data):

    # Initialize sum to zero
    sum_x = 0.0

    # Loop over data
    for x in data:

        # Add to sum
        sum_x += x 
    
    # Divide by number of elements in list, and return
    return sum_x / len(data)

import numpy as np

x = np.arange(10)
x

(x + 3)**2

y = np.random.randn(4,10)
y

y.dot(x)

import pandas as pd

data_url = 'https://www.ncdc.noaa.gov/cag/time-series/global/globe/land/all/1/1880-2015.csv'

[f for f in dir(pd) if f.startswith('read_')]

raw_data = pd.read_csv(data_url, skiprows=3)
raw_data.head()

type(raw_data)

raw_data.shape

raw_data.describe()

raw_data.values

raw_data.index

raw_data.columns

more_data = pd.Series(np.random.random(10), np.arange(6,16), name='more_data')
more_data

raw_data.join(more_data)

temp_anomaly = raw_data.set_index('Year')

temp_anomaly.head()

temp_anomaly.loc[201506]

temp_anomaly.loc[194000:194200]

temp_anomaly[:2]

get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt

temp_anomaly.plot()

temp_anomaly[:25].plot()

temp_anomaly.index[:25]

from datetime import date
pd.to_datetime(['-'.join([str(t)[:4], str(t)[4:]]) for t in temp_anomaly.index])

month_range = pd.date_range('1880-01-01', periods=temp_anomaly.shape[0], freq='M')
month_range

temp_anomaly = (raw_data.set_index(month_range)
                          .drop('Year', axis=1)
                          .rename(columns={'Value':'Anomaly'}))
temp_anomaly.head()

temp_anomaly.plot()

temp_anomaly[:25].plot()

axes = temp_anomaly.plot(figsize=(14,6), legend=False, fontsize=14)
axes.set_title('Land global temperature anomalies', fontsize=24)
axes.set_ylabel('Temperature anomaly (°C)', fontsize=18)
axes.set_xlabel('Date', fontsize=18)

temp_anomaly['03/1990':'12/1991']

temp_anomaly.resample('2Q').mean().plot()

temp_anomaly.rolling(12).mean().plot()

from sklearn import linear_model

x_grid = np.arange(temp_anomaly.shape[0])
regmod = linear_model.LinearRegression()
regmod

regmod.fit(X=x_grid.reshape(-1,1), y=temp_anomaly)

prediction = regmod.predict(x_grid.reshape(-1,1))
prediction

regmod.score(X=x_grid.reshape(-1,1), y=temp_anomaly)

prediction = pd.Series(prediction.flatten(), index=temp_anomaly.index)

axes = temp_anomaly.plot(figsize=(14,6), legend=False, fontsize=14)
prediction.plot(ax=axes, style='--', color='red')

import pymc3 as pm
from theano import shared

with pm.Model() as model:
    
    # Data
    x = shared(x_grid)
    y = shared(temp_anomaly.Anomaly.values)
    
    # Prior distributions
    intercept = pm.Normal('intercept', mu=0, sd=100)
    early_slope = pm.Normal('early_slope', mu=0, sd=100)
    late_slope = pm.Normal('late_slope', mu=0, sd=100)
    σ = pm.Uniform('σ', 0, 100)
    
    # Switchpoint
    τ = pm.DiscreteUniform('τ', 0, x_grid.max())
    
    # Early and late phase means
    μ_early = intercept + (x[:τ] - τ)*early_slope
    μ_late = intercept + (x[τ:] - τ)*late_slope
    
    # Data likelihoods
    pm.Normal('early_likelihood', μ_early, sd=σ, observed=y[:τ])
    pm.Normal('late_likelihood', μ_late, sd=σ, observed=y[τ:])

with model:
    trace = pm.sample(10000)

pm.traceplot(trace[5000:], varnames=['intercept', 'early_slope', 'late_slope']);

b = trace['intercept'].mean()
m1 = trace['early_slope'].mean()
m2 = trace['late_slope'].mean()
switch = int(trace['τ'].mean())

prediction = b + (x_grid-switch)*m1
prediction[switch:] = (b + (x_grid-switch)*m2)[switch:]
temp_anomaly['prediction'] = prediction

axes = temp_anomaly.plot(figsize=(14,6), legend=False, fontsize=14)
axes.plot(temp_anomaly.prediction)

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, DotProduct, ConstantKernel

kernel = ConstantKernel() + DotProduct() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)

gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)

X = x_grid.reshape(-1,1)
y = temp_anomaly.Anomaly.values

gp.fit(X, y)

gp.kernel_

X_pred = np.concatenate([X, np.arange(1632, 1732).reshape(-1,1)])

y_pred, sigma = gp.predict(X_pred, return_std=True)

plt.figure(figsize=(14,6))
plt.plot(x_grid, temp_anomaly.Anomaly.values, label='Data')
plt.plot(X_pred.flatten(), y_pred, color='red', label='Prediction')
plt.fill(np.concatenate([X_pred.flatten(), X_pred.flatten()[::-1]]),
         np.concatenate([y_pred - 2*sigma,
                        (y_pred + 2*sigma)[::-1]]),
         alpha=.5, fc='grey', ec='None', label='95% CI')
plt.xticks([])
plt.legend(loc='upper left');

