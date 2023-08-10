import warnings
import itertools

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

X = pd.read_csv("data/dengue_features_train.csv")
y = pd.read_csv("data/dengue_labels_train.csv")

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

#setting a datetime index
y.index = pd.DatetimeIndex(X.week_start_date)

#seperating the data for the 2 cities
sj_cases = y.total_cases[y.city == 'sj']
iq_cases = y.total_cases[y.city == 'iq']

#month sampling
sj_monthly = sj_cases.resample('M').sum()
iq_monthly = iq_cases.resample('M').sum()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(sj_monthly, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(sj_monthly, lags=40, ax=ax2)

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
# seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

pdq = [(3,0,1),
       (3,0,2),
       (3,0,3),
       (3,0,4),
       (4,0,1),
       (4,0,2),
       (4,0,3),
       (4,0,4),
       (1,0,3),
       (2,0,3),
       (1,0,4),
       (2,0,4)
      ]

seasonal_pdq = [
       (3,2,2,12),
       (4,2,1,12),
       (4,2,2,12),
       (2,2,3,12),
       (1,2,4,12),
       (2,2,4,12)
      ]

best_aic = 100000
optimal_pdq = 0
optimal_seasonal_pdq = 0

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(sj_monthly,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            
            if results.aic < best_aic:
                best_aic = results.aic
                optimal_pdq = param
                optimal_seasonal_pdq = param_seasonal

            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

best_aic

optimal_pdq

optimal_seasonal_pdq

mod = sm.tsa.statespace.SARIMAX(sj_monthly, order=(4,1,3), seasonal_order=(1,1,1,12))
results = mod.fit()
print (results.summary())

pred = results.predict()  

from sklearn.metrics import mean_absolute_error

mean_absolute_error(sj_monthly[:-1], pred.shift(-1).dropna().clip(0))

sj_monthly.diff(12).plot()

iq_monthly.diff(12).plot()

