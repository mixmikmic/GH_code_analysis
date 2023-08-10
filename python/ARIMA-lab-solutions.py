#### Starter code

import pandas as pd
import numpy as np


get_ipython().magic('matplotlib inline')
# get the Datas
data = pd.read_csv('assets/datasets/train.csv')
data.set_index('Date', inplace=True)
data.head()

import datetime

data.dtypes

store1 = data[data.Store == 1]
# there are about 36 different stores in this dataset.

store1_sales = pd.DataFrame(store1.Weekly_Sales.groupby(store1.index).sum())
store1_sales.dtypes
# Grouped weekly sales by store 1

#remove date from index to change its dtype because it clearly isnt acceptable.
store1_sales.reset_index(inplace = True)

#converting 'date' column to a datetime type
store1_sales['Date'] = pd.to_datetime(store1_sales['Date'])
# resetting date back to the index
store1_sales.set_index('Date',inplace = True)



store1_sales.head()
# I think its a datetime object now.  

store1_sales.index.dtype
#confirmed

rolmean1 = pd.rolling_mean(store1_sales, window = 1)
rolmean4 = pd.rolling_mean(store1_sales, window = 4)
rolmean13 = pd.rolling_mean(store1_sales, window = 13)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8))
mean = plt.plot(rolmean1, color='red', label='Rolling Mean (1 week)')
mean = plt.plot(rolmean4, color='blue', label='Rolling Mean(4 week)')
mean = plt.plot(rolmean13, color='green', label='Rolling Mean(13 week)')

plt.legend(loc='best')
plt.title('Rolling Mean for 1,4 and 13 Months')
plt.show()

print '1 Week Lag AutoCorr', store1_sales['Weekly_Sales'].autocorr(lag=1)
print '2 Week Lag AutoCorr', store1_sales['Weekly_Sales'].autocorr(lag=2)
print '52 Week Lag AutoCorr', store1_sales['Weekly_Sales'].autocorr(lag=52)

# 1 Month Autocorr and Partial Autocorr
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(store1_sales, lags = 1)
plot_pacf(store1_sales, lags = 1)

# If you do not use plt.show() it will plot the same vize twice.
# This may be a bug you can fix and submit, to contribute to open source!
plt.show()

# 2 Month Autocorr and Partial Autocorr

plot_acf(store1_sales, lags =2)
plot_pacf(store1_sales, lags =2)

plt.show()

# 52 Month Autocorr and Partial Autocorr
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(store1_sales, lags =52)
plot_pacf(store1_sales, lags =52)


plt.show()

store1_sales.shape

#shape of information is 143 rows.  
143*0.75
# first 107 rows for training, next 36 rows for testing?

train = store1_sales.head(107)

test = store1_sales.tail(36)

store1_sales['Weekly_Sales'].diff(periods =1)

train.head()

from statsmodels.tsa.arima_model import AR

train_values = train['Weekly_Sales'].values
train_dates = train.index


AR1 = AR(train_values, train_dates).fit()

test.tail()

test_values = test['Weekly_Sales'].values
test_dates = test.index

# The AR Predict takes a start and and end date as values and not a list.
start = '2012-02-24'
end = '2012-10-26'


AR1_pred = AR1.predict(start=start, end = end)

ARIMA()

from sklearn.metrics import mean_absolute_error
mean_absolute_error(test_values, AR1_pred)

AR1_residuals = test_values - AR1_pred

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.scatter(test_values, AR1_pred)
plt.show()

plt.plot(AR1_residuals)
plt.show()

AR2 = ARMA(endog = train_values, dates = train_dates, order = (2,0)).fit()

# Same start and end we outlined earlier.
AR2_pred = AR2.predict(start=start, end = end)

mean_absolute_error(test_values, AR2_pred)

AR2_2 = ARMA(endog = train_values, dates = train_dates, order = (2,2)).fit()

# Same start and end we outlined earlier.
AR2_2_pred = AR2_2.predict(start=start, end = end)

mean_absolute_error(test_values, AR2_2_pred)



AR2.summary()

AR2_2.summary()



