import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
matplotlib.style.use('ggplot')
from datetime import datetime as dt
import holtwinters as hw
from math import sqrt
from sklearn import linear_model
import numpy as np
from operator import add
import sklearn 

# Import the quarterly uninsured data into a dataframe. 
with open('uninsured_time_series.csv', 'rU') as f:
    reader = csv.reader(f)
    col_names = reader.next()
    rows = [(dt.strptime(row[0], '%y-%b'), float(row[1])) for row in reader]

unins_df = pd.DataFrame(rows, columns=col_names)
raw_ls = list(unins_df['uninsured_proportion'])

# Optimized parameters. 
hw_fc, alpha, beta, rmse = hw.linear(raw_ls, 3)

print alpha, beta
print rmse, rmse/np.mean(raw_ls)

plt.plot(raw_ls, color='red', marker='o', markersize=4)
plt.plot([None]*(len(raw_ls)-1) + [raw_ls[-1]] +hw_fc, color='blue', marker='o', alpha=.4)

# Tuning parameters by hand. 
a, b = (.8, .85)
hw_fc, alpha, beta, rmse = hw.linear(raw_ls, 3, a, b)

print alpha, beta
print rmse, rmse/np.mean(raw_ls)

plt.plot(raw_ls, color='red', marker='o', markersize=4)
plt.plot([None]*(len(raw_ls)-1) + [raw_ls[-1]] + hw_fc, color='blue', marker='o', alpha=.4)

# Import data. 
with open('monthly_ndi_official.csv', 'rb') as f:
    reader= csv.reader(f)
    col_names = reader.next()
    rows = [[dt.strptime(row[0], '%m/%d/%y')] + map(float, row[1:]) for row in reader]

ndi_df = pd.DataFrame(rows, columns = col_names)

ts_df = ndi_df[['date', 'brand']].copy()
ts_df.plot()

ts_df['index'] = ts_df['brand']/ts_df.loc[0, 'brand']
ts_df['log_index'] = np.log(ts_df['index'])

X_train = [[x] for x in ts_df.index]
y_train = ts_df['log_index']

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

m = regr.coef_[0]
b = regr.intercept_

print m, b, regr.score(X_train, y_train)

plt.scatter(X_train, y_train)
plt.plot(X_train, y_train)
plt.plot(X_train, regr.predict(X_train), color='blue', linewidth=2)

ts_df['index_log_model'] = regr.predict(X_train)
ts_df['index_log_resid'] = ts_df['log_index'] - ts_df['index_log_model']

temp_df = ts_df.copy()
temp_df.index = temp_df['date']
temp_df['index_log_resid'].plot()

resids = temp_df['index_log_resid']

overlap = 12
hw_forecast = hw.additive(list(resids[:-1*overlap]), 12, 12+overlap)

print "alpha is %s, beta is %s, gamma is %s." %hw_forecast[1:4]
print "rmse is %s." %hw_forecast[4]

temp_ls = list(resids[:-1*overlap]) + hw_forecast[0]

plt.plot(list(resids[:]))
plt.plot(temp_ls, '--')

line_ls = [m*x+b for x in range(len(temp_ls))]
new_log_model_ls = [x for x in map(add, line_ls, temp_ls)]
fc_ls = ts_df.loc[0, 'brand']*np.exp(new_log_model_ls)
plt.plot(ts_df['brand'])
plt.plot(fc_ls, '--')

rmse = sklearn.metrics.mean_squared_error(ts_df.loc[24:36, 'brand'], fc_ls[24:36])**0.5
print rmse, rmse/340.



