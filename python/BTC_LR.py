import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings; warnings.simplefilter('ignore')
import json
import time
import sys
import re
import datetime

# import the pickled dataframe from BTC_DF
df = pd.read_pickle('data/benson_btcsentiment_df.pkl')
df = df[['BTCPrice','logBTCPrice','logETHPrice','logBTCVol','logTxFees','logCostperTxn','logNoTxns','logAvgBlkSz','logUniqueAddresses','logHashRate','logCrypto Market Cap','logNasdaq','logGold','logInterest','Interest','TxFees','Nasdaq']]
df_all = df
df_hist = df
df.head(2)

df.tail(2)

# evaluate the correlation of the majority of features
df_all.corr().sort_values('logBTCPrice')

# sns.set(style="darkgrid", color_codes=True)
# sns.pairplot(df_all)
# plt.savefig('charts/pairplotfeatureuniverse.png')

# Reducing to few key features
df_all = df_all[['logBTCPrice','logNasdaq','logInterest','logTxFees']]
df_all.corr().sort_values('logBTCPrice')



sns.set(style="darkgrid", color_codes=True)
sns.pairplot(df_all,plot_kws={'alpha':0.3})
# plt.title('Correlation by Feature',fontsize=14)
plt.savefig('charts/modelpairplot.png')

df = df_all
# STATSMODELS
# Feature matrix (X) and target vector (y)
y, X = patsy.dmatrices('logBTCPrice ~ logInterest + logNasdaq + logTxFees', data=df, return_type="dataframe")

model = sm.OLS(y,X)
fit = model.fit()
fit.summary()

# SKLEARN
lr = LinearRegression()

# Choose the predictor variables, here all but the first which is the response variable
# This model is analogous to the Y ~ X1 + X3 + X6 model
X = df[['logInterest','logNasdaq','logTxFees']]
# Choose the response variable(s)
y = df['logBTCPrice']

lr.fit(X,y)
# Print out the R^2 for the model against the full dataset
print(lr.score(X,y))
print(lr.intercept_)
print(lr.coef_)

# Plotting residuals on a time series basis.
# Note that residuals are not random and will require further adjustments at a later time.  
fit.resid.plot(style='o');
plt.ylabel("Residual",fontsize=12)
plt.xlabel("Time",fontsize=12)
plt.title('Residual Over Time',fontsize=14)
plt.savefig('charts/residovertime.png')

y_pred = lr.predict(X)

residuals = y - y_pred

sns.distplot(residuals);
plt.ylabel("Frequency",fontsize=12)
plt.xlabel("Residual",fontsize=12)
plt.title('Residual Histogram',fontsize=14)
plt.savefig('charts/residhist.png')

# TIME SERIES CROSS VALIDATION NOT INCLUDED IN THIS ANALYSIS (But may be incorporated later)
# The cross validation for a time series would be as follows:
# split = int(round(len(df) * 0.9,0))
# X_train, X_test, y_train, y_test = df.iloc[:split,1:], df.iloc[split:,1:],df.iloc[:split,0],df.iloc[split:,0]
# Due to project requirements, I am using the standard linear regression train-test split for this analysis.

from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, train_size = 0.7)

# Fit the model against the training data
lr.fit(X_train, y_train)
# # Evaluate the model against the testing data
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))



# sklearn prediction (logBTCPrice); seaborn
ax = sns.regplot(x=y_test,y=lr.predict(X_test), data=df)
ax.set(xlabel='Actual (log)', ylabel='Predicted (log)', title = 'Bitcoin Price: Predicted vs. Actual (log)')
plt.savefig('charts/logpredictedvsactual.png')

# sklearn prediction (BTCPrice); seaborn
ax = sns.regplot(x=np.exp(y_test),y=np.exp(lr.predict(X_test)), data=df)
ax.set(xlabel='Actual, US$', ylabel='Predicted, US$', title = 'Bitcoin Price: Predicted vs. Actual')
plt.savefig('charts/predictedvsactual.png')

# limit x and y ticks to 8000 to see model prediction capability vs actual is near-aligned
ax = sns.regplot(x=np.exp(y_test),y=np.exp(lr.predict(X_test)), data=df)
ax.set(xlabel='Actual, US$', ylabel='Predicted, US$', title = 'Bitcoin Price: Predicted vs. Actual')
plt.xlim(0, 8000)
plt.ylim(0, 8000)
plt.savefig('charts/predictedvsactuallimits.png')

# GOOGLE SEARCH INTEREST
x = df['logBTCPrice']
y = df['logInterest']

ax = sns.regplot(x,y, data=df)
ax.set(xlabel='log Bitcoin Price', ylabel='log Google Search Interest', title = 'Bitcoin Price vs. Google Search Interest')
plt.savefig('charts/interestvsbtcprice.png')

y1 = pd.Series(df_hist['BTCPrice'])
y2 = pd.Series(df_hist['Interest'])
x = pd.Series(df_hist.index.values)

fig, _ = plt.subplots()

ax = plt.gca()
ax2 = ax.twinx()

ax.plot(x,y1,'b')
ax2.plot(x,y2,'g')
ax.set_ylabel("Price US$",color='b',fontsize=12)
ax2.set_ylabel("Google Search Interest",color='g',fontsize=12)
ax.grid(True)
plt.title("Bitcoin Price vs. Google Search Interest", fontsize=14)
ax.set_xlabel('Date', fontsize=12)
fig.autofmt_xdate()

plt.savefig('charts/googlesearchinterest.png')
print(plt.show())

y, X = patsy.dmatrices('logBTCPrice ~ logInterest', data=df, return_type="dataframe")

# Create your model
model2 = sm.OLS(y,X)
# Fit your model to your training set
fit2 = model2.fit()
# Print summary statistics of the model's performance
fit2.summary()

# FEATURE ANALYSIS: NASDAQ COMPOSITE INDEX
x = df['logBTCPrice']
y = df['logNasdaq']

ax = sns.regplot(x,y, data=df)
ax.set(xlabel='log Bitcoin Price', ylabel='log Nasdaq Composite Index', title = 'Bitcoin Price vs. Nasdaq')
plt.savefig('charts/nasdaqvsbtcprice.png')

y1 = pd.Series(df_hist['BTCPrice'])
y2 = pd.Series(df_hist['Nasdaq'])
x = pd.Series(df_hist.index.values)

fig, _ = plt.subplots()

ax = plt.gca()
ax2 = ax.twinx()

ax.plot(x,y1,'b')
ax2.plot(x,y2,'g')
ax.set_ylabel("Price US$",color='b',fontsize=12)
ax2.set_ylabel("Nasdaq Composite Index",color='g',fontsize=12)
# ax.grid(True)
plt.title("Bitcoin Price vs. Nasdaq Composite Index", fontsize=14)
ax.set_xlabel('Date', fontsize=12)
fig.autofmt_xdate()
# ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

plt.savefig('charts/nasdaq.png')
print(plt.show())

y, X = patsy.dmatrices('logBTCPrice ~ logNasdaq', data=df, return_type="dataframe")

# Create your model
model3 = sm.OLS(y,X)
# Fit your model to your training set
fit3 = model3.fit()
# Print summary statistics of the model's performance
fit3.summary()

# FEATURE ANALYSIS: TRANSACTION FEES
x = df['logBTCPrice']
y = df['logTxFees']

ax = sns.regplot(x,y, data=df)
ax.set(xlabel='log Bitcoin Price', ylabel='log Network Transaction Fees', title = 'Bitcoin Price vs. Network Transaction Fees')
plt.savefig('charts/txfeesvsbtcprice.png')

y1 = pd.Series(df_hist['BTCPrice'])
y2 = pd.Series(df_hist['TxFees'])
x = pd.Series(df_hist.index.values)

fig, _ = plt.subplots()

ax = plt.gca()
ax2 = ax.twinx()

ax.plot(x,y1,'b')
ax2.plot(x,y2,'g')
ax.set_ylabel("Price US$",color='b',fontsize=12)
ax2.set_ylabel("Network Transaction Fees",color='g',fontsize=12)
# ax.grid(True)
plt.title("Bitcoin Price vs. Network Transaction Fees", fontsize=14)
ax.set_xlabel('Date', fontsize=12)
fig.autofmt_xdate()
# ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

plt.savefig('charts/txfees.png')
print(plt.show())

y, X = patsy.dmatrices('logBTCPrice ~ logTxFees', data=df, return_type="dataframe")

# Create your model
model4 = sm.OLS(y,X)
# Fit your model to your training set
fit4 = model4.fit()
# Print summary statistics of the model's performance
fit4.summary()

# REGULARIZATION
from sklearn.linear_model import RidgeCV

rcv = RidgeCV(cv=10)

rcv.fit(X_train, y_train)
print(rcv.score(X_train,y_train))
rcv.score(X_test, y_test)













