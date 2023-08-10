import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import statsmodels.api as sm 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf

stock = pd.read_csv('TESLAHistoricalQuotes.csv')
stock = stock.drop(0,0)

# Fix up the date column
stock.date = pd.to_datetime(stock.date)
stock = stock.sort_values('date')

stock.set_index(stock.date, inplace=True)
stock = stock.drop('date', 1)

stock.head()

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    # This can be fine tuned.  There is some relationship, but hard to quantify.
    rolmean = pd.rolling_mean(timeseries, window=5)
    rolstd = pd.rolling_std(timeseries, window=5)

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

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(stock.close, lags=366, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(stock.close, lags=366, ax=ax2)
plt.savefig('corr_plot.png', bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(12,8))
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(stock.close, lags=81, ax=ax2)
plt.show()

stock['first_difference'] = stock.close - stock.close.shift(1)  
test_stationarity(stock.first_difference.dropna(inplace=False))

stock['season_80'] = stock.close - stock.close.shift(80)  
test_stationarity(stock.season_80.dropna(inplace=False))

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(stock.season_80.iloc[80:], lags=366, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(stock.season_80.iloc[80:], lags=366, ax=ax2)

plt.show()

stock['season_16'] = stock.close - stock.close.shift(16)  
test_stationarity(stock.season_16.dropna(inplace=False))

stock['season_16_diff'] = stock.first_difference - stock.first_difference.shift(16)  
test_stationarity(stock.season_16_diff.dropna(inplace=False))

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(stock.season_16_diff.iloc[18:], lags=50, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(stock.season_16_diff.iloc[18:], lags=50, ax=ax2)
plt.show()

mod_80 = sm.tsa.statespace.SARIMAX(stock.close, trend='n', order=(1,1,5), seasonal_order=(0,1,1,80))
results_80 = mod_80.fit()
results_80.summary()

mod_16 = sm.tsa.statespace.SARIMAX(stock.close, trend='n', order=(1,1,5), seasonal_order=(0,1,3,16))
results_16 = mod_16.fit()
results_16.summary()

mod = sm.tsa.statespace.SARIMAX(stock.close, trend='n', order=(14,1,0), seasonal_order=(0,0,0,0))
results = mod.fit()
results.summary()

mod = sm.tsa.statespace.SARIMAX(stock.close, trend='n', order=(3,1,30), seasonal_order=(0,0,0,0))
results = mod.fit()
results.summary()

mod = sm.tsa.statespace.SARIMAX(stock.close, trend='n', order=(1,1,3), seasonal_order=(0,1,1,80))
results = mod.fit()
results.summary()

# cut out 2017 data
1866-226

stock['forecast'] = results.predict(dynamic=False)
stock[['close', 'forecast']].plot(figsize=(12, 8)) 
plt.show()


ax=stock[['close', 'forecast']].iloc[1640:].plot(figsize=(12, 8)) 
 
plt.title('2017 Actual verses Predictions', fontsize=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Stock Price', fontsize=16)
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick) 
plt.yticks()
plt.xticks(rotation=25)

plt.savefig('forecast.png', bbox_inches='tight')
plt.show()

stock['next_day_open'] = stock.open.shift(-1)
stock['target'] = stock.close.shift(-1)
stock['predict_grow'] = stock[['forecast', 'next_day_open']].apply(lambda x: 1 if x[0] - x[1] >= 0 else 0, axis=1)
stock['true_grow'] = stock[['target', 'next_day_open']].apply(lambda x: 1 if x[0] - x[1] >= 0 else 0, axis=1)

stock.head(1)

def plot_profit(dataframe, predict_col, early_stop=None):  
    X = []
    y = []
    money_counter = 0
    transactions = 0
    own = False
    last_buy = 0
    current_price = 0
    for num, i in enumerate(dataframe.iloc[:early_stop].iterrows()):
        if i[1][predict_col] == 1 and own == False:
            money_counter -= i[1]['next_day_open']
            own = True
            transactions += 1
            last_buy = i[1]['next_day_open']
            X.append(i[0])
            y.append(money_counter + i[1]['next_day_open'])
        elif i[1][predict_col] == 0 and own == True:
            money_counter += i[1]['next_day_open']
            own = False
            transactions += 1
            X.append(i[0])
            y.append(money_counter)
        else:
            pass
        current_price=i[1]['next_day_open']
    fig, ax = plt.subplots()
    ax=pd.DataFrame(data=y, index=X)[0].plot(figsize=(12,8), title= 'Profit over Time', fontsize=14, ax=ax)
    plt.title('Profit over Time', fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Stock Price', fontsize=16)
    fmt = '${x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick) 
    plt.yticks()
    plt.xticks(rotation=25)
    plt.savefig('profit_plot.png', bbox_inches='tight')
    plt.show()
    print('Own?', own)
    print('Last Buying price: $', last_buy)
    print('Current price: $', current_price)
    print('Cash? $', money_counter)
    if own == True:
        print('Profit: $', current_price + money_counter)
    else:
        print('Profit: $', money_counter)
    print('Number of Transactins:', transactions)
    print('Cost of transactions: $', transactions*5)
    return X, y

a,b = plot_profit(stock.iloc[1640:], 'predict_grow', -1)

from sklearn import metrics

plt.scatter(stock.iloc[1:-1].target, stock.iloc[1:-1].forecast)
plt.xlabel("True Values")
plt.ylabel("Predictions")
print("Score:", metrics.r2_score(stock.iloc[1:-1].target, stock.iloc[1:-1].forecast))
print("MSE:", metrics.mean_squared_error(stock.iloc[1:-1].target, stock.iloc[1:-1].forecast))
plt.savefig('pred_true.png', bbox_inches='tight')
plt.show()

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_auc_score
def eval_sklearn_model(y_true, predictions, model=None, X=None):
    """This function takes the true values for y and the predictions made by the model and prints out the confusion matrix along with Accuracy, Precision, and, if model and X provided, Roc_Auc Scores."""
    cnf_matrix = confusion_matrix(y_true, predictions)

    print('True Negative: ', cnf_matrix[0, 0], '| False Positive: ', cnf_matrix[0, 1])
    print('False Negative: ', cnf_matrix[1, 0], '| True Positive: ', cnf_matrix[1, 1], '\n')

    sensitivity = cnf_matrix[1, 1]/ (cnf_matrix[1, 0] + cnf_matrix[1, 1])
    specificity = cnf_matrix[0, 0]/ (cnf_matrix[0, 1] + cnf_matrix[0, 0])

    print('Sensitivity (TP/ TP + FN): ', sensitivity)
    print('Specificity (TN/ TN + FP): ', specificity, '\n')

    print('Accuracy: ', accuracy_score(y_true, predictions, normalize=True))
    print('Precision: ', precision_score(y_true, predictions))
    if model != None:
        print('Roc-Auc: ', roc_auc_score(y_true, [x[1] for x in model.predict_proba(X)]))
    else:
        pass
    print('\n')

eval_sklearn_model(stock.true_grow, stock.predict_grow)



stock[['close', 'forecast']].iloc[1800:].plot(figsize=(12, 8)) 

plt.title('Closer Look: Actual verses Predictions', fontsize=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Stock Price', fontsize=16)
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick) 
plt.yticks()
plt.xticks(rotation=25)

plt.savefig('forecast.png', bbox_inches='tight')
plt.show()

stock['forecast'] = [np.NaN for i in range(1640)] + list(results.predict(start=1640, end=1865, dynamic=True))
stock[['close', 'forecast']].plot(figsize=(12, 8)) 
plt.show()

stock['forecast'] = [np.NaN for i in range(1840)] + list(results.predict(start=1840, end=1865, dynamic=False))
stock[['close', 'forecast']].iloc[1840:].plot(figsize=(12, 8)) 
plt.show()

#Basically I'm a day behind; always.

from statsmodels.tsa.arima_model import ARIMA

mod = ARIMA(stock.close, order=(3,1,3))
results = mod.fit()
results.summary()

# Well the AIC is lower.  No idea if it is significantly lower...



mod = sm.tsa.statespace.SARIMAX(stock.close, trend='n', order=(3,1,5), seasonal_order=(0,1,1,80))
results = mod.fit()
results.summary()

stock['forecast'] = results.predict(dynamic=False)
stock[['close', 'forecast']].plot(figsize=(12, 8)) 
plt.show()

stock['forecast'] = [np.NaN for i in range(1840)] + list(results.predict(start=1840, end=1865, dynamic=False))
stock[['close', 'forecast']].iloc[1840:].plot(figsize=(12, 8)) 
plt.show()

stock['forecast'] = [np.NaN for i in range(1640)] + list(results.predict(start=1640, end=1865, dynamic=True))
stock[['close', 'forecast']].plot(figsize=(12, 8)) 
plt.show()



