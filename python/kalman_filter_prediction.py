get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
pd.set_option('display.max_columns', 50)

adj_jpm = pd.DataFrame.from_csv('adjJPM.csv')
display(adj_jpm.head())

from os import listdir

# Get stock names
symbol_list = [filename[:-4] for filename in listdir('../Forecaster/stock_data/') if filename[-3:] == 'csv']

# Remove sp500.csv
symbol_list.remove('sp500')

# Load stock data
stock_dict = {}

for symbol in symbol_list:
    stock_dict[symbol] = pd.DataFrame.from_csv('../Forecaster/stock_data/' + symbol + '.csv')

from copy import deepcopy

def adjust_price_volume(df):
    
    df['adj_factor'] = df['Adj Close'] / df['Close']
    df['Open'] = df['Open'] * df['adj_factor']
    df['High'] = df['High'] * df['adj_factor']
    df['Low'] = df['Low'] * df['adj_factor']
    df['Volume'] = df['Volume'] / df['adj_factor']

def create_label_column(df):
    
    df['Adj Close shift'] = df['Adj Close'].shift(-1)

def create_next_day_open(df):
    
    df['Open shift'] = df['Open'].shift(-1)

def create_diffs(df):
    
    df['high_diff'] = df['High'] - df['Adj Close shift']
    df['low_diff'] = df['Low'] - df['Adj Close shift']
    df['close_diff'] = df['Adj Close'] - df['Adj Close shift']
    df['open_diff'] = df['Open shift'] - df['Adj Close shift']

def preprocess(dataframe):
    df = deepcopy(dataframe)
    
    adjust_price_volume(df)
    create_label_column(df)
    create_next_day_open(df)
    create_diffs(df)
    
    df.dropna(inplace=True)
    
    return df

# Adjust prices and volumes
for symbol in symbol_list:
    stock_dict[symbol] = preprocess(stock_dict[symbol])

stock_dict['JPM'].head()

from numpy.linalg import inv

class Kalman(object):
    def __init__(self, init_price, noise=1):
        self.dt = 1 # time scale
        self.noise = noise
        
        self.x = np.array([init_price, 0]) # State vector: [price, price_rate] (2x1)
#         self.x = np.array([init_price, 0, 0]) # State vector: [price, price_rate, price_rate2] (3x1)
        self.P = np.array([[1, 0], [0, 1]]) # Uncertainty covariance matrix (2x2)
#         self.P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # Uncertainty covariance matrix (3x3)
        
        self.F = np.array([[1, self.dt], [0, 1]]) # Prediction matrix (2x2)
#         self.F = np.array([[1, self.dt, 0.5 * self.dt**2], [0, 1, self.dt], [0, 0, 1]]) # Prediction matrix (2x2)
    
        self.Q = np.array([[noise, 0], [0, noise]]) # Unpredictable external factor noise covariance matrix (2x2)
#         self.Q = np.array([[noise, 0, 0], [0, noise, 0], [0, 0, noise]]) # Unpredictable external factor noice covariance matrix (3x3)
        
        self.H = np.array([1, 0]) # Measurement mapping function (1x2)
#         self.H = np.array([1, 0, 0]) # Measurement mapping function (1x3)
        
        self.R_h = None # Sensor noise covariance (scalar)
        self.R_l = None # Sensor noise covariance (scalar)
        self.R_c = None # Sensor noise covariance (scalar)
        self.R_o = None # Sensor noise covariance (scalar)   
        
        self.S = None # Fusion (scalar)
        
        self.y = None # error (scalar)
        self.K = None # Kalman gain (2x1)
#         self.K = None # Kalman gain (3x1)
        
    def predict(self):
        self.x = np.matmul(self.F, self.x) # Predict today's adj close
        self.P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.Q
    
    def update(self, measurement, sensor_type):
        self.y = measurement - np.matmul(self.H, self.x) # Calculate loss
        
        if sensor_type == 'high':
            self.S = np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R_h
        elif sensor_type == 'low':
            self.S = np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R_l
        elif sensor_type == 'close':
            self.S = np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R_c
        else:
            self.S = np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R_o
            
#         K = np.matmul(self.P, self.H.T) * (1/S) # Calculate Kalman gain (2x1)
        self.K = np.matmul(self.P, self.H.T) * (1/self.S) # Calculate Kalman gain (3x1)
        
        # Update x and P
        self.x = self.x + self.K * self.y
        self.P = np.matmul(np.eye(2, 2) - np.matmul(self.K, self.H), self.P)
#         self.P = np.matmul(np.eye(3, 3) - np.matmul(self.K, self.H), self.P)

kalman_dict = {} # Key: symbol, Value: (init_price, kalman object)

for symbol in symbol_list:
    init_price = stock_dict[symbol].ix[0, 0]
    kalman_dict[symbol] = init_price, Kalman(init_price)

prediction_dict = {}
symbol_count = 0

for symbol in symbol_list:
    symbol_count += 1
    
    stock = stock_dict[symbol]
    kalman = kalman_dict[symbol][1]
    
    prediction = []
    counter = 0

    for i, r in stock.iterrows():
        (u,) = stock.index.get_indexer_for([i])

        # Keep track of number of days
        counter += 1
        # Start from the 127th day of data
        if counter > 126:
            # Calculate variance
            high_var126 = stock.ix[u-126:u, 'high_diff'].std()**2
            low_var126 = stock.ix[u-126:u, 'low_diff'].std()**2
            close_var126 = stock.ix[u-126:u, 'close_diff'].std()**2
            open_var126 = stock.ix[u-126:u, 'open_diff'].std()**2

            kalman.R_h = high_var126
            kalman.R_l = low_var126 
            kalman.R_c = close_var126
            kalman.R_o = open_var126

            # Predict
            kalman.predict()

            # Measurement Update
            # Update using High
            kalman.update(measurement=r['High'], sensor_type='high')

            # Update using High
            kalman.update(measurement=r['Low'], sensor_type='high')

            # Update using High
            kalman.update(measurement=r['Adj Close'], sensor_type='close')

            # Update using High
            kalman.update(measurement=r['Open shift'], sensor_type='open')

            prediction.append(kalman.x[0])
            
    prediction_dict[symbol] = prediction
    
    print(symbol_count, symbol)

stock_dict['SUNE'].head()

from sklearn.metrics import r2_score

y_true_dict = {}
comparison_dict = {}
r2_dict = {}

for symbol in symbol_list:
    y_true_dict[symbol] = stock_dict[symbol].iloc[126:, 7]
    comparison_dict[symbol] = (stock_dict[symbol].iloc[126:, 5], prediction_dict[symbol])
    r2_dict[symbol] = (r2_score(y_true_dict[symbol], comparison_dict[symbol][0]), r2_score(y_true_dict[symbol], comparison_dict[symbol][1]))

# print(len(y_true_dict['SUNE']))
# print(len(comparison_dict['SUNE'][0]))

# print(y_true_dict['SUNE'][:10])
# print(comparison_dict['SUNE'][0][:10])
# print(r2_score(y_true_dict['SUNE'], comparison_dict['SUNE'][0]))
# print(len(comparison_dict['SUNE'][1]))

print(r2_dict['SUNE'])

naive_series = pd.Series([r2_dict[symbol][0] for symbol in r2_dict])
naive_avg = np.mean(naive_series)
naive_std = np.std(naive_series)

kalman_series = pd.Series([r2_dict[symbol][1] for symbol in r2_dict])
# kalman_avg = np.mean(kalman_series)
# kalman_std = np.std(kalman_series)

bad_symbols = [symbol for symbol in r2_dict if r2_dict[symbol][1] < 0.95]
bad_symbols.extend([symbol for symbol in r2_dict if r2_dict[symbol][0] < 0.95])

good_symbols = [symbol for symbol in r2_dict if symbol not in bad_symbols][:200]

print(bad_symbols)

naive_list = [r2_dict[symbol][0] for symbol in good_symbols]
kalman_list = [r2_dict[symbol][1] for symbol in good_symbols]
naive_series = pd.Series(naive_list)
kalman_series = pd.Series(kalman_list)

# print([symbol for symbol in r2_dict if r2_dict[symbol][0] < 0.9])

naive_series.describe()

kalman_series.describe()

test_sym = 'GHC'

pd.Series(prediction_dict[test_sym]).describe()

prediction_dict[test_sym]

display(stock_dict[test_sym].head(100))

stock_dict[test_sym]['Adj'].plot()

for i, symbol in enumerate(symbol_list):
#     test_sym = 'AMAT'
# pd.Series(prediction_dict['ZBH']).describe()
    print(i, r2_score(y_true_dict[symbol], prediction_dict[symbol]))



pd.options.display.max_rows = 5000

display(kalman_series)

print("AVG: {0} {1}".format(naive_avg, kalman_avg))
print("STD: {0} {1}".format(naive_std, kalman_std))



adj_jpm['Adj Close shift'] = adj_jpm['Adj Close'].shift(-1)
adj_jpm['Open shift'] = adj_jpm['Open'].shift(-1)

adj_jpm.head()

adj_jpm['high_diff'] = adj_jpm['High'] - adj_jpm['Adj Close shift']
adj_jpm['low_diff'] = adj_jpm['Low'] - adj_jpm['Adj Close shift']
adj_jpm['close_diff'] = adj_jpm['Adj Close'] - adj_jpm['Adj Close shift']
adj_jpm['open_diff'] = adj_jpm['Open shift'] - adj_jpm['Adj Close shift']

adj_jpm.head()

adj_jpm.tail()

adj_jpm.dropna(inplace=True)

adj_jpm.shape

high_var126 = adj_jpm.ix[:125, 'high_diff'].std()**2
low_var126 = adj_jpm.ix[:125, 'low_diff'].std()**2
close_var126 = adj_jpm.ix[:125, 'close_diff'].std()**2
open_var126 = adj_jpm.ix[:125, 'open_diff'].std()**2

print(high_var126)
print(low_var126)
print(close_var126)
print(open_var126)

init_price = adj_jpm.ix[0, 0]

# kalman = Kalman(init_price, [high_var, low_var, close_var, open_var])
kalman = Kalman(init_price)

prediction = []
counter = 0

for i, r in adj_jpm.iterrows():
    (u,) = adj_jpm.index.get_indexer_for([i])

    # Keep track of number of days
    counter += 1
    # Start from the 127th day of data
    if counter > 126:
        # Calculate variance
        high_var126 = adj_jpm.ix[u-126:u, 'high_diff'].std()**2
        low_var126 = adj_jpm.ix[u-126:u, 'low_diff'].std()**2
        close_var126 = adj_jpm.ix[u-126:u, 'close_diff'].std()**2
        open_var126 = adj_jpm.ix[u-126:u, 'open_diff'].std()**2
        
        kalman.R_h = high_var126
        kalman.R_l = low_var126 
        kalman.R_c = close_var126
        kalman.R_o = open_var126
        
        # Predict
        kalman.predict()

        # Measurement Update
        # Update using High
        kalman.update(measurement=r['High'], sensor_type='high')

        # Update using High
        kalman.update(measurement=r['Low'], sensor_type='high')

        # Update using High
        kalman.update(measurement=r['Adj Close'], sensor_type='close')

        # Update using High
        kalman.update(measurement=r['Open shift'], sensor_type='open')

        prediction.append(kalman.x[0])
        
        if u >= 206 + 126 and u < 217 + 126:
            print("u, i: {}".format((u, i)))
            print("x: {}".format(kalman.x))
            print("P: {}".format(kalman.P))
            print("y: {}".format(kalman.y))
            print("S: {}".format(kalman.S))
            print("K: {}".format(kalman.K))
            print('\n')

from copy import copy

adj_jpm_sub = copy(adj_jpm.iloc[126:])

len(adj_jpm_sub)

adj_jpm_sub.ix[:, 'kalman_pred'] = prediction

adj_jpm_sub.iloc[200:300]

from datetime import datetime

adj_jpm_sub.index.get_indexer_for([datetime(1985, 4, 25)])

adj_jpm_sub.isnull().sum()

from sklearn.metrics import r2_score

y_true = adj_jpm_sub['Adj Close shift']

naive = r2_score(y_true, adj_jpm_sub['Adj Close'])
kalman_pred = r2_score(y_true, prediction)

print(naive)
print(kalman_pred)

naive_df = pd.DataFrame(adj_jpm_sub['Adj Close'])
naive_df['Adj_Close'] = adj_jpm_sub['Adj Close shift']

naive_df.columns = ['Naive', 'Adj_Close']

naive_df.head()

ax = naive_df.ix[0:252, :].plot(figsize=(12, 6), title='JPM')
ax.set_xlabel("Date")
ax.set_ylabel("Adj Close")
plt.show()

kalman_df = pd.DataFrame(adj_jpm_sub['kalman_pred'])
kalman_df['Adj_Close'] = adj_jpm_sub['Adj Close shift']
kalman_df.columns = ['Kalman', 'Adj_Close']

ax = kalman_df.ix[0:252, :].plot(figsize=(12, 6), title='JPM')
ax.set_xlabel("Date")
ax.set_ylabel("Adj Close")
plt.show()

kalman_df.head()

kalman_df['Open'] = adj_jpm_sub['Open']

kalman_df['Open+1'] = kalman_df['Open'].shift(-1)

kalman_df.head()

class TradingInu(object):
    def __init__(self, init_cash=1000):
        
        self.cash = init_cash
        self.share = 0
        self.total_asset = [(None, self.cash, None, None, None)]
        
    def buy(self, price, trading_cost=None):
        
        affordable_share = self.cash // price
        self.cash -= price * affordable_share
        self.share += affordable_share
        
        self.apply_trading_cost(affordable_share, trading_cost)
        
        self.total_asset.append(('buy', price, self.cash, self.share, self.cash + self.share * price))
        
    def sell(self, price, trading_cost=None):
        self.cash += self.share * price
        self.apply_trading_cost(self.share, trading_cost)
        
        self.share = 0
        
        self.total_asset.append(('sell', price, self.cash, self.share, self.cash + self.share * price))
    
    def apply_trading_cost(self, trading_amount, trading_cost):        
        if trading_cost is not None:
            if trading_amount * 0.01 < 1.99:
                self.cash = self.cash - 1.99
            else:
                self.cash = self.cash - trading_amount * 0.01
        

kalman_df['kalman+1'] = kalman_df['Kalman'].shift(-1)
del kalman_df['kalman + 1']

kalman_df['Adj_Close+1'] = kalman_df['Adj_Close'].shift(-1)

kalman_df.head()

inu = TradingInu()

for i, row in kalman_df.iterrows():
    if row['kalman+1'] >= row['Open+1']:
        inu.buy(row['Open+1'], None)
    else:
        inu.sell(row['Open+1'], None)

from datetime import datetime

inu.total_asset[:50]

pd.Series([x[4] for x in inu.total_asset][:]).plot()

kalman_df['Adj_Close'][:datetime(1994, 8, 4)].plot()

adj_jpm_sub.head()



