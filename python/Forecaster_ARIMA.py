from __future__ import division

# Data Visualization
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
from IPython.display import display

# Set global random seeds
np.random.seed(0)
random.seed(0)

def load_data(stock_list_csv, market_csv, start_date, end_date, symbol_list=None, random_state=0, number_of_stocks=1, include_stocks=None):
    """The function does the following:
    1. Load the list of stocks
    2. Load the market data (SPY or other ETF data)
    3. Load the economy data (STLFSI or other index data)
    4. Randomly picks 10 stocks and load into a dictionary of dataframes if symbol_list is not provided,
       otherwise load the stocks in the symbo_list instead
    5. Return a dictionary of dataframes of stock data, with stock symbols as the keys
    """
    start_time = time.time()
    
    # Set up the empty main dataframe using the defined data range
    date_range = pd.date_range(start_date, end_date)
    df_main = pd.DataFrame(index=date_range)
    
    # Load SPY to get trading days
    dfSPY = pd.read_csv(market_csv, index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close', 'Volume'], na_values = ['nan'])
    dfSPY = dfSPY.rename(columns={'Adj Close': 'SPY', 'Volume': 'SPY_Vol'})
    
    # Get SPY within the target date range
    df_main = df_main.join(dfSPY)
    
    # Drop NaN values
    df_main = df_main.dropna()
    
    # Load target stock list
    dfSPY500_2009 = pd.read_csv(stock_list_csv, header=None, usecols = [1])
    
    # Ready the symbol list
    if symbol_list is None:
#         np.random.seed(random_state)
        symbol_list = np.random.choice(dfSPY500_2009[1].tolist(), size=number_of_stocks, replace=False).tolist()
    
    if include_stocks is not None:
        try:
            symbol_list.extend(include_stocks)
            symbol_set = set(symbol_list)
            symbol_list = list(symbol_set)
        except TypeError:
            print("The stocks to be included should be put in a list.")

    
    # Load the FSI data
    dfFSI = pd.read_csv('STLFSI.csv', index_col='DATE', parse_dates=True, na_values = ['nan'])
    
    # Load target stocks
    result_dict = {}
    
    for symbol in symbol_list:
        if symbol != 'SPY':
            df_temp = pd.read_csv('stock_data/' + symbol + '.csv', index_col="Date", parse_dates=True, usecols = ['Date', 'Volume', 'Adj Close'], na_values=['nan'])
            df_temp = df_temp.rename(columns={'Volume': symbol + '_Vol', 'Adj Close': symbol})
            df_temp2 = df_main.join(df_temp, how='left')
            df_temp2 = df_temp2.join(dfFSI, how='left')
            result_dict[symbol] = {'original_data': df_temp2}
    
    print("{} seconds elapsed for loading data.".format(time.time() - start_time))
    print("\n")
    return result_dict

# Set stock
symbol = 'GOOGL'

# Set date range
start_date = '2009-01-01'
end_date = '2016-06-30'

# Load stock dataframes into dictionary
stock_dict = load_data('sp500_2009.csv', 'stock_data/SPY.csv', start_date, end_date, number_of_stocks=0, include_stocks=[symbol])

# Get target stock names
print("Stocks included:")
print(stock_dict.keys())

# Set example stock dataframe
df_main = stock_dict[symbol]['original_data']

# First glance at the data
display(df_main.head())

# Inspect missing values
print("Inspect missing values:")
print(df_main.isnull().sum())

# Inspect the stock price column
n_days = df_main.shape[0]
min_price = df_main[symbol].min()
max_price = df_main[symbol].max()
mean_price = df_main[symbol].mean()
median_price = np.median(df_main[symbol])
std_price = df_main[symbol].std()
cv = std_price / mean_price

print("Number of traded days: {}".format(n_days))
print("Minimum stock price: {}".format(min_price))
print("Maximum stock price: {}".format(max_price))
print("Mean stock price: {}".format(mean_price))
print("Median stock price: {}".format(median_price))
print("Standard deviation of stock price: {}".format(std_price))
print("Coefficient of variation of stock price: {}".format(cv))

# Inspect all columns
display(df_main.describe())

# Inspect stock trend
ax = df_main[symbol].plot(title=symbol, figsize=(12, 6))
ax.set_xlabel("Date")
ax.set_ylabel("Adj Close")

from pandas.tools.plotting import scatter_matrix

# Save time during Dev

scatter_matrix(df_main, alpha=0.3, figsize=(15, 15), diagonal='kde')


# g = sns.PairGrid(df_main)
# g.map_upper(plt.scatter, alpha=0.3)
# g.map_lower(plt.scatter, alpha=0.3)
# g.map_diag(sns.kdeplot, lw=3, legend=False)

# plt.subplots_adjust(top=0.95)
# g.fig.suptitle('Feature Pair Grid')

### Forward/Back Fill missing data
def fill_missing_data(stock_dict):
    for stock_name in stock_dict:
        stock_dict[stock_name]['original_data'].fillna(method='ffill', inplace=True)
        stock_dict[stock_name]['original_data'].fillna(method='bfill', inplace=True)

    return stock_dict

stock_dict = fill_missing_data(stock_dict)

# Update the main stock
df_main = stock_dict[symbol]['original_data']

## Display
# df_main[symbol].plot()
display(df_main.head(10))
print(df_main.isnull().sum())

## Split and create label data
def split_data(stock_dict):
    for stock_name in stock_dict:
        stock_dict[stock_name]['original_data_y'] = stock_dict[stock_name]['original_data'][stock_name]

    return stock_dict

## Apply n-day shift to data
def create_n_day_forecast_data(df, symbol, day):
    df = df.shift(-day)
    return df

## Add label_name key to dictionary for the feature engineering stage
def add_label_name_to_stock_dict(stock_dict, window):
    for stock_name in stock_dict:
        stock_dict[stock_name]['label_name'] = stock_name + str(window) + 'd'
    return stock_dict

## Assign the generated labels to 'data_y'
def create_label_data(stock_dict, window):
    for stock_name in stock_dict:
        stock_dict[stock_name]['data_y'] = create_n_day_forecast_data(stock_dict[stock_name]['original_data_y'], stock_name, window)
    return stock_dict

def ready_label_data(stock_dict, window):
    stock_dict = split_data(stock_dict)
    stock_dict = add_label_name_to_stock_dict(stock_dict, window)
    stock_dict = create_label_data(stock_dict, window)
    
    return stock_dict

# Ready label data
window = 5
stock_dict = ready_label_data(stock_dict, window)

for stock_name in stock_dict:
    stock_dict[stock_name]['data_y'].dropna(inplace=True)

label_name = stock_dict[symbol]['label_name']
original_data_y = stock_dict[symbol]['original_data_y']
data_y = stock_dict[symbol]['data_y']

# Signed-log transform
def sign_log(df):
    return np.sign(df) * np.log(abs(df) + 1)
    
log_start_time = time.time()
for stock_name in stock_dict:
    stock_dict[stock_name]['log_data'] = sign_log(stock_dict[stock_name]['original_data'])
#     stock_dict[stock_name]['log_data'] = np.sign(stock_dict[stock_name]['original_data']) * np.log(abs(stock_dict[stock_name]['original_data']) + 1)

log_data = stock_dict[symbol]['log_data']
print("{} seconds elapsed for log transforming data.".format(time.time() - log_start_time))

# Save time during Dev
g = sns.PairGrid(log_data)
g.map_upper(plt.scatter, alpha=0.2)
g.map_lower(plt.scatter, alpha=0.2)
g.map_diag(sns.kdeplot, lw=3, legend=False)

plt.subplots_adjust(top=0.95)
g.fig.suptitle('Log-transformed Pair Grid')

### Feature Engineering Section
### Make Daily Return Columns
def compute_daily_returns(df, adj_close_name):
    return (df / df.shift(1) - 1)[adj_close_name]

def make_daily_return_column(symbol, df):
    symbols = ['SPY']
    symbols.append(symbol)

    for symbol in symbols:
        df[symbol + '_return'] = compute_daily_returns(df, symbol)

    return df

### Make Beta columns (63 days)
def make_mean_std_columns(symbol, df):
    symbols = ['SPY']
    symbols.append(symbol)
    
    mean_dict = {}
    std_dict = {}

    for symbol in symbols:
        mean_dict[symbol] = []
        std_dict[symbol] = []

        for i in df.index:
            (u,) = df.index.get_indexer_for([i])
            if u - 63 >= 0:
                mean_dict[symbol].append(df[symbol + '_return'].iloc[u - 62:u+1].mean())
                std_dict[symbol].append(df[symbol + '_return'].iloc[u - 62:u+1].std())
            else:
                mean_dict[symbol].append(np.nan)
                std_dict[symbol].append(np.nan)

        df[symbol + '_Mean63d'] = mean_dict[symbol]
        df[symbol + '_Std63d'] = std_dict[symbol]

    return df

def make_beta_column(symbol, df):
    symbols = ['SPY']
    symbols.append(symbol)
    cov_dict = {}

    for symbol in symbols:
        cov_dict[symbol] = []
        for i in df.index:
            (u,) = df.index.get_indexer_for([i])
            if u - 62 >= 0:
                cov_dict[symbol].append(df['SPY_return'].iloc[(u - 62):u+1].cov(df[symbol + '_return'].iloc[(u - 62):u+1]))
            else:
                cov_dict[symbol].append(np.nan)
        df[symbol + '_Cov63d'] = cov_dict[symbol]
        df[symbol + '_Beta'] = df[symbol + '_Cov63d'] / df[symbol + '_Std63d']**2

    return df

### Make EMA column (100 days)
def make_ema_column(symbol, df):
    symbols = ['SPY']
    symbols.append(symbol)

    EMA_dict = {}
    alpha = 2 / (100 + 1)

    for symbol in symbols:
        EMA_dict[symbol] = []
        EMA_dict[symbol].append(df[symbol].iloc[0])

        for i in df.index[1:]:
            (u,) = df.index.get_indexer_for([i])
            EMA_dict[symbol].append(EMA_dict[symbol][u - 1] + alpha * (df[symbol].iloc[u] - EMA_dict[symbol][u - 1]))

        df[symbol + '_EMA'] = df[symbol]
    
    return df

### Make MMA column (100 days)
def make_mma_column(symbol, df):
    symbols = ['SPY']
    symbols.append(symbol)
    
    MMA_dict = {}
    alpha = 1 / 100

    for symbol in symbols:
        MMA_dict[symbol] = []
        MMA_dict[symbol].append(df[symbol].iloc[0])

        for i in df.index[1:]:
            (u,) = df.index.get_indexer_for([i])
            MMA_dict[symbol].append(MMA_dict[symbol][u - 1] + alpha * (df[symbol].iloc[u] - MMA_dict[symbol][u - 1]))

        df[symbol + '_MMA'] = MMA_dict[symbol]

    return df    

# ### Make SMA column (100 days)
def make_sma_column(symbol, df):
    symbols = ['SPY']
    symbols.append(symbol)
    
    for symbol in symbols:
            df[symbol + '_SMA'] = df[symbol].rolling(window=101, center=False).mean()

    return df

### SMA Momentum
def compute_SMA_momentum(df, SMA_column):
    return (df - df.shift(1))[SMA_column]*(100 + 1)

def make_sma_momentum_column(symbol, df):
    symbols = ['SPY']
    symbols.append(symbol)
    
    for symbol in symbols:
        df[symbol + '_SMA_Momentum'] = compute_SMA_momentum(df, symbol + '_SMA')

    return df

### Volume Momentum
def compute_Volume_momentum(df, Volume_column):
    return (df - df.shift(1))[Volume_column]*(100 + 1)

def make_vol_momentum_column(symbol, df):
    df[symbol + '_Vol_Momentum'] = compute_Volume_momentum(df, symbol + '_Vol')
    
    return df

### Vol_Momentum Marker 1
def make_vol_momentum_marker1_column(symbol, df):
    df[symbol + '_Vol_M1'] = np.nan
    df.loc[df[symbol + '_Vol_Momentum'] >= 0, symbol + '_Vol_M1'] = 1
    df.loc[df[symbol + '_Vol_Momentum'] < 0, symbol + '_Vol_M1'] = 0

    return df

### Vol_Momentum Marker 2
def make_vol_momentum_marker2_column(symbol, df):
    df[symbol + '_Vol_M2'] = np.nan
    df.loc[df[symbol + '_Vol'] >= (df[symbol + '_Vol'].rolling(window=101, center=False).mean() + df[symbol + '_Vol'].rolling(window=101, center=False).std()), symbol + '_Vol_M2'] = 1
    df.loc[df[symbol + '_Vol'] < (df[symbol + '_Vol'].rolling(window=101, center=False).mean() + df[symbol + '_Vol'].rolling(window=101, center=False).std()), symbol + '_Vol_M2'] = 0

    return df

### Vol_Momentum Marker 3
def make_vol_momentum_marker3_column(symbol, df):
    df[symbol + '_Vol_M3'] = np.nan
    df.loc[df[symbol + '_Vol'] < (df[symbol + '_Vol'].rolling(window=101, center=False).mean() - df[symbol + '_Vol'].rolling(window=101, center=False).std()), symbol + '_Vol_M3'] = 0
    df.loc[df[symbol + '_Vol'] >= (df[symbol + '_Vol'].rolling(window=101, center=False).mean() - df[symbol + '_Vol'].rolling(window=101, center=False).std()), symbol + '_Vol_M3'] = 1
    
    return df

### Make SR column
def make_SR_column(symbol, df):
    symbols = ['SPY']
    symbols.append(symbol)
    
    for symbol in symbols:
        df[symbol + '_SR63d'] = df[symbol + '_return'].rolling(window=63, center=False).mean() / df[symbol + '_Std63d']
    
    return df

### Drop not used SPY columns    
### Put back SPY keeper
def drop_keep_SPY_columns(symbol, df):
    ### Drop not used SPY columns    
    SPY_keeper = df[['SPY_SMA_Momentum', 'SPY_Std63d']]

    for column in df.columns:
        if 'SPY' in column:
            df.drop([column], axis=1, inplace=True)

    ### Put back SPY keeper
    df = pd.concat([df, SPY_keeper], axis=1)

    return df

### Drop not used columns
### Drop NaN rows
def drop_unused_nan_columns(symbol, df):
    ### Drop not used columns
    try:
#         df.drop([symbol, symbol + '_return', symbol + '_Mean63d', symbol + '_Cov63d', symbol + '_Vol'], axis=1, inplace=True)
        df.drop([symbol, symbol + '_return', symbol + '_Mean63d', symbol + '_Cov63d'], axis=1, inplace=True)
    except ValueError:
        print('OK, seems like these columns are already gone...')
    
    ### Drop NaN rows
    df.dropna(inplace=True)
        
    return df

def feature_engineer(symbol, df):    
    df = make_daily_return_column(symbol, df)
    df = make_mean_std_columns(symbol, df)
    df = make_beta_column(symbol, df)
    df = make_ema_column(symbol, df)
    df = make_mma_column(symbol, df)
    df = make_sma_column(symbol, df)
    df = make_sma_momentum_column(symbol, df)
    df = make_vol_momentum_column(symbol, df)
    df = make_vol_momentum_marker1_column(symbol, df)
    df = make_vol_momentum_marker2_column(symbol, df)
    df = make_vol_momentum_marker3_column(symbol, df)
    df = make_SR_column(symbol, df)
    df = drop_keep_SPY_columns(symbol, df)
    df = drop_unused_nan_columns(symbol, df)
    
    return df

feature_eng_start_time = time.time()

for stock_name in stock_dict:
    stock_dict[stock_name]['log_data'] = feature_engineer(stock_name, stock_dict[stock_name]['log_data'])
    stock_dict[stock_name]['data_X'] = stock_dict[stock_name]['log_data']
    stock_dict[stock_name]['data_y'] = stock_dict[stock_name]['data_y'].ix[stock_dict[stock_name]['data_X'].index]
    stock_dict[stock_name]['data_y'].dropna(inplace=True)
    stock_dict[stock_name]['data_X'] = stock_dict[stock_name]['data_X'].ix[stock_dict[stock_name]['data_y'].index]

print("{} seconds elapsed for feature engineering.".format(time.time() - feature_eng_start_time))

log_data = stock_dict[symbol]['log_data']
data_X = stock_dict[symbol]['data_X']
data_y = stock_dict[symbol]['data_y']

# Inspect the scatter matrix (full)
data_y_temp = data_y.copy().rename(columns=[label_name])
full_data = data_X.copy()
full_data[label_name] = np.sign(data_y_temp) * np.log(abs(data_y_temp) + 1)
display(full_data.head())

g = sns.PairGrid(full_data)
g.map_upper(plt.scatter, alpha=0.15)
g.map_lower(plt.scatter, alpha=0.15)
g.map_diag(sns.kdeplot, lw=3, legend=False)

plt.subplots_adjust(top=0.95)
g.fig.suptitle('Complete Feature Pair Grid', fontsize = 35)

# Supervised Learning
from sklearn import grid_search
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Find the n smallest elements in list
def more_min(arr, n_smallest=1):
    result_list = []

    if n_smallest > len(arr):
        raise ValueError("n_smallest greater than the length of the list!")
    else:
        arr_temp = list(arr)

        for i in range(n_smallest):
            result_list.append(np.min(arr_temp))
            arr_temp.remove(np.min(arr_temp))

        return result_list

# Find outliers
def get_outliers(arr):
    q75, q25 = np.percentile(arr, [75 ,25])
    iqr = q75 - q25
    
    thres = q25 - 1.5 * iqr
    print(thres)
    result = [u for u in arr if u < thres]
    
    return result

def make_predictions(data_X, data_y, date_range, estimator, window=105):
    pred_y_list = []
    drift_y_list = []
### Commented out code for monitoring feature importance of the RF
#     feature_importance_list = []
    
    for date in date_range:
        test_X = data_X.ix[date]
        (u,) = data_X.index.get_indexer_for([date])
        
        if u - window < 0:
            raise ValueError("Not enough training data!")
            
        train_X = data_X.iloc[(u - window):u - 5]
        train_y = data_y.iloc[(u - window):u - 5]

        estimator.fit(train_X, train_y)
        pred_y = estimator.predict(test_X.reshape(1, -1))
        pred_y_list.append(pred_y)
        
        drift_y = ((train_y.iloc[-1] - train_y.iloc[0]) / len(train_y)) * window * 1 + train_y.iloc[0]
        drift_y_list.append(drift_y)
        ## Create feature importance histogram
#         feature_importance_list.append(estimator.feature_importances_)
    
#     vif = [int(np.argmax(x)) for x in feature_importance_list]
#     vif = pd.DataFrame(vif)
#     d = np.diff(np.unique(vif)).min()
#     left_of_first_bin = vif.min() - float(d)/2
#     right_of_last_bin = vif.max() + float(d)/2
#     vif.plot.hist(alpha=0.5, bins=np.arange(left_of_first_bin, right_of_last_bin + d, d))
    return pred_y_list, drift_y_list

lr = RandomForestRegressor(criterion='mse', bootstrap=True)
reg = lr

r2_list = []
mse_list = []
min_list = []

# def train_and_report():
#     start_time = time.time()
    
#     for stock_name in stock_dict:
#         stock_dict[stock_name]['test_y'] = stock_dict[stock_name]['data_y'][121:]
#         stock_dict[stock_name]['pred_y'] = pd.Series(make_predictions(stock_dict[stock_name]['data_X'], stock_dict[stock_name]['data_y'], stock_dict[stock_name]['data_y'].index[121:], reg, 121))
#         stock_dict[stock_name]['pred_y'] = pd.DataFrame(stock_dict[stock_name]['pred_y'].tolist(), index=stock_dict[stock_name]['test_y'].index, columns=['Predict'])

#     for stock_name in stock_dict:
#         stock_dict[stock_name]['R2'] = r2_score(stock_dict[stock_name]['test_y'], stock_dict[stock_name]['pred_y'])
#         stock_dict[stock_name]['MSE'] = mean_squared_error(stock_dict[stock_name]['test_y'], stock_dict[stock_name]['pred_y'])
#         stock_dict[stock_name]['R2v2'] = r2_score(stock_dict[stock_name]['test_y'][datetime(2009, 12, 17):datetime(2016, 5, 27)], stock_dict[stock_name]['pred_y'][datetime(2009, 12, 17):datetime(2016, 5, 27)])
#         stock_dict[stock_name]['MSEv2'] = mean_squared_error(stock_dict[stock_name]['test_y'][datetime(2009, 12, 17):datetime(2016, 5, 27)], stock_dict[stock_name]['pred_y'][datetime(2009, 12, 17):datetime(2016, 5, 27)])

#     test_y = stock_dict[symbol]['test_y']
#     pred_y = stock_dict[symbol]['pred_y']

#     print(symbol)
#     print(reg.__class__.__name__)
#     print("R^2: {}".format(stock_dict[symbol]['R2']))
#     print("MSE: {}".format(stock_dict[symbol]['MSE']))
#     print("R^2 new range: {}".format(stock_dict[symbol]['R2v2']))
#     print("MSE new range: {}".format(stock_dict[symbol]['MSEv2']))

#     for stock_name in stock_dict:
#         stock_dict[stock_name]['dfResult'] = stock_dict[stock_name]['pred_y'].join(stock_dict[stock_name]['test_y'], )
#         stock_dict[stock_name]['dfResult'].columns = ['Predict', stock_dict[stock_name]['label_name']]

#     print(test_y.head())
#     print(pred_y.head())
#     dfResult2 = stock_dict[symbol]['dfResult']

#     # Stock price line chart
#     ax = dfResult2.plot(figsize=(12, 6), title=symbol)
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Adj Close")
    
#     ax = dfResult2.ix[datetime(2015, 7, 1):datetime(2015, 9, 1)].plot(figsize=(12, 6), title=symbol)
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Adj Close")
    
#     for stock_name in stock_dict:
#         stock_dict[stock_name]['diff'] = (stock_dict[stock_name]['dfResult']['Predict'] - stock_dict[stock_name]['dfResult'][stock_dict[stock_name]['label_name']]).to_frame()
#         stock_dict[stock_name]['diff'] = stock_dict[stock_name]['diff'].join(stock_dict[stock_name]['dfResult'][stock_dict[stock_name]['label_name']])
#         stock_dict[stock_name]['diff'].columns = ['err', stock_dict[stock_name]['label_name']]

#     diff2 = stock_dict[symbol]['diff']

#     diff2.plot.scatter(x=label_name, y='err', title='Residual Plot', figsize=(12, 6))
#     plt.axhline(y=diff2['err'].mean())
#     plt.show()

#     print("Err mean: {}".format(diff2['err'].mean()))
#     print("Err Std: {}".format(diff2['err'].std()))

#     for stock_name in stock_dict:
#         stock_dict[stock_name]['diff_percent'] = ((stock_dict[stock_name]['dfResult']['Predict'] - stock_dict[stock_name]['dfResult'][stock_dict[stock_name]['label_name']]) / stock_dict[stock_name]['dfResult'][stock_dict[stock_name]['label_name']]).to_frame()
#         stock_dict[stock_name]['diff_percent'] = stock_dict[stock_name]['diff_percent'].join(stock_dict[stock_name]['dfResult'][stock_dict[stock_name]['label_name']])
#         stock_dict[stock_name]['diff_percent'].columns = ['err percentage', stock_dict[stock_name]['label_name']]

#     diff3 = stock_dict[symbol]['diff_percent']    

#     diff3.plot.scatter(x=label_name, y='err percentage', title='Residual Plot (percentage)', figsize=(12, 6))
#     plt.axhline(y=diff3['err percentage'].mean())
#     plt.show()

#     print("Err percentage mean: {}%".format(diff3['err percentage'].mean() * 100))
#     print("Err percentage Std: {}%".format(diff3['err percentage'].std() * 100))
#     print("Err percentage 95% CI: {}".format((diff3['err percentage'].mean() - 2 * diff3['err percentage'].std(), diff3['err percentage'].mean() + 2 * diff3['err percentage'].std())))
#     print("\n")

#     # Get R2 and MSE list of all stocks
#     global r2_list
#     global mse_list
#     global min_list
    
#     r2_list = [stock_dict[x]['R2'] for x in stock_dict]
#     mse_list = [stock_dict[x]['MSE'] for x in stock_dict]
    
#     # Smallest R^2 list
#     min_list = more_min(r2_list, 1)
#     print("Smallest R^2:")
#     print(min_list)
#     print("\n")

#     # Get smallest R^2 symbols
#     min_id_list = [r2_list.index(x) for x in min_list]
#     min_symbol_list = [stock_dict.keys()[x] for x in min_id_list]
#     print("Smallest symbols:")
#     print(min_symbol_list)
    
#     # Outlier R^2 list
#     outlier_list = get_outliers(r2_list)
#     print("Outlier R^2:")
#     print(outlier_list)
#     print("\n")
    
#     # Get outlier R^2 symbols
#     outlier_id_list = [r2_list.index(x) for x in outlier_list]
#     outlier_symbol_list = [stock_dict.keys()[x] for x in outlier_id_list]
#     print("Outlier symbols:")
#     print(outlier_symbol_list)
    
#     print("{} seconds elapsed for training/predicting/reporting.".format(time.time() - start_time))
#     print("\n")

def report_multiple():
    print("R^2 (mean): {}%".format(np.mean(r2_list) * 100))
    print("R^2 (std): {}%".format(np.std(r2_list) * 100))
    print("R^2 (95% CI): {}".format((np.mean(r2_list) - 2 * np.std(r2_list), np.mean(r2_list) + 2 * np.std(r2_list))))

    # R2 Boxplot
    df_r2 = pd.DataFrame(r2_list)
    df_r2['Stock ID'] = df_r2.index
    df_r2.columns = ["R2", "Stock ID"]
    
    display(df_r2.describe())

    ax = sns.boxplot(y='R2', data=df_r2, orient='v')
    ax = sns.stripplot(y="R2", data=df_r2, jitter=0.05)

    #Evaluate the consistency of the residuals
    err_percent_mean_list = [stock_dict[x]['diff_percent']['err percentage'].mean() for x in stock_dict]
    err_percent_std_list = [stock_dict[x]['diff_percent']['err percentage'].std() for x in stock_dict]
    print("Err percentage mean (avg): {}".format(np.mean(err_percent_mean_list)))
    print("Err percentage Std (avg): {}".format(np.mean(err_percent_std_list)))
    print("Err percentage 95% CI (avg): {}".format((np.mean(err_percent_mean_list) - 2 * np.mean(err_percent_std_list), np.mean(err_percent_mean_list) + 2 * np.mean(err_percent_std_list))))
    
    # Err Mean Boxplot
    df_err = pd.DataFrame(err_percent_mean_list)
    df_err['Stock ID'] = df_err.index
    df_err.columns = ["Err Percentage Mean", "Stock ID"]

    display(df_err.describe())
    
    plt.figure()
    ax_err = sns.boxplot(y="Err Percentage Mean", data=df_err, orient='v', color='.45')
    ax_err = sns.stripplot(y="Err Percentage Mean", data=df_err, jitter=0.05)

    # Err Std Boxplot
    df_err_std = pd.DataFrame(err_percent_std_list)
    df_err_std['Stock ID'] = df_err_std.index
    df_err_std.columns = ["Err Percentage Std", "Stock ID"]

    display(df_err_std.describe())
    
    plt.figure()
    ax_err_std = sns.boxplot(y="Err Percentage Std", data=df_err_std, orient='v', color='.55')
    ax_err_std = sns.stripplot(y="Err Percentage Std", data=df_err_std, jitter=0.05)
    
# train_and_report()

start_time = time.time()
    
for stock_name in stock_dict:
    stock_dict[stock_name]['test_y'] = stock_dict[stock_name]['data_y'][121:]
    stock_dict[stock_name]['pred_y'] = pd.Series(make_predictions(stock_dict[stock_name]['data_X'], stock_dict[stock_name]['data_y'], stock_dict[stock_name]['data_y'].index[121:], reg, 121)[0])
    stock_dict[stock_name]['pred_y'] = pd.DataFrame(stock_dict[stock_name]['pred_y'].tolist(), index=stock_dict[stock_name]['test_y'].index, columns=['Predict'])
    stock_dict[stock_name]['drift_y'] = pd.Series(make_predictions(stock_dict[stock_name]['data_X'], stock_dict[stock_name]['data_y'], stock_dict[stock_name]['data_y'].index[121:], reg, 121)[1])
    stock_dict[stock_name]['drift_y'] = pd.DataFrame(stock_dict[stock_name]['drift_y'].tolist(), index=stock_dict[stock_name]['test_y'].index, columns=['Drift'])

for stock_name in stock_dict:
    stock_dict[stock_name]['R2'] = r2_score(stock_dict[stock_name]['test_y'], stock_dict[stock_name]['pred_y'])
    stock_dict[stock_name]['MSE'] = mean_squared_error(stock_dict[stock_name]['test_y'], stock_dict[stock_name]['pred_y'])
    stock_dict[stock_name]['R2v2'] = r2_score(stock_dict[stock_name]['test_y'][datetime(2009, 12, 17):datetime(2016, 5, 27)], stock_dict[stock_name]['pred_y'][datetime(2009, 12, 17):datetime(2016, 5, 27)])
    stock_dict[stock_name]['MSEv2'] = mean_squared_error(stock_dict[stock_name]['test_y'][datetime(2009, 12, 17):datetime(2016, 5, 27)], stock_dict[stock_name]['pred_y'][datetime(2009, 12, 17):datetime(2016, 5, 27)])

test_y = stock_dict[symbol]['test_y']
pred_y = stock_dict[symbol]['pred_y']
drift_y = stock_dict[symbol]['drift_y']
print("Creating test_y: {}".format(len(test_y)))
print("Creating pred_y: {}".format(len(pred_y)))
print("Creating pred_y: {}".format(len(drift_y)))
print(test_y)
print(pred_y)
print(drift_y)

print(symbol)
print(reg.__class__.__name__)
print("R^2: {}".format(stock_dict[symbol]['R2']))
print("MSE: {}".format(stock_dict[symbol]['MSE']))
print("R^2 new range: {}".format(stock_dict[symbol]['R2v2']))
print("MSE new range: {}".format(stock_dict[symbol]['MSEv2']))

for stock_name in stock_dict:
    stock_dict[stock_name]['dfResult'] = stock_dict[stock_name]['pred_y'].join(stock_dict[stock_name]['test_y'], )
    stock_dict[stock_name]['dfResult'].columns = ['Predict', stock_dict[stock_name]['label_name']]

print(test_y.head())
print(pred_y.head())
dfResult2 = stock_dict[symbol]['dfResult']

# Stock price line chart
ax = dfResult2.plot(figsize=(12, 6), title=symbol)
ax.set_xlabel("Date")
ax.set_ylabel("Adj Close")

ax = dfResult2.ix[datetime(2015, 7, 1):datetime(2015, 9, 1)].plot(figsize=(12, 6), title=symbol)
ax.set_xlabel("Date")
ax.set_ylabel("Adj Close")

for stock_name in stock_dict:
    stock_dict[stock_name]['diff'] = (stock_dict[stock_name]['dfResult']['Predict'] - stock_dict[stock_name]['dfResult'][stock_dict[stock_name]['label_name']]).to_frame()
    stock_dict[stock_name]['diff'] = stock_dict[stock_name]['diff'].join(stock_dict[stock_name]['dfResult'][stock_dict[stock_name]['label_name']])
    stock_dict[stock_name]['diff'].columns = ['err', stock_dict[stock_name]['label_name']]

diff2 = stock_dict[symbol]['diff']

diff2.plot.scatter(x=label_name, y='err', title='Residual Plot', figsize=(12, 6))
plt.axhline(y=diff2['err'].mean())
plt.show()

print("Err mean: {}".format(diff2['err'].mean()))
print("Err Std: {}".format(diff2['err'].std()))

for stock_name in stock_dict:
    stock_dict[stock_name]['diff_percent'] = ((stock_dict[stock_name]['dfResult']['Predict'] - stock_dict[stock_name]['dfResult'][stock_dict[stock_name]['label_name']]) / stock_dict[stock_name]['dfResult'][stock_dict[stock_name]['label_name']]).to_frame()
    stock_dict[stock_name]['diff_percent'] = stock_dict[stock_name]['diff_percent'].join(stock_dict[stock_name]['dfResult'][stock_dict[stock_name]['label_name']])
    stock_dict[stock_name]['diff_percent'].columns = ['err percentage', stock_dict[stock_name]['label_name']]

diff3 = stock_dict[symbol]['diff_percent']    

diff3.plot.scatter(x=label_name, y='err percentage', title='Residual Plot (percentage)', figsize=(12, 6))
plt.axhline(y=diff3['err percentage'].mean())
plt.show()

print("Err percentage mean: {}%".format(diff3['err percentage'].mean() * 100))
print("Err percentage Std: {}%".format(diff3['err percentage'].std() * 100))
print("Err percentage 95% CI: {}".format((diff3['err percentage'].mean() - 2 * diff3['err percentage'].std(), diff3['err percentage'].mean() + 2 * diff3['err percentage'].std())))
print("\n")

# Get R2 and MSE list of all stocks
global r2_list
global mse_list
global min_list

r2_list = [stock_dict[x]['R2'] for x in stock_dict]
mse_list = [stock_dict[x]['MSE'] for x in stock_dict]

# Smallest R^2 list
min_list = more_min(r2_list, 1)
print("Smallest R^2:")
print(min_list)
print("\n")

# Get smallest R^2 symbols
min_id_list = [r2_list.index(x) for x in min_list]
min_symbol_list = [stock_dict.keys()[x] for x in min_id_list]
print("Smallest symbols:")
print(min_symbol_list)

# Outlier R^2 list
outlier_list = get_outliers(r2_list)
print("Outlier R^2:")
print(outlier_list)
print("\n")

# Get outlier R^2 symbols
outlier_id_list = [r2_list.index(x) for x in outlier_list]
outlier_symbol_list = [stock_dict.keys()[x] for x in outlier_id_list]
print("Outlier symbols:")
print(outlier_symbol_list)

print("{} seconds elapsed for training/predicting/reporting.".format(time.time() - start_time))
print("\n")

# Inverse sign log transform
def inv_sign_log(df):
    inv_df = np.sign(df) * (np.exp(np.abs(df)) - 1)
    return inv_df

for stock_name in stock_dict:
    stock_dict[stock_name]['original_SMA'] = inv_sign_log(stock_dict[stock_name]['data_X'][symbol + '_SMA'])

sma_y = stock_dict[symbol]['original_SMA'][test_y.index]
naive_y = stock_dict[symbol]['original_data_y'][test_y.index]
drift_y = drift_y

# print(stock_dict['GOOGL']['original_SMA'])
print(len(test_y))
print(len(sma_y))
print(len(naive_y))
print(len(drift_y))
print(test_y.head())
print(sma_y.head())
print(naive_y.head())
print(drift_y.head())

drift_test_df = drift_y.join(test_y)
drift_test_df.columns = ['Drift', 'GOOGL21d']

print("Drift R^2: {}".format(r2_score(drift_test_df['GOOGL21d'], drift_test_df['Drift'])))
print("Drift MSE: {}".format(mean_squared_error(drift_test_df['GOOGL21d'], drift_test_df['Drift'])))

ax = drift_test_df.plot(figsize=(12, 6), title=symbol)
ax.set_xlabel("Date")
ax.set_ylabel("Adj Close")
plt.show()



stock_dict[symbol]['sma_y'] = pd.Series(sma_y)
stock_dict[symbol]['sma_y'] = pd.DataFrame(stock_dict[symbol]['sma_y'].tolist(), index=stock_dict[symbol]['test_y'].index, columns=['SMA'])

sma_y = stock_dict[symbol]['sma_y']

sma_test_df = sma_y.join(test_y)
sma_test_df.columns = ['SMA', 'GOOGL21d']

print("SMA R^2: {}".format(r2_score(sma_test_df['GOOGL21d'], sma_test_df['SMA'])))
print("SMA MSE: {}".format(mean_squared_error(sma_test_df['GOOGL21d'], sma_test_df['SMA'])))

ax = sma_test_df.plot(figsize=(12, 6), title=symbol)
ax.set_xlabel("Date")
ax.set_ylabel("Adj Close")
plt.show()

stock_dict[symbol]['naive_y'] = pd.Series(naive_y)
stock_dict[symbol]['naive_y'] = pd.DataFrame(stock_dict[symbol]['naive_y'].tolist(), index=stock_dict[symbol]['test_y'].index, columns=['Naive'])

naive_y = stock_dict[symbol]['naive_y']

naive_test_df = naive_y.join(test_y)
naive_test_df.columns = ['Naive', 'GOOGL21d']

print("Naive R^2: {}".format(r2_score(naive_test_df['GOOGL21d'], naive_test_df['Naive'])))
print("Naive MSE: {}".format(mean_squared_error(naive_test_df['GOOGL21d'], naive_test_df['Naive'])))

ax = naive_test_df.plot(figsize=(12, 6), title=symbol)
ax.set_xlabel("Date")
ax.set_ylabel("Adj Close")
plt.show()



GOOGL = stock_dict[symbol]['original_data_y']
display(GOOGL)

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=100, center=False).mean()
    rolstd = timeseries.rolling(window=100, center=False).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.figure(figsize=(12,6))
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

test_stationarity(GOOGL)

from statsmodels.tsa.stattools import acf, pacf

def plot_acf_pacf(data, nlags=21):
    lag_acf = acf(data, nlags=nlags)
    lag_pacf = pacf(data, nlags=nlags, method='ols')

    f, ax = plt.subplots(figsize=(12, 6))
    #Plot ACF: 
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(data)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(data)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')

    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(data)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(data)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    
plot_acf_pacf(GOOGL, 200)

GOOGL_log = np.sign(GOOGL) * np.log(abs(GOOGL) + 1)
GOOGL_log_diff = GOOGL_log - GOOGL_log.shift()
print(len(GOOGL_log))
print(len(GOOGL_log_diff))
print(GOOGL_log.head())
print(GOOGL_log_diff.head())

print(len(GOOGL_log_diff))
print(GOOGL_log_diff.head())
plot_acf_pacf(GOOGL_log_diff.dropna(), 10)

test_stationarity(GOOGL_log_diff.dropna())

# GOOGL_log_diff2.dropna(inplace=True)
# print(len(GOOGL_log_diff2))
# print(GOOGL_log_diff2.head())

GOOGL_log_diff.dropna(inplace=True)
print(len(GOOGL_log_diff))
print(GOOGL_log_diff.head())

from statsmodels.tsa.arima_model import ARMA
import warnings

date_length = len(GOOGL_log_diff)
pred_ARMA_list = []

for i in range(date_length - 121 + 1):
    model = ARMA(GOOGL_log_diff.ix[i:i+100], order=(0, 0))
    results_AR = model.fit(disp=-1)
    warnings.filterwarnings("ignore")
    pred_ARMA = results_AR.predict(start=len(results_AR.fittedvalues) - 1, end=len(results_AR.fittedvalues) + 20, dynamic=True)
    pred_ARMA_list.append(pred_ARMA.iloc[-1])

print(len(GOOGL_log_diff))
print(len(pred_ARMA_list))

########

# date_length2 = len(GOOGL_log_diff2)
# pred_ARMA_list2 = []

# for i in range(date_length2 - 121 + 1):
#     model = ARMA(GOOGL_log_diff2.ix[i:i+100], order=(0, 0))
#     results_AR = model.fit(disp=-1)
    
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore")
#         pred_ARMA = results_AR.predict(start=len(results_AR.fittedvalues) - 1, end=len(results_AR.fittedvalues) + 20)
#     pred_ARMA_list2.append(pred_ARMA.iloc[-1])

# print(len(GOOGL_log_diff2))
# print(len(pred_ARMA_list2))

########

# date_length = len(GOOGL_log_diff2)
# pred_ARIMA_list = []

# for i in range(date_length - 121 + 1):
#     model = ARIMA(GOOGL_log.ix[i:i+100], order=(0, 2, 1))
#     results_AR = model.fit(disp=-1)
#     warnings.filterwarnings("ignore")
#     pred_ARIMA = results_AR.predict(start=len(results_AR.fittedvalues) - 1, end=len(results_AR.fittedvalues) + 20)
#     pred_ARIMA_list.append(pred_ARIMA.iloc[-1])

# print(len(GOOGL_log))
# print(len(pred_ARIMA_list))

predictions_ARMA_log_diff = pd.Series(pred_ARMA_list, index=GOOGL_log_diff.index[120:], copy=True)
print(predictions_ARMA_log_diff.head())
print(predictions_ARMA_log_diff.tail())
print(GOOGL_log_diff.iloc[120:].head())
print(GOOGL_log_diff.iloc[120:].tail())

plt.plot(GOOGL_log_diff.iloc[120:])
plt.plot(predictions_ARMA_log_diff)

##########

# predictions_ARMA_log_diff2 = pd.Series(pred_ARMA_list2, index=GOOGL_log_diff.index[121:], copy=True)
# # print(predictions_ARIMA_log_diff.head(50))
# # print(GOOGL_log_diff.iloc[120:].head(50))

# # plt.plot(GOOGL_log_diff2.iloc[120:][datetime(2011, 4, 1):datetime(2011, 6, 31)])
# # plt.plot(predictions_ARMA_log_diff2[datetime(2011, 4, 1):datetime(2012, 6, 31)])

# plt.plot(GOOGL_log_diff2.iloc[121:])
# plt.plot(predictions_ARMA_log_diff2)

##########

# predictions_ARIMA_log_diff2 = pd.Series(pred_ARIMA_list, index=GOOGL_log_diff2.index[120:], copy=True)
# print(predictions_ARIMA_log_diff.head(50))
# print(GOOGL_log_diff.iloc[120:].head(50))

# plt.plot(GOOGL_log_diff2.iloc[120:][datetime(2011, 4, 1):datetime(2011, 6, 31)])
# plt.plot(predictions_ARMA_log_diff2[datetime(2011, 4, 1):datetime(2012, 6, 31)])


# plt.plot(GOOGL_log_diff2.iloc[121:])
# plt.plot(predictions_ARIMA_log_diff2)

# print(GOOGL_log.head())
# print(pred_ARIMA_list.head())

GOOGL_log_diff_df = GOOGL_log_diff.to_frame()
# print(GOOGL_log_diff_df)
GOOGL_log_diff_df['ARMA'] = predictions_ARMA_log_diff
# print(GOOGL_log_diff_df)
# print(GOOGL_log_diff_df['ARIMA'].isnull().sum())

GOOGL_log_diff_df['ARMA'].fillna(GOOGL_log_diff_df['GOOGL'], inplace=True)

print(GOOGL_log_diff_df)

predictions_ARMA_log_diff = GOOGL_log_diff_df['ARMA']

GOOGL_log_addback = GOOGL_log_diff_df['GOOGL'].cumsum() + GOOGL_log.ix[0]
predictions_ARMA_log = predictions_ARMA_log_diff.cumsum() + GOOGL_log.ix[0]
print(GOOGL_log_addback.head())
print(GOOGL_log[1:].head())
print(predictions_ARMA_log.head())

print(GOOGL_log_addback.tail())
print(GOOGL_log[1:].tail())
print(predictions_ARMA_log.tail())

# predictions_ARMA_log_diff_cumsum = predictions_ARMA_log_diff.cumsum()
# print(predictions_ARMA_log_diff_cumsum.head())
# print(predictions_ARMA_log_diff_cumsum.tail())

# predictions_ARMA_log = predictions_ARMA_log_diff_cumsum + GOOGL_log.ix[0]

# print(predictions_ARMA_log.head())
# print(predictions_ARMA_log.tail())
# print(len(predictions_ARMA_log))

plt.plot(predictions_ARMA_log)
plt.plot(GOOGL_log.iloc[120:])

############
# print(GOOGL_log_diff2.iloc[120:].head())
# print(GOOGL_log_diff2.iloc[120:].tail())

# print(predictions_ARMA_log_diff2.head())
# print(predictions_ARMA_log_diff2.tail())

# print(GOOGL_log_diff2.iloc[120:].cumsum().head())
# print(GOOGL_log_diff2.iloc[120:].cumsum().tail())
# predictions_ARMA_log_diff2_cumsum = predictions_ARMA_log_diff2.cumsum()
# print(predictions_ARMA_log_diff2_cumsum.head())
# print(predictions_ARMA_log_diff2_cumsum.tail())

# print((GOOGL_log_diff2.iloc[120:].cumsum() + GOOGL_log_diff.ix[0]).head())
# print((GOOGL_log_diff2.iloc[120:].cumsum() + GOOGL_log_diff.ix[0]).tail())
# predictions_ARMA_log_diff1 = predictions_ARMA_log_diff2_cumsum + GOOGL_log_diff.ix[0]
# print(predictions_ARMA_log_diff1.head())
# print(predictions_ARMA_log_diff1.tail())
# print(len(GOOGL_log_diff2))
# print(len(predictions_ARIMA_log_diff2))

# print(type(GOOGL_log_diff2))
# print(type(predictions_ARIMA_log_diff2))

# GOOGL_log_diff2_df = GOOGL_log_diff2.to_frame()
# print(GOOGL_log_diff2_df)
# GOOGL_log_diff2_df['ARIMA'] = predictions_ARIMA_log_diff2
# print(GOOGL_log_diff2_df)
# # print(GOOGL_log_diff2_df['ARIMA'].isnull())

# GOOGL_log_diff2_df['ARIMA'].fillna(GOOGL_log_diff2_df['GOOGL'], inplace=True)

# print(GOOGL_log_diff2_df)
# predictions_ARIMA_log_diff2 = GOOGL_log_diff2_df['ARIMA']

# # print(((GOOGL_log_diff2.cumsum() - GOOGL_log.ix[0] + GOOGL_log.ix[1]).cumsum() + GOOGL_log.ix[1]).head())
# # print(((GOOGL_log_diff2.cumsum() - GOOGL_log.ix[0] + GOOGL_log.ix[1]).cumsum() + GOOGL_log.ix[1]).tail())

# # print(GOOGL_log[2:].head())
# # print(GOOGL_log[2:].tail())

# GOOGL_log_diff1_addback = GOOGL_log_diff2.cumsum() - GOOGL_log.ix[0] + GOOGL_log.ix[1]
# predictions_ARIMA_log_diff1_addback = predictions_ARIMA_log_diff2.cumsum() - GOOGL_log.ix[0] + GOOGL_log.ix[1]

# print(GOOGL_log_diff1_addback.head())
# print(GOOGL_log_diff1_addback.tail())
# print(predictions_ARIMA_log_diff1_addback.head())
# print(predictions_ARIMA_log_diff1_addback.tail())

# predictions_ARIMA_log_diff1_addback_adj = predictions_ARIMA_log_diff1_addback - 0.0198777033263

# predictions_ARIMA_log = (predictions_ARIMA_log_diff1_addback_adj).cumsum() + GOOGL_log.ix[1]
# print(predictions_ARIMA_log.head())
# print(predictions_ARIMA_log.tail())

# predictions_ARMA_log_diff1_cumsum = predictions_ARMA_log_diff1.cumsum()
# print(predictions_ARMA_log_diff1_cumsum.head())
# print(predictions_ARMA_log_diff1_cumsum.tail())

# predictions_ARMA_log = predictions_ARMA_log_diff1_cumsum + GOOGL_log_diff.ix[0] + GOOGL_log.ix[0]
# print(predictions_ARMA_log_diff1_cumsum.head())
# print(predictions_ARMA_log_diff1_cumsum.tail())

# print(predictions_ARMA_log.head())
# print(predictions_ARMA_log.tail())
# print(len(predictions_ARMA_log))

# plt.plot(GOOGL_log_diff1_addback.iloc[120:])
# plt.plot(predictions_ARIMA_log_diff1_addback)

# print(predictions_ARIMA_log_diff1_addback.mean())
# print(GOOGL_log_diff1_addback.iloc[120:].mean())

# plt.plot(np.exp(GOOGL_log.ioc[120:]))
# plt.plot(GOOGL_log.iloc[120:])
# plt.plot(predictions_ARIMA_log)

# plt.plot(predictions_ARIMA_log)
# plt.plot(GOOGL_log.iloc[120:])

predictions_ARMA = np.sign(predictions_ARMA_log) * (np.exp(np.abs(predictions_ARMA_log)) - 1)

plt.plot(predictions_ARMA)
plt.plot(GOOGL.iloc[120:])

from sklearn.metrics import r2_score, mean_squared_error

# print(GOOGL21d_log.head())
# print(predictions_ARIMA_log.isnull().sum())
print(r2_score(GOOGL_log[1:], predictions_ARMA_log))

predictions_ARMA = np.sign(predictions_ARMA_log) * (np.exp(np.abs(predictions_ARMA_log)) - 1)

# plt.plot(predictions_ARMA)
# plt.plot(GOOGL.iloc[120:])

print(r2_score(GOOGL[1:], predictions_ARMA))


res = (predictions_ARMA - GOOGL.iloc[1:]).to_frame()
# res = res.join(stock_dict[stock_name]['dfResult'][stock_dict[stock_name]['label_name']])
res.columns = ['Residual']
# res = res.join(GOOGL[1:].index)
res = res.join(GOOGL[1:])
print(res)

# plt.plot_date(res.index, res['Residual'])
res.plot.scatter(x='GOOGL', y='Residual', title='Residual Plot', figsize=(12, 6))
plt.axhline(y=res['Residual'].mean())
plt.show()

# print(predictions_ARMA_log.head())
# print(predictions_ARMA_log.tail())

# print(predictions_ARMA_log.shift(-21).head())
# print(predictions_ARMA_log.shift(-21).tail())

data_X['ARMA21d'] = predictions_ARMA_log.shift(-21)
data_X['Naive21d'] = sign_log(naive_y)
data_X['SMA21d'] = sign_log(sma_y)
data_X['Drift21d'] = sign_log(drift_y)
# data_X['ARMA_EMA'] = data_X['ARMA21d'] + data_X[symbol + '_EMA']

print(data_X.head())
print(data_X.tail())

# data_X.dropna(inplace=True)

# print(data_X.head())
# print(data_X.tail())

# print(data_X.isnull().sum())
# print(data_y.isnull().sum())

print(len(data_X))
print(len(data_y))

print(data_X.head())
print(data_X.tail())
print(data_y.head())
print(data_y.tail())

data_X.dropna(inplace=True)
data_y = data_y[data_X.index]

print(len(data_X))
print(len(data_y))

try:
    data_X.drop([symbol + '_SMA', symbol + '_MMA', symbol + '_Vol_M1', symbol + '_Vol_M2', symbol + '_Vol_M3'], axis=1, inplace=True)
except ValueError:
    print('OK, seems like these columns are already gone...')

print(data_X.columns)

def make_predictions(data_X, data_y, date_range, estimator, window=121):
    pred_y_list = []
### Commented out code for monitoring feature importance of the RF
#     feature_importance_list = []
    
    for date in date_range:
        test_X = data_X.ix[date]
        (u,) = data_X.index.get_indexer_for([date])
        
#         print("u: {}".format(u))
#         print("windows: {}".format(window))
        if u - window < 0:
            raise ValueError("Not enough training data!")
            
        train_X = data_X.iloc[(u - window):u - 21]
        train_y = data_y.iloc[(u - window):u - 21]

        estimator.fit(train_X, train_y)
        pred_y = estimator.predict(test_X.reshape(1, -1))
        pred_y_list.append(pred_y)
        
        ## Create feature importance histogram
#         feature_importance_list.append(estimator.feature_importances_)
    
#     vif = [int(np.argmax(x)) for x in feature_importance_list]
#     vif = pd.DataFrame(vif)
#     d = np.diff(np.unique(vif)).min()
#     left_of_first_bin = vif.min() - float(d)/2
#     right_of_last_bin = vif.max() + float(d)/2
#     vif.plot.hist(alpha=0.5, bins=np.arange(left_of_first_bin, right_of_last_bin + d, d))
    return pred_y_list

lr = RandomForestRegressor(n_estimators=10, criterion='mse', max_features='log2', bootstrap=True)
reg = lr

test_y = data_y[121:]
pred_y_series = pd.Series(make_predictions(data_X, data_y, data_y.index[121:], reg, window=121))
pred_y = pd.DataFrame(pred_y_series.tolist(), index=test_y.index, columns=['Predict'])

print(test_y)
print(pred_y)

# print(data_X.isnull().sum())

R2 = r2_score(test_y, pred_y)
MSE = mean_squared_error(test_y, pred_y)

print("R2: {}".format(R2))
print("MSE: {}".format(MSE))
dfResult4 = pred_y.join(test_y)
print(dfResult4.head())
print(dfResult4.tail())
dfResult4.columns = ['Predict', 'Actual']

dfResult4.plot(figsize=(12, 6))

# print(pred_y)
# print(type(pred_y))
# print(type(test_y))
res = (pred_y['Predict'] - test_y)
# print(res)
print(type(res))
res = res.to_frame()
print(type(res))
# print(res)
res.columns = ['Residual']
res = res.join(test_y)
print(res)
# plt.plot_date(res.index, res['Residual'])
res.plot.scatter(x='GOOGL', y='Residual', title='Residual Plot', figsize=(12, 6))
plt.axhline(y=res['Residual'].mean())
plt.show()

print(data_X.columns)

GOOGL_Vol = inv_sign_log(data_X['GOOGL_Vol'])
plt.plot(GOOGL_Vol[datetime(2011, 1, 6):datetime(2011, 3, 28)])
display(GOOGL_Vol.describe())

GOOGL_Vol_log = data_X['GOOGL_Vol']

test_stationarity(GOOGL_Vol_log)
plot_acf_pacf(GOOGL_Vol_log, 10)

GOOGL_Vol_log_diff = GOOGL_Vol_log - GOOGL_Vol_log.shift()
print(len(GOOGL_Vol_log))
print(len(GOOGL_Vol_log_diff))
print(GOOGL_Vol_log.head())
print(GOOGL_Vol_log_diff.head())

plot_acf_pacf(GOOGL_Vol_log_diff.dropna())

test_stationarity(GOOGL_Vol_log_diff.dropna())

GOOGL_Vol_log_diff.describe()

GOOGL_Vol_log_diff.dropna(inplace=True)

GOOGL_Vol_log_diff_cumsum = GOOGL_Vol_log_diff.cumsum()

GOOGL_Vol_log_diff_cumsum_df = GOOGL_Vol_log_diff_cumsum.to_frame()

plt.plot(GOOGL_Vol_log_diff_cumsum_df)

date_length = len(GOOGL_Vol_log_diff)
pred_list = []

for i in range(date_length - 257 + 1):
    model = ARMA(GOOGL_Vol_log_diff.ix[i:i+252], order=(8, 0))
    results_AR = model.fit(disp=-1)
    warnings.filterwarnings("ignore")
    pred_ARMA = results_AR.predict(start=len(results_AR.fittedvalues) - 1, end=len(results_AR.fittedvalues) + 5, dynamic=True)
    pred_list.append(pred_ARMA.iloc[-1])

print(len(GOOGL_Vol_log_diff))
print(len(pred_list))

predictions_log_diff = pd.Series(pred_list, index=GOOGL_Vol_log_diff.index[256:], copy=True)
print(predictions_log_diff.head())
print(predictions_log_diff.tail())
print(GOOGL_Vol_log_diff.iloc[256:].head())
print(GOOGL_Vol_log_diff.iloc[256:].tail())

plt.plot(GOOGL_Vol_log_diff.iloc[256:])
plt.plot(predictions_log_diff)

GOOGL_Vol_log_diff_df = GOOGL_Vol_log_diff.to_frame()

GOOGL_Vol_log_diff_df['ARMA_Vol'] = predictions_log_diff

GOOGL_Vol_log_diff_df['ARMA_Vol'].fillna(GOOGL_Vol_log_diff_df['GOOGL_Vol'], inplace=True)

print(GOOGL_Vol_log_diff_df)

predictions_log_diff = GOOGL_Vol_log_diff_df['ARMA_Vol']

GOOGL_Vol_log_addback = GOOGL_Vol_log_diff_df['GOOGL_Vol'].cumsum() + GOOGL_Vol_log.ix[0]
predictions_log = predictions_log_diff.cumsum() + GOOGL_Vol_log.ix[0]
print(GOOGL_Vol_log_addback.head())
print(GOOGL_Vol_log[1:].head())
print(predictions_log.head())

print(GOOGL_Vol_log_addback.tail())
print(GOOGL_Vol_log[1:].tail())
print(predictions_log.tail())

plt.plot(predictions_log)
plt.plot(GOOGL_Vol_log)

# plt.plot(predictions_log[datetime(2014, 1, 1):datetime(2014, 1, 15)])
# plt.plot(GOOGL_Vol_log[datetime(2014, 1, 1):datetime(2014, 1, 15)])



predictions = inv_sign_log(predictions_log)
GOOGL_Vol = inv_sign_log(GOOGL_Vol_log)

# plt.plot(predictions)
# plt.plot(GOOGL_Vol)

plt.plot(predictions[datetime(2012, 1, 1):datetime(2012, 1, 31)])
plt.plot(GOOGL_Vol[datetime(2012, 1, 1):datetime(2012, 1, 31)])

print(r2_score(GOOGL_Vol_log[1:], predictions_log))
print(r2_score(GOOGL_Vol[1:], predictions))
# (u,) = predictions_log.index.get_indexer_for([datetime(2014, 1, 1)])
# print(u)

plt.plot(GOOGL_Vol[:251])



