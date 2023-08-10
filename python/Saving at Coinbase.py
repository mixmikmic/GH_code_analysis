# %load ../../load_magic/storage.py

get_ipython().system('mkdir ..\\data')
get_ipython().system('mkdir ..\\data\\csv')
get_ipython().system('mkdir ..\\saves')
get_ipython().system('mkdir ..\\saves\\pickle')
get_ipython().system('mkdir ..\\saves\\csv')
import pickle
import pandas as pd
import os

# Handy list of the different types of encodings
encoding = ['latin1', 'iso8859-1', 'utf-8'][2]

# Change this to your data and saves folders
data_folder = r'../data/'
saves_folder = r'../saves/'

def load_csv(csv_name=None, folder_path=None):
    if folder_path is None:
        csv_folder = data_folder + 'csv/'
    else:
        csv_folder = folder_path + 'csv/'
    if csv_name is None:
        csv_path = max([os.path.join(csv_folder, f) for f in os.listdir(csv_folder)],
                       key=os.path.getmtime)
    else:
        csv_path = csv_folder + csv_name + '.csv'
    data_frame = pd.read_csv(csv_path, encoding=encoding)
    
    return(data_frame)

def load_dataframes(**kwargs):
    frame_dict = {}
    for frame_name in kwargs:
        pickle_path = saves_folder + 'pickle/' + frame_name + '.pickle'
        if not os.path.isfile(pickle_path):
            csv_folder = saves_folder + 'csv/'
            csv_path = csv_folder + frame_name + '.csv'
            if not os.path.isfile(csv_path):
                csv_path = data_folder + 'csv/' + frame_name + '.csv'
                if not os.path.isfile(csv_path):
                    frame_dict[frame_name] = None
                else:
                    frame_dict[frame_name] = load_csv(csv_name=frame_name)
            else:
                frame_dict[frame_name] = load_csv(csv_name=frame_name, folder_path=csv_folder)
        else:
            frame_dict[frame_name] = load_object(frame_name)
    
    return frame_dict

def load_object(obj_name, download_url=None):
    pickle_path = saves_folder + 'pickle/' + obj_name + '.pickle'
    if not os.path.isfile(pickle_path):
        csv_path = saves_folder + 'csv/' + obj_name + '.csv'
        if not os.path.isfile(csv_path):
            object = pd.read_csv(download_url, low_memory=False,
                                 encoding=encoding)
        else:
            object = pd.read_csv(csv_path, low_memory=False,
                                 encoding=encoding)
        if isinstance(object, pd.DataFrame):
            object.to_pickle(pickle_path)
        else:
            with open(pickle_path, 'wb') as handle:
                pickle.dump(object, handle, pickle.HIGHEST_PROTOCOL)
    else:
        try:
            object = pd.read_pickle(pickle_path)
        except:
            with open(pickle_path, 'rb') as handle:
                object = pickle.load(handle)
    
    return(object)

def save_dataframes(include_index=False, **kwargs):
    csv_folder = saves_folder + 'csv/'
    for frame_name in kwargs:
        if isinstance(kwargs[frame_name], pd.DataFrame):
            csv_path = csv_folder + frame_name + '.csv'
            kwargs[frame_name].to_csv(csv_path, sep=',', encoding=encoding,
                                      index=include_index)

# Classes, functions, and methods cannot be pickled
def store_objects(**kwargs):
    for obj_name in kwargs:
        if hasattr(kwargs[obj_name], '__call__'):
            raise RuntimeError('Functions cannot be pickled.')
        obj_path = saves_folder + 'pickle/' + str(obj_name)
        pickle_path = obj_path + '.pickle'
        if isinstance(kwargs[obj_name], pd.DataFrame):
            kwargs[obj_name].to_pickle(pickle_path)
        else:
            with open(pickle_path, 'wb') as handle:
                pickle.dump(kwargs[obj_name], handle, pickle.HIGHEST_PROTOCOL)

def attempt_to_pickle(df, pickle_path, raise_exception=False):
    try:
        print('Pickling to ' + pickle_path)
        df.to_pickle(pickle_path)
    except Exception as e:
        os.remove(pickle_path)
        print(e, ': Couldn\'t save ' + '{:,}'.format(df.shape[0]*df.shape[1]) + ' cells as a pickle.')
        if raise_exception:
            raise


# Download CoinBase prices
from urllib.request import urlretrieve
from gzip import GzipFile
import os
import numpy as np
from datetime import datetime


# http://api.bitcoincharts.com/v1/csv/
def get_price_data(price_history_name, price_history_url, column_list):
    frame_name = str(price_history_name) + '_df'
    pickle_path = saves_folder + 'pickle/' + frame_name + '.pickle'
    if not os.path.isfile(pickle_path):
        
        csv_path = data_folder + 'csv/' + frame_name + '.csv'
        if not os.path.isfile(csv_path):

            out_file_path = data_folder + 'csv/' + str(price_history_name) + '.csv'
            if not os.path.isfile(out_file_path):
                print('Retrieving ' + price_history_url)
                local_filename, headers = urlretrieve(price_history_url)
                print('Decompressing ' + local_filename)
                with gzip.open(local_filename, 'rb') as f:
                    price_history_decompressed = f.read()
                print('Decoding to ' + out_file_path)
                with open(out_file_path, 'w') as output:
                    size = output.write(price_history_decompressed.decode())
            
            print('Converting ' + out_file_path)
            price_history_df = pd.read_csv(out_file_path, encoding=encoding, header=None)

            price_history_df.columns = column_list
            if 'time_stamp' in column_list:
                price_history_df['date'] = price_history_df['time_stamp'].map(lambda x: pd.to_datetime(x, unit='s'))
            else:
                price_history_df['date'] = price_history_df['date'].map(lambda x: datetime.strptime(str(x).strip(), '%x'))
            price_history_df = price_history_df.sort_values('date')
            price_history_df['year'] = price_history_df['date'].map(lambda x: x.year)
            price_history_df['month'] = price_history_df['date'].map(lambda x: x.month)
            price_history_df['day'] = price_history_df['date'].map(lambda x: x.day)
            price_history_df['week_day'] = price_history_df['date'].map(lambda x: x.weekday())
            hard_date = price_history_df ['date'].min()
            price_history_df['days_from'] = price_history_df ['date'].map(lambda x: (x - hard_date).days)
            price_history_df['log_price'] = price_history_df['price'].map(lambda x: np.log(x))
            price_history_df['price_diff'] = price_history_df['price'].diff()
            price_history_df['log_diff'] = price_history_df['log_price'].diff()

            match_series = (price_history_df['log_price'] <= 0)
            price_history_df = price_history_df[~match_series]
            
            try:
                attempt_to_pickle(price_history_df, pickle_path, raise_exception=True)
            except:
                print('Now saving as a csv.')
                price_history_df.to_csv(csv_path, sep=',', encoding=encoding, index=False)
                
        else:
            print('Loading ' + frame_name + ' from csv')
            price_history_df = load_csv(frame_name, folder_path=saves_folder)
            price_history_df['date'] = price_history_df['date'].map(lambda x: pd.to_datetime(x))
            attempt_to_pickle(price_history_df, pickle_path)

    else:
        print('Loading ' + frame_name + ' from pickle')
        price_history_df = load_object(frame_name)
    
    return price_history_df


# Validates with https://en.wikipedia.org/wiki/History_of_bitcoin
price_history_url = 'http://api.bitcoincharts.com/v1/csv/coinbaseUSD.csv.gz'
column_list = ['time_stamp', 'price', 'idono']
coinbaseUSD_df = get_price_data('coinbaseUSD', price_history_url, column_list)
coinbaseUSD_df.head()


hard_date = coinbaseUSD_df ['date'].min()
coinbaseUSD_df['days_from'] = coinbaseUSD_df ['date'].map(lambda x: (x - hard_date).days)
coinbaseUSD_df['price_diff'] = coinbaseUSD_df.sort_values('date')['price'].diff()
coinbaseUSD_df['log_price'] = coinbaseUSD_df['price'].map(lambda x: np.log(x))
coinbaseUSD_df['log_diff'] = coinbaseUSD_df.sort_values('date')['log_price'].diff()
coinbaseUSD_df.head()


pickle_path = saves_folder + 'pickle/coinbaseUSD_df.pickle'
try:
    attempt_to_pickle(coinbaseUSD_df, pickle_path, raise_exception=True)
except:
    print('Now saving as a csv.')
    csv_path = data_folder + 'csv/coinbaseUSD_df.csv'
    coinbaseUSD_df.to_csv(csv_path, sep=',', encoding=encoding, index=False)
coinbaseUSD_df.head()


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from pyhsmm.util.plot import pca_project_data

column_list = ['days_from', 'price']
line_2d = plt.plot(pca_project_data(coinbaseUSD_df[column_list].as_matrix(), 2))


def get_min_max(df, column_name, circle_min=5, circle_max=500):
    min_max_scaler = MinMaxScaler(feature_range=(circle_min, circle_max))
    min_max = min_max_scaler.fit_transform(df[column_name].values.reshape(-1, 1))
    
    return min_max


get_ipython().run_line_magic('matplotlib', 'inline')
import random
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

numeric_list = ['price', 'year', 'month', 'day', 'week_day', 'log_price', 'price_diff', 'log_diff']
column_list = random.sample(numeric_list, 4)
df = coinbaseUSD_df[column_list].dropna()
df.sample(5).T


c_column_name = column_list.pop()
s_column_name = column_list.pop()
df_matrix = df[column_list].as_matrix()
y_column_name = column_list.pop()
x_column_name = column_list.pop()
cmap = plt.get_cmap('viridis_r')


from sklearn.cluster import DBSCAN

# Compute DBSCAN
db_DBSCAN = DBSCAN().fit(df_matrix)


# Don't want to fit this puppy again
store_objects(db_DBSCAN=db_DBSCAN)


fig = plt.figure(figsize=(13, 13))
ax = fig.add_subplot(111, autoscale_on=True)
path_collection = ax.scatter(df_matrix[:, 0], df_matrix[:, 1],
                             s=get_min_max(df, s_column_name),
                             c=get_min_max(df, c_column_name),
                             edgecolors=(0, 0, 0), cmap=cmap)
title_text = ('Scatterplot of the Coinbase ' +
              x_column_name + ' and ' +
              y_column_name + ' Fields with ' +
              s_column_name + ' as the Size and ' +
              c_column_name + ' as the Color')
text = plt.title(title_text)


coinbaseUSD_df.loc[coinbaseUSD_df.shape[0]-1]


coinbaseUSD_df['date'].dtype


groupby_list = ['year', 'month', 'day']
monday_price_df = coinbaseUSD_df.groupby(groupby_list,
                                         as_index=False).apply(lambda x: x.loc[[x.date.idxmax()]]).copy()
match_series = (monday_price_df['week_day'] == 0)
monday_price_df = monday_price_df[match_series].reset_index(drop=True)


from datetime import datetime
import pandas as pd

trading_fee = 0.0149
weekly_budget = 5.0
latest_price = coinbaseUSD_df.loc[coinbaseUSD_df.shape[0]-1]['price']
def total_saved(weekly_budget, format_result=True):
    weekly_amount = weekly_budget - weekly_budget*trading_fee
    column_list = ['date_time', 'shares_added']
    rows_list = []
    trading_date = datetime(monday_price_df.loc[0, 'year'].squeeze(),
                            monday_price_df.loc[0, 'month'].squeeze(),
                            monday_price_df.loc[0, 'day'].squeeze())
    for row_index, row_series in monday_price_df.iterrows():
        row_dict = {}
        row_dict['date_time'] = row_series['date']
        row_dict['shares_added'] = weekly_amount/row_series['price']
        rows_list.append(row_dict)

    weekly_amount_df = pd.DataFrame(rows_list, columns=column_list)
    total_amount = weekly_amount_df['shares_added'].sum()*latest_price
    if format_result:
        total_amount = '${:,.2f}'.format(total_amount)
    
    return total_amount


print(total_saved(10.))


import statsmodels.formula.api as smf

# Use ols function for calculating the F-statistic and associated p value
price_history_ols = smf.ols(formula='log_price ~ time_stamp', data=coinbaseUSD_df)
price_history_fitted = price_history_ols.fit()
price_history_fitted.summary()


params_series = price_history_fitted.params

def date_when(amount):
    time_stamp = (np.log(amount)-params_series.loc['Intercept'])/params_series.loc['time_stamp']
    date_time = pd.to_datetime(time_stamp, unit='s')
    
    return date_time.strftime('%Y-%m-%d')


date_when(110000)


get_ipython().run_line_magic('matplotlib', 'inline')

axes_sub_plot = coinbaseUSD_df.plot(x='date', y='price', kind='line', figsize=(15, 5), logy=True)


# How do you add a second line to the plot?
axes_sub_plot.plot(x=coinbaseUSD_df['date'], y=np.exp(params_series.loc['Intercept']+coinbaseUSD_df['log_price']*params_series.loc['time_stamp']),
                   kind='line', figsize=(15, 5), logy=True)


coinbaseUSD_df.loc[0, 'date']


column_list = ['amount_per_week', 'total_saved']
amount_per_week = range(0, 500)
amount_saved = [total_saved(amount, format_result=False) for amount in amount_per_week]
rows_list = [dict(zip(column_list, [w, s])) for w, s in zip(amount_per_week, amount_saved)]
weekly_amount_df = pd.DataFrame(rows_list, columns=column_list)
axes_sub_plot = weekly_amount_df.plot(x='amount_per_week', y='total_saved', kind='line', figsize=(15, 5), logy=False)


price_history_url = None
column_list = ['date', 'close', 'volume', 'price', 'high', 'low']
WMT_df = get_price_data('WMT', price_history_url, column_list)
WMT_df.head()


coinbaseUSD_df['log_diff'].std()


match_series = (WMT_df['date'] < coinbaseUSD_df.loc[0, 'date'])
WMT_df[~match_series]['log_diff'].std()


coinbaseUSD_df.shape


SaP_df = pd.read_html(data_folder + 'html/S_and_P_500_by_month.html')[0]


SaP_df.columns = ['date', 'price']
SaP_df = SaP_df.reindex(SaP_df.index.drop(0))
SaP_df['date'] = SaP_df['date'].map(lambda x: pd.to_datetime(x.strip()))
match_series = (SaP_df['date'] < coinbaseUSD_df.loc[0, 'date'])
SaP_df = SaP_df[~match_series]
SaP_df['price'] = SaP_df['price'].map(lambda x: float(x))
SaP_df['year'] = SaP_df['date'].map(lambda x: x.year)
SaP_df['month'] = SaP_df['date'].map(lambda x: x.month)
SaP_df['day'] = SaP_df['date'].map(lambda x: x.day)
SaP_df['week_day'] = SaP_df['date'].map(lambda x: x.weekday())
SaP_df['log_price'] = SaP_df['price'].map(lambda x: np.log(x))
SaP_df['log_diff'] = SaP_df.sort_values('date')['price'].diff()
SaP_df.sort_values('log_diff', ascending=False).head()


# Why is the S&P 500 volatility so much greater than Coinbase?
match_series = (SaP_df['date'] < coinbaseUSD_df.loc[0, 'date'])
SaP_df[~match_series]['log_diff'].std()


# From https://www.investopedia.com/ask/answers/021015/what-best-measure-given-stocks-volatility.asp



