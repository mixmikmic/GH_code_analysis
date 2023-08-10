#print list of currencies being analyzed and construct the list of currencies symbols

import cryptocompare
coin = cryptocompare.get_coin_list(format=False)

#list the currencies you wish to analyze here
coin_top10 = ['Bitcoin', 'Ethereum ', 'Bitcoin Cash / BCC', 'Ripple', 'Litecoin', 'NEM', 'DigitalCash', 'ZCash','Monero', 'IOTA', 'Ethereum Classic', 'Nxt', 'Stellar']
coin_top10_sym = []
for key, value in coin.items():
    if value['CoinName'] in coin_top10:
        print(key, value['CoinName'])
        coin_top10_sym.append(key)

#making API calls to get hourly price data then construct a dataframe of each currencies price data with UNIX timestamp as index

import requests
import json
import pandas as pd
from pandas.io.json import json_normalize

#determine how many rows of data here
lim = '18000' #querying 2 weeks back

df = pd.DataFrame()
for i in coin_top10_sym:
    URL = 'https://min-api.cryptocompare.com/data/histohour?fsym='+i+'&tsym=USD&limit='+lim
    data = requests.get(URL)
    json_data = data.json()
    table = json_normalize(json_data, 'Data').set_index('time')
    table.index = pd.to_datetime(table.index ,unit='s')
    df = pd.concat([df, table.high], axis=1)
df.columns = coin_top10_sym

#df = df.loc[:'2017-11-03', :]

# Performing Dickey-Fuller stationary test

from statsmodels.tsa.stattools import adfuller

for i in df.columns: 
    x = df[i].values
    result = adfuller(x)
    print('\033[1m' + i + '\033[0m')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

# Perfrom differencing to stationalize the series

# Creat difference function, with default value of lag 1
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

# Differencing the dataset 
df_diff = pd.DataFrame()
for i in df.columns:
    df_diff[i] = difference(df[i])

# Re-perform Dickey-fuller test to test the differenced series for stationality

for i in df.columns: 
    x = df_diff[i].values
    result = adfuller(x)
    print('\033[1m' + i + '\033[0m')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

#create dataframes for each of the posible currencies combinations

from itertools import permutations
com = list(permutations(coin_top10_sym, 2))
pair_list = []
count = 0
for i in com:
    globals()[str(i[0]) + '_' + str(i[1])] = df_diff[[com[count][0], com[count][1]]]
    count = count + 1
    pair_list.append(str(i[0]) + '_' + str(i[1]))

# Performing Granger Causality Test

import statsmodels.tsa.stattools as sm
lag = 24
cor = {}
for i in pair_list:
    cor[i] = sm.grangercausalitytests(eval(i), lag)

# Granger Causality test results

for key, values in cor.items():
    print('\n')
    print('\033[1m' + key + '\033[0m')
    for i in range(1, lag+1):
        print('lag', i, '=', values[i][0]['lrtest'][1])
    

# Printing top correlated coins results

# Manually gone through the test results to identify these pairs.
top = ['NXT_LTC', 'NXT_IOT', 'BCH_ETC', 'XEM_NXT', 'XEM_LTC', 'XEM_ETH', 'XEM_IOT', 'DASH_BCH', 'XMR_BCH', 'XMR_DASH', 'ZEC_DASH']
#rep = ['NXT_LTC', 'XEM_NXT', 'XEM_LTC', 'XEM_IOT']
#new = ['NXT_ETC', 'XEM_ETC', 'XEM_BTC', 'XEM_DASH','DASH_LTC', 'DASH_IOT']
for key, values in cor.items():
    if key in top:
        print('\n')
        print('\033[1m' + key + '\033[0m')
        for i in range(1, lag+1):
            print('lag', i, '=', values[i][0]['lrtest'][1])

# Scaling and visualizing XEM_IOT

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
df_plot = pd.DataFrame(index=df.index)
df_plot['XEM'] = sc_x.fit_transform(df['XEM'].values.reshape(-1,1))
df_plot['IOT'] = sc_x.fit_transform(df['IOT'].values.reshape(-1,1))

# Plot 1- scaled overlay plot of DASH/ETH prices 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.plot(df.index, df_plot['XEM'], color='blue')
plt.plot(df.index, df_plot['IOT'], color='green')
#plt.xlim(('2017-09-15', '2017-10-15'))
plt.ylim((-4, 4))
plt.legend(loc='lower left')
plt.xlabel('Time')
plt.ylabel('Prices (scaled)')
plt.title('XEM/IOT Prices over Time')
plt.show()

# Performing Granger Test on XEM_IOT, with maxlag = 50

highest_cor = sm.grangercausalitytests(XEM_IOT, 50)

# XEM_IOT Grenger Test results
for key, values in highest_cor.items():
        print('lag', key, '=', values[0]['lrtest'][1])

#XEM_IOT.to_csv('xem_iot.csv')



