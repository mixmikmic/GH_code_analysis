import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import dateutil.parser
from pprint import pprint
import json
import time
import sys
import re
import datetime

# APIs
import quandl

# Quandl API Calls
df_price = pd.read_csv('https://www.quandl.com/api/v3/datasets/BNC3/GWA_BTC.csv?api_key=pvPBMBW8afR_HqVfio9o') # Price, volume
df_eth = pd.read_csv('https://www.quandl.com/api/v3/datasets/GDAX/ETH_USD.csv?api_key=pvPBMBW8afR_HqVfio9o') # ETH Price, volume
df_fees = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/TRFUS.csv?api_key=pvPBMBW8afR_HqVfio9o') # Txn fees
df_cost = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/CPTRA.csv?api_key=pvPBMBW8afR_HqVfio9o') # cost per txn
df_no = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/NTRAN.csv?api_key=pvPBMBW8afR_HqVfio9o') # num txns
df_noblk = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/NTRBL.csv?api_key=pvPBMBW8afR_HqVfio9o') # txns per block
df_blksz = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/AVBLS.csv?api_key=pvPBMBW8afR_HqVfio9o') # blk size
df_unq = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/NADDU.csv?api_key=pvPBMBW8afR_HqVfio9o') # unique addys
df_hash = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/HRATE.csv?api_key=pvPBMBW8afR_HqVfio9o') # hash rate
df_diff = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/DIFF.csv?api_key=pvPBMBW8afR_HqVfio9o') # difficulty

df_nasdaq = pd.read_csv('https://www.quandl.com/api/v3/datasets/NASDAQOMX/COMP.csv?api_key=pvPBMBW8afR_HqVfio9o') # NASDAQ Composite
df_nasdaq = df_nasdaq.rename(columns={'Trade Date': 'Date','Index Value':'Nasdaq'})
df_nasdaq = df_nasdaq.drop(['High','Low','Total Market Value','Dividend Market Value'], 1)

df_gold = pd.read_csv('https://www.quandl.com/api/v3/datasets/NASDAQOMX/QGLD.csv?api_key=pvPBMBW8afR_HqVfio9o') # Nasdaq GOLD Index
df_gold = df_gold.rename(columns={'Trade Date': 'Date','Index Value':'Gold'})
df_gold = df_gold.drop(['High','Low','Total Market Value','Dividend Market Value'], 1)

# Helper functions
def to_date(datestring):
    date = dateutil.parser.parse(datestring)
    return date

def list_to_average(list):
    try:
        avg = list[0]/list[1]
    except:
        avg = 0
    return avg

def to_log(num):
    return np.log(num)

df = df_price.drop('Open', 1)
df = df.drop(['High','Low'], 1)
df = df.rename(columns={'Close': 'BTCPrice','Volume':'BTCVol'})
df = df_eth.merge(df,how='inner',on='Date')
df = df.rename(columns={'Open': 'ETHPrice'})
df = df.drop(['High','Low'], 1)
df = df_fees.merge(df, how='inner', on='Date')
df = df.rename(columns={'Value': 'TxFees'})
df = df_cost.merge(df,how='inner',on='Date')
df = df.rename(columns={'Value': 'CostperTxn'})
df = df_no.merge(df,how='inner',on='Date')
df = df.rename(columns={'Value': 'NoTxns'})
df = df_noblk.merge(df,how='inner',on='Date')
df = df.rename(columns={'Value': 'NoperBlock'})
df = df_blksz.merge(df,how='inner',on='Date')
df = df.rename(columns={'Value': 'AvgBlkSz'})
df = df_unq.merge(df,how='inner',on='Date')
df = df.rename(columns={'Value': 'UniqueAddresses'})
df = df_hash.merge(df,how='inner',on='Date')
df = df.rename(columns={'Value': 'HashRate'})
df = df_diff.merge(df,how='inner',on='Date')
df = df.rename(columns={'Value': 'Difficulty'})

df = df_nasdaq.merge(df,how='inner',on='Date')
df = df_gold.merge(df,how='inner',on='Date')

ct = [i for i in reversed(range(len(df)))]
df['DateNum'] = ct 

df['Date'] = df['Date'].apply(to_date)
df['Date'] = pd.to_datetime(df['Date'])
df['Date2'] = df['Date']
df = df.set_index('Date2')

df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['Weekday'] = df['Date'].dt.weekday


df = df[['BTCPrice','ETHPrice','BTCVol','TxFees','CostperTxn','NoTxns','NoperBlock','AvgBlkSz','UniqueAddresses',
         'HashRate','Difficulty','Nasdaq','Gold','DateNum','Date','Month','Quarter','Weekday']]
df_hist = df

print(df_hist.shape)
df_hist.info()

df_hist.head(2)

df_hist.tail(2)

# run coinmarketcap_hist.py weekly to generate .json file (note file number)
mkt_cap = pd.read_json('data/coinmarketcap_hist2.json').T
mkt_cap['Date'] = mkt_cap.index
mkt_cap['Date'] = pd.to_datetime(mkt_cap['Date'],format='%Y%m%d',errors='ignore')
mkt_cap = mkt_cap.set_index('Date')
mkt_cap = mkt_cap[['BTC','ETH','Crypto Market Cap']]
mkt_cap.tail()

# Download weekly from google trends from time period 2013-04-28 to present day
# https://trends.google.com/trends/explore?date=2013-04-28%202018-01-31&q=bitcoin
df_goog = pd.read_csv('data/20180131_GoogleTrendsSearchInterest.csv') # Google Trends "bitcoin" interest over time
df_goog = df_goog.iloc[2:]
df_goog = df_goog.rename(columns={'Category: All categories': 'Interest'})
df_goog['Date2'] = df_goog.index
df_goog['Date2'] = pd.to_datetime(df_goog['Date2'])
df_goog = df_goog.set_index('Date2')
# df_goog.info()

df_mc = pd.concat([mkt_cap, df_goog], axis=1)
df_mc.tail()

df_all = pd.concat([df_hist, df_mc], axis=1)
df_all = df_all.fillna(method='ffill')
df_all = df_all.iloc[200:,:]
df_all.head()

df_all = df_all[['BTCPrice','ETHPrice','BTCVol','Crypto Market Cap', 'CostperTxn','TxFees','NoTxns','AvgBlkSz','UniqueAddresses','HashRate','Difficulty','Nasdaq','Gold','Interest','DateNum','Quarter','Month','Weekday']]
df_all = pd.DataFrame(df_all,dtype=np.float64) # convert all values to float64

# add log columns
df_all['logBTCPrice'] = df_all['BTCPrice'].apply(to_log)
df_all['logNasdaq'] = df_all['Nasdaq'].apply(to_log)
df_all['logETHPrice'] = df_all['ETHPrice'].apply(to_log)
df_all['logGold'] = df_all['Gold'].apply(to_log)
df_all['logCrypto Market Cap'] = df_all['Crypto Market Cap'].apply(to_log)
df_all['logInterest'] = df_all['Interest'].apply(to_log)
df_all['logCostperTxn'] = df_all['CostperTxn'].apply(to_log)
df_all['logTxFees'] = df_all['TxFees'].apply(to_log)
df_all['logNoTxns'] = df_all['NoTxns'].apply(to_log)
df_all['logAvgBlkSz'] = df_all['AvgBlkSz'].apply(to_log)
df_all['logUniqueAddresses'] = df_all['UniqueAddresses'].apply(to_log)
df_all['logHashRate'] = df_all['HashRate'].apply(to_log)
df_all['logBTCVol'] = df_all['BTCVol'].apply(to_log)
df_all['logDifficulty'] = df_all['Difficulty'].apply(to_log)

df_all.columns

# pickle the consolidate DataFrame
df_all.to_pickle('data/benson_btcsentiment_df.pkl')

# Coinmarketcap: Current market cap information by top coin
url = 'https://coinmarketcap.com/all/views/all/'
response=requests.get(url)
page=response.text
soup=BeautifulSoup(page,"lxml")

tables=soup.find_all("table")

rows=[row for row in tables[0].find_all('tr')]
df_curr = pd.read_html(tables[0].prettify())[0]
df.to_pickle('data/benson_btcsentiment_dfcurr.pkl')
df_curr.head()

# Coinmarketcap scraping: Bitcoin by time period.  When updating, be sure to download for period 2013-04-28 to present
url = 'https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=20180131'
response=requests.get(url)
page=response.text
soup=BeautifulSoup(page,"lxml")
tables=soup.find_all("table")

rows=[row for row in tables[0].find_all('tr')]
df = pd.read_html(tables[0].prettify())[0]
df['Date']=df['Date'].apply(to_date)
df = df.set_index('Date')
df.to_pickle('data/benson_btcsentiment_dfts.pkl')
df.head()

df.tail()

# Coinmarketcap scraping: Ethereum. When updating, be sure to download for period 2013-04-28 to present
url = 'https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20130428&end=20180131'
response=requests.get(url)
page=response.text
soup=BeautifulSoup(page,"lxml")
tables=soup.find_all("table")

rows=[row for row in tables[0].find_all('tr')]
df_eth = pd.read_html(tables[0].prettify())[0]
df_eth['Date']=df_eth['Date'].apply(to_date)
df_eth = df_eth.set_index('Date')
df_eth.to_pickle('data/benson_btcsentiment_dftseth.pkl')
df_eth.head()

# a script was written to collect twitter sentiment data and coinmarketcap market cap data in real time (every 5 mins)
# as historical twitter sentiment analysis data could not be located.  One week of data has been collected to date, 
# but is very noisy.  To continue collecting data by running the script as a daemon and incorporate in a future
# analysis.  See the script at btcpricesentiment*.py
btcsa = pd.read_json('data/btcpricesentiment_mc7.json',convert_axes=False).T # connect to most recent active file

btcsa['Date'] = btcsa.index
btcsa['Date'] = pd.to_datetime(btcsa['Date'])
btcsa['Bitcoin_S'] = btcsa['bitcoin_S'].apply(list_to_average)
btcsa['Ethereum_S'] = btcsa['ethereum_S'].apply(list_to_average)
btcsa['Blockchain_S'] = btcsa['blockchain_S'].apply(list_to_average)
btcsa = btcsa[['Date','Bitcoin_S','Ethereum_S','Blockchain_S','BTC','ETH','LTC','ADA','EOS','Crypto Market Cap']]
btcsa = btcsa.sort_values('Date')
btcsa = btcsa.set_index('Date')
btcsa.to_pickle('data/benson_btcsentiment_dfs.pkl')
print(btcsa.shape)
btcsa.head(2)

# Bitcoin futures scraping.  As futures began trading in December 2017, it appears there is not yet enough information
# to incorporate into the analysis at this time.  To revisit in the future.  
url = 'http://www.cmegroup.com/trading/equity-index/us-index/bitcoin.html'
response=requests.get(url)
page=response.text
soup=BeautifulSoup(page,"lxml")

tables=soup.find_all("table")

rows=[row for row in tables[0].find_all('tr')]
df = pd.read_html(tables[0].prettify())[0]
df = df[:5]
df



# TWITTER'S MAIN API FOR SENTIMENT ANALYSIS IS UTILIZED AS FOLLOWS:
# import pandas
# import json

# from tweepy import Stream
# from tweepy.streaming import StreamListener

# class MyListener(StreamListener):

#     def on_data(self, data):
#         try:
#             with open('bitcoin.json', 'a') as f:
#                 f.write(data)
#                 return True
#         except BaseException as e:
#             print("Error on_data: %s" % str(e))
#         return True

#     def on_error(self, status):
#         print(status)
#         return True

# twitter_stream = Stream(auth, MyListener())
# twitter_stream.filter(track=['#bitcoin'])

