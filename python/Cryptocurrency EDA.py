import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')

def CryptoData(symbol, frequency):
    #Parameters: String symbol, int frequency = 300,900,1800,7200,14400,86400
    #Returns: df from first available date
    url ='https://poloniex.com/public?command=returnChartData&currencyPair='+symbol+'&end=9999999999&period='+str(frequency)+'&start=0'
    df = pd.read_json(url)
    df.set_index('date',inplace=True)
    return df

df = CryptoData(symbol = 'USDT_BTC', frequency = 86400)['close']
df.tail(5)

df.plot(figsize = (10,7), title = 'BTC USD Historical Price', grid = True)

df.pct_change().describe()

df.pct_change().hist(bins=100, figsize = (10, 7))

#Below shows the distribution of the daily percentage changes of BTC

def CryptoDataCSV(symbol, frequency):
    #Parameters: String symbol, int frequency = 300,900,1800,7200,14400,86400
    #Returns: df from first available date
    url ='https://poloniex.com/public?command=returnChartData&currencyPair='+symbol+'&end=9999999999&period='+str(frequency)+'&start=0'
    df = pd.read_json(url)
    df.set_index('date',inplace=True)
    df.to_csv(symbol + '.csv')
    print('Processed: ' + symbol)
    
#Use this method to extract data into a csv file so we can combine the data together

tickers =  ['USDT_BTC','USDT_ETC','USDT_XMR','USDT_ETH','USDT_DASH',
 'USDT_XRP','USDT_LTC']
    
for ticker in tickers:
    CryptoDataCSV(ticker, 86400)

crypto_df = pd.DataFrame()
for ticker in tickers:
    crypto_df[ticker] = pd.read_csv(ticker+'.csv', index_col = 'date')['close']
    
crypto_df.dropna(inplace=True)

crypto_df.head()

crypto_df.tail()

crypto_df_norm = crypto_df.divide(crypto_df.iloc[0])

#To compare the relative performance of the coins, we divide the whole dataframe by the first row
#The subsequent data points represent each coins' respective percentage gain since July 29, 2017

crypto_df_norm.plot(figsize = (16,8), grid = True)
plt.xlabel('Date')
plt.ylabel('Percent Change')

crypto_df_pct = crypto_df.pct_change().dropna()
crypto_df_pct.describe()

corr = crypto_df_pct.corr()
sns.heatmap(corr, cmap = 'rainbow', annot=True, linewidth = 1)

#Below is a heatmap of the correlation matrix of the cryptocurrencies

fig, ax = plt.subplots(figsize=(10,7))
regplot = sns.regplot(x='USDT_ETH', y='USDT_ETC', data=crypto_df_pct, color = 'g', marker = '+')
regplot.set(xlabel = 'USDT_ETH % Return', ylabel = 'USDT_ETC % Return')
sns.set_style("whitegrid")

import statsmodels.api as sm
model = sm.OLS(crypto_df_pct['USDT_ETH'],
               crypto_df_pct['USDT_ETC']).fit()
model.summary()

