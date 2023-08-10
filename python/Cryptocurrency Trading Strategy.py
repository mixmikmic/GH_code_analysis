import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

def CryptoData(symbol, frequency):
    #Parameters: String symbol, int frequency = 300,900,1800,7200,14400,86400
    #Returns: df from first available date
    url ='https://poloniex.com/public?command=returnChartData&currencyPair='+symbol+'&end=9999999999&period='+str(frequency)+'&start=0'
    df = pd.read_json(url)
    df.set_index('date',inplace=True)
    return df

df = CryptoData(symbol = 'BTC_XRP', frequency = 300)
#When frequency = 300, data is refreshed every 5 minutes

df['Leading'] = df['close'].rolling(1000).mean()
df['Lagging'] = df['close'].rolling(5000).mean()
df[['close','Leading','Lagging']][270000:].plot(figsize = (18,10), grid = True)
plt.xlabel('Date')
plt.ylabel('Price')

def test_ma(df, lead, lag, pc_thresh = 0.025):
    ma_df = df.copy()
    ma_df['lead'] = ma_df['close'].rolling(lead).mean()
    ma_df['lag'] = ma_df['close'].rolling(lag).mean()
    ma_df.dropna(inplace = True)
    ma_df['lead-lag'] = ma_df['lead'] - ma_df['lag']
    ma_df['pc_diff'] = ma_df['lead-lag'] / ma_df['close']
    ma_df['regime'] = np.where(ma_df['pc_diff'] > pc_thresh, 1, 0)
    ma_df['regime'] = np.where(ma_df['pc_diff'] < -pc_thresh, -1, ma_df['regime'])
    ma_df['Market'] = np.log(ma_df['close'] / ma_df['close'].shift(1))
    ma_df['Strategy'] = ma_df['regime'].shift(1) * ma_df['Market']
    ma_df[['Market','Strategy']] = ma_df[['Market','Strategy']].cumsum().apply(np.exp)
    return ma_df

ma_df = test_ma(df, 1000, 5000).dropna()
ma_df.tail()

ma_df[['Market','Strategy']].iloc[-1]

#Would have outperformed the market by a factor of about 46

ma_df[['Market','Strategy']][270000:].plot(figsize = (16,10), grid = True)
plt.xlabel('Date')
plt.ylabel('BTC_XRP Return')

leads = np.arange(100, 4100, 100)
lags = np.arange(4100, 8100, 100)
lead_lags = [[lead,lag] for lead in leads for lag in lags]
pnls = pd.DataFrame(index=lags,columns = leads)

get_ipython().run_cell_magic('capture', '', "\nfor lead, lag in lead_lags:\n    pnls[lead][lag] = test_ma(df, lead, lag)['Strategy'][-1]\n    print(lead,lag,pnls[lead][lag])\n    \n#Output is hidden because it includes 1,600 rows")

PNLs = pnls[pnls.columns].astype(float)
plt.subplots(figsize = (14,10))
sns.heatmap(PNLs,cmap= 'PiYG')

PNLs.max()

optimal_ma_df = test_ma(df, 400, 4300).dropna()
optimal_ma_df[['close','lead','lag']][270000:].plot(figsize = (18,10), grid = True)
plt.xlabel('Date')
plt.ylabel('Price')

optimal_ma_df[['Market','Strategy']][270000:].plot(figsize = (16,10), grid = True)
plt.xlabel('Date')
plt.ylabel('BTC_XRP Return')

