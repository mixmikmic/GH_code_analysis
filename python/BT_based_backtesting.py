import bt
import pandas as pd
import seaborn as sns
import datetime
import dask.dataframe as dd

get_ipython().magic('pylab inline')

end = datetime.datetime.now()
start = end - datetime.timedelta(days=364)
pairs = ['USDT_BTC','USDT_ETH', 'USDT_LTC']

def get_full_table(pair, start, end):
    """
    gets 300000 raw trades
    """
    df = pd.read_json('https://poloniex.com/public?command=returnTradeHistory&currencyPair={}&start={}&end={}'.format(pair, int(start.timestamp()), int(end.timestamp())))
    df.set_index(['date'], inplace=True)
    print('fetched {} {} trades.'.format(df.size,pair))
    return df

def get_price_table(pair, start, end):
    """
    Poloniex API only returns maximum of 300000 trades or 1 year for each pair.
    :returns:
    dictionary with one dataframe per pair
    """
    print('Downloading {} from {} to {}.'.format(pair,start,end))
    
    df = get_full_table(pair,start,end)
    
    df = df.resample('1T').mean() #resample in windows of 1 minute
    df[pair] = df.rate
    for cname in df.columns:
        if cname not in pairs:
            del df[cname]

    return df

def concatenate_series(rates):
    """
    :parameters:
    - rates: dictionary with the pairs dataframes
    """
    for k, df in rates.items(): #Solve non-unique indices
        rates[k] = df.loc[~df.index.duplicated(keep='first')]
    data = pd.concat(rates, axis=1)
    data.columns = data.columns.droplevel(0)
    print(data.columns)
    data.columns = [name.lower() for name in data.columns] #convenient to save to PGSQL
    return data

def extend_history(pair, df):
    End = df.index.min()
    Start = end - datetime.timedelta(days=364)
    dfextra = get_price_table(pair,Start,End)
    df = df.append(dfextra)#pd.concat([df,dfextra], axis=0)
    return df

rates = {pair: get_price_table(pair, start, end) for pair in pairs}

rates['USDT_ETH'].head()

rates['USDT_ETH'].tail()

rates = {pair: extend_history(pair, rates[pair]) for pair in rates}

rates['USDT_ETH'].info()

data = concatenate_series(rates)
data.plot(figsize=(10,10), logy=True);

data.dropna().head()

# create the strategy
s = bt.Strategy('s1', [bt.algos.RunDaily(),
                       bt.algos.SelectAll(),
                       bt.algos.WeighInvVol(),
                       bt.algos.Rebalance()])

# create a backtest and run it
test = bt.Backtest(s, data.dropna())
res = bt.run(test)

res.plot()

res.display()

res.plot_histogram(alpha=0.5)
plt.grid()

res.plot_security_weights();





df.iloc[-1]



