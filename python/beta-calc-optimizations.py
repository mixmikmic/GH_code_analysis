import pandas as pd
import numpy as np
import datetime

# source: http://stackoverflow.com/questions/2030053/random-strings-in-python
import random, string
def randomword(length): return ''.join(random.choice(string.ascii_uppercase) for i in range(length))

# settings
start = '01-01-2000'
end = '01-01-2016'
ticker_count = 1000       # number of tickers
window = 100              # window for beta calc

idx = pd.date_range(start=start, end=end)
tickers = [randomword(4) for i in range(ticker_count+1)]
dt = {t: np.random.uniform(-.05, .05, len(idx)) for t in tickers}
df = pd.DataFrame(data=dt, index=idx)

mkt = tickers[0]
window = 100

df.head() # dataframe created 

df.shape # number of values = days x tickers

df_beta = df.copy()
df_beta['date'] = df_beta.index
stock = tickers[1]

def historical_beta(date, mkt, stock, window):
    start = date + datetime.timedelta(days=-window)
    data = df.loc[(df.index < date) & (df.index > start)][[stock, mkt]]
    if (len(data) < 10): return np.nan
    cov = np.cov(data)[0][1]
    var=np.var(data[mkt])
    beta = cov / var
    return beta

df_beta[stock] = df_beta.apply(lambda x: historical_beta(x.date, mkt=mkt, stock=stock, window=200), axis=1)

start = datetime.datetime(2012,1,1)
end = start + datetime.timedelta(days=window)
data = df.loc[(df.index < end) & (df.index > start)][[tickers[0], tickers[1]]]

x = np.array(data[tickers[0]])
y = np.array(data[tickers[1]])
A = np.vstack([x, np.ones(len(x))]).T

get_ipython().magic('timeit cov = np.cov(x, y)[0][1]')

get_ipython().magic('timeit m, c = np.linalg.lstsq(A, y)[0]')

print(m, c)

covs = df.rolling(window=window).cov(df[mkt], pairwise=True)
var = df[mkt].rolling(window=window).var()
beta = covs.div(var,axis=0)

# difference
(17.7 * 1000) / 5.81



