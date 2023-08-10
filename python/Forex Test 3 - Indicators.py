import numpy as np
from scipy.signal import argrelextrema
from matplotlib import pyplot
import pandas as pd
import pandas.io.data as web
get_ipython().magic('matplotlib inline')

XD = web.get_data_yahoo('F')['Close'] 

def MACD(y, a=26, b=12):
    return pd.ewma(y, span=12) - pd.ewma(y, span=26) #12-26

def RSI(y, windown=14):
    dy = y.diff()
    dy.iat[0] = dy.iat[1]
    u = dy.apply(lambda x: x if (x > 0) else 0) # uptrend 0 with where it goes down
    d = dy.apply(lambda x: -x if (x < 0) else 0) # downtred 0 with where it goes up
    # simple exponential moving average
    rs = pd.ewma(u, span=windown)/pd.ewma(d, span=windown)
    return 100 - (100/(1+rs))

X = pd.DataFrame(XD).iloc[-80::,:]
#X = pd.read_pickle('GOOG_may_29.pandas')
fig, arx = pyplot.subplots(3, sharex=True, figsize=(15,8))
X['Close'].plot(ax=arx[0])
arx[0].set_ylabel('GOOG Close')
X['Close_RSI'] = RSI(X['Close'])
X['Close_RSI'].plot(ax=arx[1])
arx[1].set_ylim(0, 100)
arx[1].set_ylabel('RSI indicator')
X['Close_MACD'] = MACD(X['Close'])
X['Close_MACD'].plot(ax=arx[2])
arx[2].set_ylabel('MACD indicator')

import pandas as pd
from matplotlib import pyplot
import seaborn as sns
get_ipython().magic('matplotlib inline')
#pyplot.figure()

def plot_quotes(df):    
    sns.set_context("talk")
    quotes = list(df.columns[1:]) # ignore first collumn
    f, axr = pyplot.subplots(len(quotes), sharex=True, figsize=(15,15))
    for i, ax in enumerate(axr):
        df.iloc[:,i+1].plot(ax=axr[i])
        axr[i].set_ylabel(quotes[i])

df = pd.read_pickle('cotes_21hs_29_may_16_6hours.pandas')

df2 = df.copy()
del df2['USD/JPY']
#df2.consolidate?

plot_quotes(df)

import numpy as np
from scipy.signal import argrelextrema
from matplotlib import pyplot
get_ipython().magic('matplotlib inline')
x = X['Close'].values
# for local maxima indexes
maxx = argrelextrema(x, np.greater)
# for local minima indexes
maxy = argrelextrema(x, np.less)

pyplot.figure(figsize=(17,6))
pyplot.plot(x, '+-')
pyplot.plot(maxx[0], x[maxx[0]], '--')
pyplot.plot(maxy[0], x[maxy[0]], '--')
pyplot.plot(maxx[0])

