get_ipython().run_line_magic('matplotlib', 'inline')
# Imports
from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Jupyter")
AddReference("QuantConnect.Indicators")
from System import *
from QuantConnect import *
from QuantConnect.Data.Market import TradeBar, QuoteBar
from QuantConnect.Jupyter import *
from QuantConnect.Indicators import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create an instance
qb = QuantBook()

qb.AddEquity("SPY")
qb.AddEquity("VXX")

symbols = ["SPY","VXX"]
data = {}
for syl in symbols:
    qb.AddEquity(syl)
    data[syl] = qb.History(syl, datetime(2014,1,1), datetime.now(), Resolution.Daily).loc[syl]['close']
    data[syl].index = data[syl].index.date
#     data[syl].plot(label=syl)
# plt.legend()
# plt.ylabel('Adjusted Price')
# plt.figure(figsize =(15,7))
# for syl in symbols:    
#     (1+data[syl].pct_change()).cumprod().plot(label=syl)
# plt.legend()
# plt.ylabel('Cumulative Return')

price = pd.DataFrame(data, columns=data.keys())

day=len(data["SPY"])
ema = qb.Indicator(ExponentialMovingAverage(30), "VXX", day, Resolution.Daily)
ema.columns = ['EMA']

df = pd.concat([ema,price], axis=1, join='inner')

plt.style.use('seaborn-whitegrid')
df.plot(figsize=(18,10),style=['r--','g','b'])

df['sell'] = df['EMA'][(df['VXX'] > df['EMA']) & (df['VXX'].shift(1) < df['EMA'])]                      
df['buy'] = df['EMA'][(df['VXX'] < df['EMA']) & (df['VXX'].shift(1) > df['EMA'])]                            
df.plot(figsize =(17,10), style=['r--', 'g', 'b', 'm^','cv'])                                             

signal = []
for i in range(len(df)):
    if not np.isnan(df['buy'][i]):
        signal.append(1)
    elif not np.isnan(df['sell'][i]):
        signal.append(0)
    else:
        signal.append(np.nan)
   

df['signal'] = signal

df = df.fillna(method='ffill')
df['sell'] = df['EMA'][(df['VXX'] > df['EMA']) & (df['VXX'].shift(1) < df['EMA'])]                      
df['buy'] = df['EMA'][(df['VXX'] < df['EMA']) & (df['VXX'].shift(1) > df['EMA'])]     

# SPY_return = np.log(df['SPY']) - np.log(df['SPY'].shift(1))
# cum_return = (np.exp(SPY_return)*(df['signal'].shift(1))).cumsum()

SPY_return = df['SPY'].pct_change()
cum_return = ((SPY_return)*df['signal'].shift(1)).cumsum()

plt.figure(figsize =(15,7))
plt.plot(df.index, cum_return+1,label ='cum_return')
plt.plot(df.index, df['SPY']/df['SPY'][0],label ='SPY')
plt.legend()





