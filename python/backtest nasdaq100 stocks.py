get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import time
from buy_and_hold import BuyAndHoldStrategy
from scipy.stats.mstats import gmean

data = pd.read_csv("buy_and_hold_prices.csv", index_col = "Date") 
#database of NASDAQ100 stocks from yahoo (Adj Close)
data.index = pd.to_datetime(data.index)

strategy = BuyAndHoldStrategy(data)
strategy.plot()

strategy.summary

strategy.annual_returns

windows = np.array([50,60,70,80,90,100,110,120])
quantiles = np.array([0.75,0.8,0.85,0.9,0.95])

#optimization on 1/2006-6/2017 = ALL
returns_all = {}
start_time = time.time()
for w in range(len(windows)):
    ret_row = []
    for q in range(len(quantiles)):
        s = BuyAndHoldStrategy(data, window=windows[w], quantile=quantiles[q],
                               spx = False, maxdrawdown = False)
        ret_row.append(gmean(s.annual_returns+1)-1)
    returns_all[windows[w]] = ret_row
print("Computation time: " + str(np.round(time.time() - start_time,2)))

returns_all = pd.DataFrame(returns_all, index=quantiles)
returns_all.columns = windows
returns_all = returns_all.astype(np.float64)

returns_all

window_int = [100,110,120]
quantile_int = [0.85,0.9]

start_dates = np.random.randint(0,500,50)

start_dates = np.sort(data.index[start_dates])

#grid search on 1/2006-6/2009 = WORST CASE random starts ... ROBUST 
start_time = time.time()
ret = []
for start in start_dates:
    returns = {}
    for w in range(len(window_int)):
        ret_row = []
        for q in range(len(quantile_int)):
            s = BuyAndHoldStrategy(data[start:dt.date(2009,6,30)], window=windows[w], 
                                   quantile=quantiles[q], spx = False, 
                                   maxdrawdown = False)
            ret_row.append(gmean(s.annual_returns+1)-1)
        returns[windows[w]] = ret_row
    returns = pd.DataFrame(returns, index=quantile_int)
    returns.columns = window_int
    returns = returns.astype(np.float64)
    ret.append(returns)

print("Computation time: " + str(np.round(time.time() - start_time,2)))

arr = []
for i in range(len(start_dates)):
    arr.append(np.array(ret[i]))

arr = np.array(arr)

output = pd.Panel(arr, items=start_dates, 
                  major_axis=quantile_int, 
                  minor_axis=window_int)

print("Mean of 50 average returns:")
pd.DataFrame(np.mean(output, axis = 0), index=quantile_int, 
             columns=window_int)

print("Window length:")
pd.Series(np.mean(np.mean(output, axis = 0), axis = 0), 
          index=window_int)

print("Quantile:")
pd.Series(np.mean(np.mean(output, axis = 0), axis = 1), 
          index = quantile_int)

#grid search on 1/2006-6/2017 =  random starts ... ALL TIME ROBUST 
start_time = time.time()
ret2 = []
for start in start_dates:
    returns = {}
    for w in range(len(window_int)):
        ret_row = []
        for q in range(len(quantile_int)):
            s = BuyAndHoldStrategy(data[start:], window=windows[w], 
                                   quantile=quantiles[q], spx = False, 
                                   maxdrawdown = False)
            ret_row.append(gmean(s.annual_returns+1)-1)
        returns[windows[w]] = ret_row
    returns = pd.DataFrame(returns, index=quantile_int)
    returns.columns = window_int
    returns = returns.astype(np.float64)
    ret2.append(returns)

print("Computation time: " + str(np.round(time.time() - start_time,2)))

arr = []
for i in range(len(start_dates)):
    arr.append(np.array(ret2[i]))
arr = np.array(arr)
output2 = pd.Panel(arr, items=start_dates, major_axis=quantile_int, 
                   minor_axis=window_int)

print("Mean of 50 average returns:")
pd.DataFrame(np.mean(output2, axis = 0), index=quantile_int, 
             columns=window_int)

print("Window length:")
pd.Series(np.mean(np.mean(output2, axis = 0), axis = 0), 
          index=window_int )

print("Quantile:")
pd.Series(np.mean(np.mean(output2, axis = 0), axis = 1), 
          index = quantile_int)

