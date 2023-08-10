get_ipython().run_line_magic('matplotlib', 'inline')
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
from datetime import date, timedelta
from calendar import monthrange
get_ipython().run_line_magic('pylab', 'inline')
# Create an instance
qb = QuantBook()

symbols = [u'AAPL', u'C', u'XOM', u'JPM', u'GOOG', u'MSFT', u'LVS', u'INTC', u'AMZN', 
           u'CSCO', u'F', u'WFC', u'NFLX', u'FCX', u'QCOM', u'PBR', u'GS', u'JNJ', u'WMT',
           u'VALE', u'HPQ', u'GE', u'CVX', u'ORCL', u'IBM', u'POT', u'BIDU', u'MRK', u'PG', u'VZ',
           u'KO', u'PCLN', u'MS', u'BP', u'CMCSA', u'MCD', u'PFE', u'T', u'COP', u'CAT', 
           u'SLB', u'MA', u'BA', u'DTV', u'APC', u'V', u'MOS', u'PEP', u'EBAY', u'NEM', u'NTAP',
           u'AMGN', u'WYNN', u'CRM', u'CLX', u'S', u'ABX', u'OXY', u'ABT', u'TEVA', u'USB', u'FFIV', 
           u'GILD', u'MON', u'NVDA', u'SLW', u'HAL', u'AXP', u'MGM', u'TGT', u'MU', u'UPS', u'BTU',
           u'EXPE', u'CLF', u'CNX', u'ESRX', u'PM', u'MET', u'CMG', u'CF', u'UNH', u'BBY', u'CHK',
           u'COST', u'RIG', u'AIG', u'EMR', u'CME', u'CVS', u'DIS', u'CL', u'UNP', u'DE', u'DD',
           u'NOV', u'HD']
symbols.sort()

start_date = datetime.datetime(2011,1,1)
end_date = datetime.datetime.now()
num_stocks = 45 # should be less than len(symbols)
num_ports = 5

data = qb.GetFundamental(symbols, "ValuationRatios.PERatio", start_date, end_date)

df = data
# fill the NaN with the forward data, 
# drop the NaN rows, 
# transpose the dataframe with the symbol index
df = df.fillna(method='ffill').dropna().T
df.index= symbols # change index to symbol names
# remove the stocks if there are zero values
df = df[~(df == 0).any(axis=1)][:num_stocks]
# change columns name to date type
df.columns =[i.date() for i in df.columns]

drop_columns = []
for i in range(len(df.columns)):
    date = df.columns[i]
    end_day_of_month = date.replace(day=monthrange(date.year, date.month)[1])
    if date != end_day_of_month:
        drop_columns.append(df.columns[i])
# drop columns if it is not the last day of the month        
df = df.drop(drop_columns, axis=1)

# create a dictionary keyed by symbols, valued by a list of history close
# add the benchmark "SPY"
hist = {}
sym_list = list(df.index)
sym_list.append(u'SPY')
for symbol in sym_list:    
    qb.AddEquity(symbol) 
    history = qb.History(symbol, start_date, end_date, Resolution.Daily).loc[symbol]["close"]
    hist[symbol] = history

def port_monthly_return(syls, month_date):
    # syls(list): symbols
    # month_date(datetime.date): date for calculate the monthly return
    # return value: a list of average return for each portfolio
    num_each_port = int(num_stocks/float(num_ports))
    port_ret = []
    for i in range(num_ports):
        sum_ret = 0  # the sum of return in one portfolio
        for j in range(i*num_each_port,(i+1)*num_each_port):
            price = hist[syls[j]].to_frame()[month_date.strftime("%Y-%m")]
            sum_ret += (price.iloc[-1] - price.iloc[0]) / price.iloc[0]
        port_ret.append(np.mean(sum_ret)) 
    # add monthly return of "SPY" to the end of the list
    hist_benchmark = hist[syls[-1]].to_frame()[month_date.strftime("%Y-%m")]
    res_benchmark = (hist_benchmark.iloc[-1] - hist_benchmark.iloc[0]) / hist_benchmark.iloc[0]
    port_ret.append(res_benchmark[0])
    return port_ret

ret = []
for i in range(len(df.columns)):
    ranked_syls = df.sort_values(df.columns[i]).index  
    ret.append(port_monthly_return(ranked_syls,df.columns[i]))
df_return = pd.DataFrame(ret, index = df.columns)

# plot the cumulative return for five portfolios and the benchmark
plt.figure(figsize =(15,7))
for i in range(num_ports):
    plt.plot(df_return.cumsum()[i], label = 'port%d'%(i+1))
plt.plot(df_return.cumsum()[num_ports], label = 'benchmark', linestyle='--', color='b', linewidth=2)
plt.xlabel('Portfolio Return: factor PE ratio', fontsize=12)
plt.legend(loc=0)

total_return = {}
annual_return = {}
excess_return = {}
win_prob = {}
loss_prob = {}
effect_test = {}
MinCorr = 0.3
Minbottom = -0.05
Mintop = 0.05
effect_test = {}
total_return = (df_return+1).cumprod().iloc[-1,:]-1
for i in range(len(total_return)):
    if total_return.iloc[i]<-1:
        total_return.iloc[i] = -0.99999
num_years = len(df_return)/12.0
annual_return = list((total_return+1)**(12.0/len(df_return))-1)
excess_return = list(annual_return - annual_return[-1])

result =[]
correlation = np.corrcoef(annual_return[:num_ports],[i+1 for i in range(num_ports)])[0][1]
result.append(correlation)

if total_return.iloc[0]<total_return.iloc[-2]:
    loss_excess = df_return.iloc[:,0]-df_return.iloc[:,-1]
    loss_prob = loss_excess[loss_excess<0].count()/float(len(loss_excess))
    win_excess = df_return.iloc[:,-2]-df_return.iloc[:,-1]
    win_prob = win_excess[win_excess>0].count()/float(len(win_excess))
    result.append(loss_prob)
    result.append(win_prob)
    
    excess_return_win = excess_return[-2]
    excess_return_loss = excess_return[0]
    result.append(excess_return_win)
    result.append(excess_return_loss)    
    
elif total_return.iloc[0]>total_return.iloc[-2]:
    loss_excess = df_return.iloc[:,-2]-df_return.iloc[:,-1]
    loss_prob = loss_excess[loss_excess<0].count()/float(len(loss_excess))
    win_excess = df_return.iloc[:,0]-df_return.iloc[:,-1]
    win_prob = win_excess[win_excess>0].count()/float(len(win_excess))
    result.append(loss_prob)
    result.append(win_prob)
    
    excess_return_win = excess_return[0]
    excess_return_loss = excess_return[-2]
    result.append(excess_return_win)
    result.append(excess_return_loss)    

result

factors = ["ValuationRatios.PERatio", "ValuationRatios.BookValuePerShare", 
           "ValuationRatios.FCFYield", "ValuationRatios.BookValueYield",
           "ValuationRatios.PricetoEBITDA","ValuationRatios.EVToEBITDA",
            "ValuationRatios.SalesPerShare", "ValuationRatios.EarningYield",
            "ValuationRatios.TrailingDividendYield","ValuationRatios.PriceChange1M",
            "ValuationRatios.TangibleBookValuePerShare","ValuationRatios.PEGRatio",
            "ValuationRatios.PCFRatio",  "ValuationRatios.PBRatio",
            "ValuationRatios.CFYield", "ValuationRatios.WorkingCapitalPerShare",
            "ValuationRatios.ExpectedDividendGrowthRate","ValuationRatios.SalesYield"]

factor_test = {}
for factor_name in factors:
    
    data = qb.GetFundamental(symbols, factor_name, start_date, end_date)
    df = data
    # fill the NaN with the forward data, 
    # drop the NaN rows, 
    # transpose the dataframe with the symbol index
    df = df.fillna(method='ffill').dropna().T
    df.index= symbols # change index to symbol names
    # remove the stocks if there are zero values
    df = df[~(df == 0).any(axis=1)][:num_stocks]
    # change columns name to date type
    df.columns =[i.date() for i in df.columns]

    
    drop_columns = []
    for i in range(len(df.columns)):
        date = df.columns[i]
        end_day_of_month = date.replace(day=monthrange(date.year, date.month)[1])
        if date != end_day_of_month:
            drop_columns.append(df.columns[i])
    # drop columns if it is not the last day of the month        
    df = df.drop(drop_columns, axis=1)
    
    # create a dictionary keyed by symbols, valued by a list of history close
    # add the benchmark "SPY"
    hist = {}
    sym_list = list(df.index)
    sym_list.append(u'SPY')
    for symbol in sym_list:    
        qb.AddEquity(symbol) 
        history = qb.History(symbol, start_date, end_date, Resolution.Daily).loc[symbol]["close"]
        hist[symbol] = history
    

    ret = []
    for i in range(len(df.columns)):
        ranked_syls = df.sort_values(df.columns[i]).index  
        ret.append(port_monthly_return(ranked_syls,df.columns[i]))
    df_return = pd.DataFrame(ret, index = df.columns)
    
    
    # plot the cumulative return for five portfolios and the benchmark
    plt.figure(figsize =(13,6))
    for i in range(num_ports):
        plt.plot(df_return.cumsum()[i], label = 'port%d'%(i+1))
    plt.plot(df_return.cumsum()[num_ports], label = 'benchmark', linestyle='--', color='b', linewidth=2)
    plt.xlabel('Portfolio Return: factor %s'%factor_name, fontsize=12)
    plt.legend(loc=0)
 

    total_return = (df_return+1).cumprod().iloc[-1,:]-1
    for i in range(len(total_return)):
        if total_return.iloc[i]<-1:
            total_return.iloc[i] = -0.99999
    num_years = len(df_return)/12.0
    annual_return = list((total_return+1)**(12.0/len(df_return))-1)
    excess_return = list(annual_return - annual_return[-1])
    
    
    result =[]
    correlation = np.corrcoef(annual_return[:num_ports],[i+1 for i in range(num_ports)])[0][1]
    result.append(correlation)
    
    if total_return.iloc[0]<total_return.iloc[-2]:
        loss_excess = df_return.iloc[:,0]-df_return.iloc[:,-1]
        loss_prob = loss_excess[loss_excess<0].count()/float(len(loss_excess))
        win_excess = df_return.iloc[:,-2]-df_return.iloc[:,-1]
        win_prob = win_excess[win_excess>0].count()/float(len(win_excess))
        result.append(loss_prob)
        result.append(win_prob)

        excess_return_win = excess_return[-2]
        excess_return_loss = excess_return[0]
        result.append(excess_return_win)
        result.append(excess_return_loss)    
    
    elif total_return.iloc[0]>total_return.iloc[-2]:
        loss_excess = df_return.iloc[:,-2]-df_return.iloc[:,-1]
        loss_prob = loss_excess[loss_excess<0].count()/float(len(loss_excess))
        win_excess = df_return.iloc[:,0]-df_return.iloc[:,-1]
        win_prob = win_excess[win_excess>0].count()/float(len(win_excess))
        result.append(loss_prob)
        result.append(win_prob)

        excess_return_win = excess_return[0]
        excess_return_loss = excess_return[-2]
        result.append(excess_return_win)
        result.append(excess_return_loss) 
    
    factor_test[factor_name] = result

pd.DataFrame(factor_test, 
             index = ['correlation','loss probability','win probability',
                      'excess return(win)','excess return(loss)'])



