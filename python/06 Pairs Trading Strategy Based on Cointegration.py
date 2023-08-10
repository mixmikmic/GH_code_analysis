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
# Create an instance
qb = QuantBook()
# plt.style.available

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import statsmodels.api as sm
from math import floor
plt.style.use('seaborn-whitegrid')
from sklearn import linear_model

def calculateQuantity(signal, price, cash):
    """
    This function calculate the quantity based on the signal and initial cash 
    
    Parameters:
            signal(pandas.Series): The trading signal series of stock indexed by date (long 1, short -1, holding 0)
            price(pandas.Series) : The price series of stock indexed by date
            cash(float): The total cash for trading
    
    Returns(pandas.Series):
            quantity(pandas.Series): The number of holding shares indexed by date 
    """

    index = np.where(signal.shift(1) != signal)[0][1:]
    quantity_temp = floor(cash/price[0])
    cash_left_temp = cash
    quantity = [quantity_temp] #* len(signal)
    cash_left = [cash_left_temp] #* len(signal)

    for i in range(1,len(price)):
        if i in index:
            if signal[i-1] * signal[i] == -1:
                cash_left_liquidate = cash_left[i-1] + (signal[i-1]- 0) * quantity[-1] * price[i]                    
                quantity_temp = floor(cash_left_liquidate / price[i])
                cash_left_temp = cash_left_liquidate + (0 - signal[i]) * quantity_temp * price[i] 
 
                if quantity_temp == 0:
                    # print("{0} Order Failed (No enough money)  Cash left: {1} share price: {2}".format(signal.index[i],cash_left_temp,price[i]))                    
                    quantity_temp =  quantity[i-1]
                
                if cash_left_liquidate < 0:
                    quantity_temp = 0
                                         
            elif signal[i-1] * signal[i] == 0:
                quantity_temp = floor(cash_left[i-1] / price[i])
                cash_left_temp = cash_left[i-1] + (signal[i-1]- signal[i]) * quantity_temp * price[i]        
                    
                if quantity_temp == 0:
                    # print("{0} Order Failed (No enough money)  Cash left: {1} share price: {2}".format(signal.index[i],cash_left_temp,price[i]))
                    quantity_temp =  quantity[i-1]
                       
        quantity.append(quantity_temp)
        cash_left.append(cash_left_temp)
   
    return pd.Series(quantity, index = signal.index)

class NetProfit:
    """
    This class calculates the net profit for strategy trading individual stock
    
    Args:
        price(pandas.Series) : The price series of stock indexed by date
        signal(pandas.Series): The trading signal series of stock indexed by date (long 1, short -1, holding 0)

    Attributes:
        price(pandas.Series) : The price series of stock indexed by date
        signal(pandas.Series): The trading signal series of stock indexed by date (long 1, short -1, holding 0)
        quantity(pandas.Series): The number of holding shares indexed by date 
    
    Note:
        If there is no quantity, the default value of quantity is 1 share at each time step)       
    """
    
    def __init__(self, price, signal):
        self.price = price
        self.signal = signal
        self.quantity = pd.Series([1]*len(self.price),index = self.price.index)
        
        
    def net_profit(self):
       
        """
        calculate the net profit
        
        Returns(pandas.Series):
                The net profit for strategy        
        """
        #   log_return = np.log(self.price/self.price.shift(1))
        #   cum_return = np.exp(((log_return)*self.signal.shift(1)).cumsum())*self.quantity
        pct_return = self.price.pct_change()
        cum_return = ((pct_return)*self.signal.shift(1) + 1).cumprod()*self.quantity 
        net_profit = cum_return.dropna()*self.price[0] #- self.quantity * self.price[0]

        return net_profit   
    
class PortfolioNetProfit:
    """
    This class calculates the net profit for strategy trading a porfolio of stocks or singal stock
    
    Args:
        data(dict): A dictionary stores the data for multiple stocks 
                    keys(string): 
                                symbols
                    values(dataframe): 
                                Index: date 
                                Columns: ['price','signal','quantity'] or ['price','signal']         
    """
    
    def __init__(self,data):
        self.data = data
        
    def net_profit(self):
        """
        Calculate the net profit for the portfolio
        
        Returns(pandas.Series):
                The net profit for strategy
            
        """
        dataframe = self.data[self.data.keys()[0]]
        net_profit_port = pd.Series([0]*(len(dataframe)),index = dataframe.index)
       
        for i in self.data:
            df = self.data[i]
            net_profit_each = NetProfit(df['price'],df['signal'])
            try:
                net_profit_each.quantity = df['quantity'] # if there is no quantity, the default is 1 
            except:
                pass
            cum_return = net_profit_each.net_profit()
            net_profit_port = net_profit_port.add(cum_return,fill_value=0)        

        return net_profit_port[1:]
    
    def curve(self):
        """
        Plot the equity curve for strategy contain a portfolio of stocks
        """
        net_profit_port = self.net_profit()
        plt.figure(figsize =(15,7))
        plt.plot(net_profit_port.index, net_profit_port,label ='Profit ($)')
        plt.legend()        

syls = ["XOM","CVX"]
qb.AddEquity(syls[0])
qb.AddEquity(syls[1])
start = datetime(2003,1,1)
end = datetime(2017,1,1)
x = qb.History(syls[0],start ,end, Resolution.Daily).loc[syls[0]]['close']
y = qb.History(syls[1],start ,end, Resolution.Daily).loc[syls[1]]['close']

price = pd.concat([x, y], axis=1)
price.columns = syls 
lp = np.log(price)

in_sample_size = len(lp[:'2009'])
out_sample_size = len(lp)-in_sample_size
in_sample = lp[:in_sample_size]
out_sample = lp[out_sample_size:]

def reg(x,y):
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x,pd.Series([1]*len(x),index = x.index)], axis=1)
    regr.fit(x_constant, y)    
    beta = regr.coef_[0]
    alpha = regr.intercept_
    spread = y - x*beta - alpha
    return spread

x = in_sample[syls[0]]
y = in_sample[syls[1]]
spread = reg(x,y)

# check if the spread is stationary 
adf = sm.tsa.stattools.adfuller(spread, maxlag=1)
print 'ADF test statistic: %.02f' % adf[0]
for key, value in adf[4].items():
    print('\t%s: %.3f' % (key, value))
print 'p-value: %.03f' % adf[1]

numdays = 250 # set the length of formation period
threshold = 1.5
signal_x = [0] * out_sample_size
signal_y = [0] * out_sample_size
for i in range(1,out_sample_size):
    df = lp[in_sample_size+i-numdays:in_sample_size+i]
    x = df[syls[0]]
    y = df[syls[1]]
    spread = reg(x,y)
    mean = np.mean(spread)
    std = np.std(spread)
    if spread[-1] > mean + threshold*std:
        signal_x[i] = 1
        signal_y[i] = -1
    elif spread[-1] < mean - threshold*std:
        signal_x[i] = -1
        signal_y[i] = 1
    else:
        signal_x[i] = 0
        signal_y[i] = 0        

data_price = price[out_sample_size+1:]
data_signal = pd.DataFrame({syls[0]:signal_x, 
                            syls[1]:signal_y},index = data_price.index)

data_dict = {}
total_cash = 10000.0
for i in syls:
    data_dict[i] = pd.DataFrame({'price':data_price[i],
                                 'signal':data_signal[i],#, index = data_price[i].index) 
                                 'quantity':calculateQuantity(data_signal[syls[0]],data_price[syls[0]], total_cash/2)}, index = data_price[i].index)

profit_strategy = PortfolioNetProfit(data_dict).net_profit()
PortfolioNetProfit(data_dict).curve()
qb.AddEquity('SPY')
benchmark = qb.History('SPY',start,end, Resolution.Daily).loc['SPY']['close'][in_sample_size:]
profit_benchmark = benchmark*(total_cash/benchmark[0])
plt.plot(profit_benchmark, label = 'profit_benchmark')
plt.legend()

performance = pd.concat([profit_benchmark,profit_strategy],axis=1)[1:]
performance.columns = ['benchmark','equity']
stats = qb.GetPortfolioStatistics(performance)
stats

