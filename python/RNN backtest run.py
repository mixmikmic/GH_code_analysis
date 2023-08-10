get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from weekly_buysell_rnn import WeeklyRebalanceRNN

DATA = pd.read_hdf("nasdaq_split-adjusted.hdf5")

DATA

stocks = ["AAPL","AMZN","EA","EBAY", "INTC", "GOOGL","NFLX", "NVDA", "REGN" ,"XRAY"]
stocks.sort()
dta = DATA.loc[stocks,:,:]

strategy = WeeklyRebalanceRNN(dta) #64-32-16
strategy.backtest()

strategy.plot()

strategy.weights[-5:]

strategy.calculate_annual_returns()

