get_ipython().magic('pylab inline --no-import-all')
from __future__ import division

import pandas as pd
import seaborn as sns
import numpy as np

import time
import datetime
from datetime import datetime

import pytz
utc=pytz.UTC

import pyfolio as pf

figsize(13, 9)

SPY = pf.utils.get_symbol_rets('SPY')
GLD = pf.utils.get_symbol_rets('GLD')
TLT = pf.utils.get_symbol_rets('TLT')
EEM = pf.utils.get_symbol_rets('EEM')
FXE = pf.utils.get_symbol_rets('FXE')
SPY.name = 'SPY'
GLD.name = 'GLD'
TLT.name = 'TLT'
EEM.name = 'EEM'
FXE.name = 'FXE'
stocks_str = ['SPY', 'GLD', 'TLT', 'EEM' , 'FXE']
stocks = [SPY, GLD, TLT, EEM, FXE]
stocks_df_na = pd.DataFrame(stocks).T
stocks_df_na.columns = stocks_str
stocks_df = stocks_df_na.dropna()

# USAGE: Equal-Weight Portfolio.
# 1) if 'exclude_non_overlapping=True' below, the portfolio will only contains 
#    days which are available across all of the algo return timeseries.
# 
#    if 'exclude_non_overlapping=False' then the portfolio returned will span from the
#    earliest startdate of any algo, thru the latest enddate of any algo.
#
# 2) Weight of each algo will always be 1/N where N is the total number of algos passed to the function

portfolio_rets_ts, data_df = pf.timeseries.portfolio_returns_metric_weighted([SPY, FXE, GLD],
                                                                             exclude_non_overlapping=True
                                                                            )
to_plot = ['SPY', 'GLD', 'FXE'] + ["port_ret"]
data_df[to_plot].apply(pf.timeseries.cum_returns).plot()

pf.timeseries.perf_stats(data_df['port_ret'])

# USAGE: Portfolio based on volatility weighting.
# The higher the volatility the _less_ weight the algo gets in the portfolio
# The portfolio is rebalanced monthly. For quarterly reblancing, set portfolio_rebalance_rule='Q'

stocks_port, data_df = pf.timeseries.portfolio_returns_metric_weighted([SPY, FXE, GLD], 
                                                                       weight_function=np.std, 
                                                                       weight_function_window=126, 
                                                                       inverse_weight=True
                                                                      )
to_plot = ['SPY', 'GLD', 'FXE'] + ["port_ret"]
data_df[to_plot].apply(pf.timeseries.cum_returns).plot()

pf.timeseries.perf_stats(data_df['port_ret'])

stocks_port, data_df = pf.timeseries.portfolio_returns_metric_weighted([SPY, FXE, GLD], 
                                                                       weight_function=np.std,
                                                                       weight_func_transform=pf.timeseries.min_max_vol_bounds,
                                                                       weight_function_window=126, 
                                                                       inverse_weight=True)
to_plot = ['SPY', 'GLD', 'FXE'] + ["port_ret"]
data_df[to_plot].apply(pf.timeseries.cum_returns).plot()

pf.timeseries.perf_stats(data_df['port_ret'])

stocks_port, data_df = pf.timeseries.portfolio_returns_metric_weighted([SPY, FXE, GLD], 
                                                                       weight_function=np.std,
                                                                       weight_func_transform=pf.timeseries.bucket_std,
                                                                       weight_function_window=126, 
                                                                       inverse_weight=True)
to_plot = ['SPY', 'GLD', 'FXE'] + ["port_ret"]
data_df[to_plot].apply(pf.timeseries.cum_returns).plot()

pf.timeseries.perf_stats(data_df['port_ret'])

