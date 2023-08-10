# Import Zipline, the open source backester, and a few other libraries that we will use
import zipline
from zipline import TradingAlgorithm
from zipline.api import order_target, record, symbol, history

import pytz
from datetime import datetime
import matplotlib.pyplot as pyplot
import numpy as np

# Load data from get_trades for AAPL
data = get_pricing(
    ['AAPL'],
    start_date='2002-01-01',
    end_date = '2015-02-15',
    frequency='daily'
)
data.price.plot(use_index=False)

# Define the algorithm - this should look familiar from the Quantopian IDE
# For more information on writing algorithms for Quantopian
# and these functions, see https://www.quantopian.com/help

def initialize(context):
    context.i = 0
    context.aapl = symbol('AAPL')

def handle_data(context, data):
    # Skip first 300 days to get full windows
    context.i += 1
    if context.i < 300:
        return

    # Compute averages
    # history() has to be called with the same params
    # from above and returns a pandas dataframe.
    short_mavg = history(100, '1d', 'price').mean()
    long_mavg = history(300, '1d', 'price').mean()

    # Trading logic
    if short_mavg[context.aapl] > long_mavg[context.aapl]:
        # order_target orders as many shares as needed to
        # achieve the desired number of shares.
        order_target(context.aapl, 100)
    elif short_mavg[context.aapl] < long_mavg[context.aapl]:
        order_target(context.aapl, 0)

    # Save values for later inspection
    record(AAPL=data[context.aapl].price,
           short_mavg=short_mavg[context.aapl],
           long_mavg=long_mavg[context.aapl])

# Analyze is a post-hoc analysis method available on Zipline. 
# It accepts the context object and 'perf' which is the output 
# of a Zipline backtest.  This API is currently experimental, 
# and will likely change before release.

def analyze(context, perf):
    fig = pyplot.figure()
    
    # Make a subplot for portfolio value.
    ax1 = fig.add_subplot(211)
    perf.portfolio_value.plot(ax=ax1, figsize=(16,12))
    ax1.set_ylabel('portfolio value in $')

    # Make another subplot showing our trades.
    ax2 = fig.add_subplot(212)
    perf['AAPL'].plot(ax=ax2, figsize=(16, 12))
    perf[['short_mavg', 'long_mavg']].plot(ax=ax2)

    perf_trans = perf.ix[[t != [] for t in perf.transactions]]
    buys = perf_trans.ix[[t[0]['amount'] > 0 for t in perf_trans.transactions]]
    sells = perf_trans.ix[
        [t[0]['amount'] < 0 for t in perf_trans.transactions]]

    # Add buy/sell markers to the second plot
    ax2.plot(buys.index, perf.short_mavg.ix[buys.index],
             '^', markersize=10, color='m')
    ax2.plot(sells.index, perf.short_mavg.ix[sells.index],
             'v', markersize=10, color='k')
    
    # Set figure metadata
    ax2.set_ylabel('price in $')
    pyplot.legend(loc=0)
    pyplot.show()

# NOTE: This cell will take a few minutes to run.

# Create algorithm object passing in initialize and
# handle_data functions
algo_obj = TradingAlgorithm(
    initialize=initialize, 
    handle_data=handle_data
)

# HACK: Analyze isn't supported by the parameter-based API, so
# tack it directly onto the object.
algo_obj._analyze = analyze

# Run algorithm
perf_manual = algo_obj.run(data.transpose(2,1,0))

