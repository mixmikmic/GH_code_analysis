import numpy as np
import pandas as pd
from scipy.signal import argrelmin, argrelmax
import statsmodels.api as sm
import talib
import matplotlib.pyplot as plt
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import Latest
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.research import run_pipeline
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import CustomFactor

# check if entire column is NaN. If yes, return True
def nan_check(col):
    if np.isnan(np.sum(col)):
        return True
    else: 
        return False

# helper to calculate lag
def lag_helper(col):
    
    # TA-Lib raises an error if whole colum is NaN,
    # so we check if this is true and, if so, skip
    # the lag calculation
    if nan_check(col):
        return np.nan
    # 20-day simple moving average
    else:
        return talib.SMA(col, 20)[20:]

AAPL_frame = get_pricing('AAPL', start_date='2014-08-08', end_date='2015-08-08', fields='close_price')

# convert to np.array for helper function and save index of timeseries
AAPL_index = AAPL_frame.index
AAPL_frame = AAPL_frame.as_matrix()

# calculate lag
AAPL_frame_lagged = lag_helper(AAPL_frame)

plt.plot(AAPL_index, AAPL_frame, label='Close')
plt.plot(AAPL_index[20:], AAPL_frame_lagged, label='Lagged Close')
plt.legend(loc=2)
plt.xlabel('Date')
plt.title('Close Prices vs Close Prices (20-Day Lag)')
plt.ylabel('AAPL Price');

# Custom Factor 1 : Slope of 52-Week trendline
def trendline_function(col, support):
    
    # NaN check for speed
    if nan_check(col):
        return np.nan  
    
    # lag transformation
    col = lag_helper(col)
    
    # support trendline
    if support:
        
        # get local minima
        minima_index = argrelmin(col, order=5)[0]
        
        # make sure line can be drawn
        if len(minima_index) < 2:
            return np.nan
        else:
            # return gradient
            return (col[minima_index[-1]] - col[minima_index[0]]) / (minima_index[-1] - minima_index[0])
    
    # resistance trandline
    else:
        
        # get local maxima
        maxima_index = argrelmax(col, order=5)[0]
        if len(maxima_index) < 2:
            return np.nan
        else:
            return (col[maxima_index[-1]] - col[maxima_index[0]]) / (maxima_index[-1] - maxima_index[0])

# make the lagged frame the default 
AAPL_frame = AAPL_frame_lagged

# use day count rather than dates to ensure straight lines
days = list(range(0,len(AAPL_frame),1))

# get points to plot
points_low = [(101.5 + (trendline_function(AAPL_frame, True)*day)) for day in days]
points_high = [94 + (trendline_function(AAPL_frame, False)*day) for day in days]

# create graph
plt.plot(days, points_low, label='Support')
plt.plot(days, points_high, label='Resistance')
plt.plot(days, AAPL_frame, label='Lagged Closes')
plt.xlim([0, max(days)])
plt.xlabel('Days Elapsed')
plt.ylabel('AAPL Price')
plt.legend(loc=2);

def create_trendline_factor(support):
    
    class Trendline(CustomFactor):

        # 52 week + 20d lag
        window_length = 272
        inputs=[USEquityPricing.close]

        def compute(self, today, assets, out, close): 
            out[:] = np.apply_along_axis(trendline_function, 0, close, support)
    return Trendline
    
temp_pipe_1 = Pipeline()
trendline = create_trendline_factor(support=True)
temp_pipe_1.add(trendline(), 'Trendline')
results_1 = run_pipeline(temp_pipe_1, '2015-08-08', '2015-08-08')
results_1.head(20)

# Custom Factor 2 : % above 260 day low
def percent_helper(col):
    if nan_check(col):
        return np.nan 
    else:
        col = lag_helper(col)
        return (col[-1] - min(col)) / min(col)

print 'Percent above 260-day Low: %f%%' % (percent_helper(AAPL_frame) * 100)

# create the graph
plt.plot(days, AAPL_frame)
plt.axhline(min(AAPL_frame), color='r', label='260-Day Low')
plt.axhline(AAPL_frame[-1], color='y', label='Latest Price')
plt.fill_between(days, AAPL_frame)
plt.xlabel('Days Elapsed')
plt.ylabel('AAPL Price')
plt.xlim([0, max(days)])
plt.title('Percent Above 260-Day Low')
plt.legend();

class Percent_Above_Low(CustomFactor):
    
    # 260 days + 20 lag
    window_length = 280
    inputs=[USEquityPricing.close]
    
    def compute(self, today, asseys, out, close):
        out[:] = np.apply_along_axis(percent_helper, 0, close)

temp_pipe_2 = Pipeline()
temp_pipe_2.add(Percent_Above_Low(), 'Percent Above Low')
results_2 = run_pipeline(temp_pipe_2, '2015-08-08', '2015-08-08')
results_2.head(20)

# set 48-week average
av_52w = 100.

# create list of possible last four-week averages
av_4w = xrange(0,200)

# create list of oscillator values
osc = [(x / av_52w) - 1 for x in av_4w]

# draw graph
plt.plot(av_4w, osc)
plt.axvline(100, color='r', label='52-Week Average')
plt.xlabel('Four-Week Average')
plt.ylabel('4/52 Oscillator')
plt.legend();

# Custom Factor 3: 4/52 Price Oscillator
def oscillator_helper(col):
    if nan_check(col):
        return np.nan   
    else:
        col = lag_helper(col)
        return np.nanmean(col[-20:]) / np.nanmean(col) - 1

class Price_Oscillator(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 272
    
    def compute(self, today, assets, out, close):
        out[:] = np.apply_along_axis(oscillator_helper, 0, close)
        
temp_pipe_3 = Pipeline()
temp_pipe_3.add(Price_Oscillator(), 'Price Oscillator')
results_3 = run_pipeline(temp_pipe_3, '2015-08-08', '2015-08-08')
results_3.head(20)

# get two averages
av_4w = np.nanmean(AAPL_frame[-20:])
av_52w = np.nanmean(AAPL_frame)

# create the graph
plt.plot(days, AAPL_frame)
plt.fill_between(days[-20:], AAPL_frame[-20:])
plt.axhline(av_4w, color='y', label='Four-week Average' )
plt.axhline(av_52w, color='r', label='Year-long Average')
plt.ylim([80,140])
plt.xlabel('Days Elapsed')
plt.ylabel('AAPL Price')
plt.title('4/52 Week Oscillator')
plt.legend();

# create a new longer frame of AAPL close prices
AAPL_frame = get_pricing('AAPL', start_date='2002-08-08', end_date='2016-01-01', fields='close_price')

# use dates as index
AAPL_index = AAPL_frame.index[20:]
AAPL_frame = lag_helper(AAPL_frame.as_matrix())

# 1d returns
AAPL_1d_returns = ((AAPL_frame - np.roll(AAPL_frame, 1))/ np.roll(AAPL_frame,1))[1:]

# 1w returns
AAPL_1w_returns = ((AAPL_frame - np.roll(AAPL_frame, 5))/ np.roll(AAPL_frame, 5))[5:]

# 1m returns
AAPL_1m_returns = ((AAPL_frame - np.roll(AAPL_frame, 30))/ np.roll(AAPL_frame, 30))[30:]

# 39w returns
AAPL_39w_returns = ((AAPL_frame - np.roll(AAPL_frame, 215))/ np.roll(AAPL_frame, 215))[215:]

# plot close prices
plt.plot(AAPL_index[1:], AAPL_1d_returns, label='1-day Returns')
plt.plot(AAPL_index[5:], AAPL_1w_returns, label='1-week Returns')
plt.plot(AAPL_index[30:], AAPL_1m_returns, label='1-month Returns')
plt.plot(AAPL_index[215:], AAPL_39w_returns, label='39-week Returns')

# show events
# iPhone release
plt.axvline('2007-07-29')
# iPod mini 2nd gen. release
plt.axvline('2005-02-23')
# iPad release
plt.axvline('2010-04-03')
# iPhone 5 release
plt.axvline('2012-09-21')
# Apple Watch
plt.axvline('2015-04-24')

# labels
plt.xlabel('Days')
plt.ylabel('Returns')
plt.title('Returns')
plt.legend();

# Custom Fator 4: 39-week Returns
def return_helper(col):
    if nan_check(col):
        return np.nan 
    else:
        col = lag_helper(col)
        return (col[-1] - col[-215]) / col[-215]

class Return_39_Week(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 235
    
    def compute(self, today, assets, out, close):
        out[:] = np.apply_along_axis(return_helper, 0, close)
        
temp_pipe_4 = Pipeline()
temp_pipe_4.add(Return_39_Week(), '39 Week Return')
results_4 = run_pipeline(temp_pipe_4, '2015-08-08','2015-08-08')
results_4.head(20)

# This factor creates the synthetic S&P500
class SPY_proxy(CustomFactor):
    inputs = [morningstar.valuation.market_cap]
    window_length = 1
    
    def compute(self, today, assets, out, mc):
        out[:] = mc[-1]

# using helpers to boost speed
class Pricing_Pipe(CustomFactor):
    
    inputs = [USEquityPricing.close]
    outputs = ['trendline', 'percent', 'oscillator', 'returns']
    window_length=280
    
    def compute(self, today, assets, out, close):
        out.trendline[:] = np.apply_along_axis(trendline_function, 0, close[-272:], True)
        out.percent[:] = np.apply_along_axis(percent_helper, 0, close)
        out.oscillator[:] =  np.apply_along_axis(oscillator_helper, 0, close[-272:])
        out.returns[:] = np.apply_along_axis(return_helper, 0, close[-235:])
        
def Data_Pull():
    
    # create the piepline for the data pull
    Data_Pipe = Pipeline()
    
    # create SPY proxy
    Data_Pipe.add(SPY_proxy(), 'SPY Proxy')

    # run all on same dataset for speed
    trendline, percent, oscillator, returns = Pricing_Pipe()
    
    # add the calculated values
    Data_Pipe.add(trendline, 'Trendline')
    Data_Pipe.add(percent, 'Percent')
    Data_Pipe.add(oscillator, 'Oscillator')
    Data_Pipe.add(returns, 'Returns')
        
    return Data_Pipe

results = run_pipeline(Data_Pull(), '2015-08-08', '2015-08-08')
results.head(20)

# limit effect of outliers
def filter_fn(x):
    if x <= -10:
        x = -10.0
    elif x >= 10:
        x = 10.0
    return x   

# combine data
def aggregate_data(df):

    # basic clean of dataset to remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # need standardization params from synthetic S&P500
    df_SPY = df.sort(columns='SPY Proxy', ascending=False)

    # create separate dataframe for SPY
    # to store standardization values
    df_SPY = df_SPY.head(500)

    # get dataframes into numpy array
    df_SPY = df_SPY.as_matrix()

    # store index values
    index = df.index.values

    # get data intp a numpy array for speed
    df = df.as_matrix()

    # get one empty row on which to build standardized array
    df_standard = np.empty(df.shape[0])

    for col_SPY, col_full in zip(df_SPY.T, df.T):

        # summary stats for S&P500
        mu = np.mean(col_SPY)
        sigma = np.std(col_SPY)
        col_standard = np.array(((col_full - mu) / sigma))

        # create vectorized function (lambda equivalent)
        fltr = np.vectorize(filter_fn)
        col_standard = (fltr(col_standard))

        # make range between -10 and 10
        col_standard = (col_standard / df.shape[1])

        # attach calculated values as new row in df_standard
        df_standard = np.vstack((df_standard, col_standard))

    # get rid of first entry (empty scores)
    df_standard = np.delete(df_standard, 0, 0)

    # sum up transformed data
    df_composite = df_standard.sum(axis=0)

    # put into a pandas dataframe and connect numbers
    # to equities via reindexing
    df_composite = pd.Series(data=df_composite, index=index)

    # sort descending
    df_composite.sort(ascending=False)

    return df_composite

ranked_scores = aggregate_data(results)
ranked_scores

# histogram
ranked_scores.hist()

# baskets
plt.axvline(ranked_scores[26], color='r')
plt.axvline(ranked_scores[-6], color='r')
plt.xlabel('Ranked Scores')
plt.ylabel('Frequency')
plt.title('Histogram of Ranked Scores of Stock Universe');

