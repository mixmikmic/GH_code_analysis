# Import libraries we will use
from matplotlib import pyplot
import datetime
import time
from pykalman import KalmanFilter
import numpy
import scipy
import pandas
import math

# grab the prattle dataset
prattle_data = local_csv('prattle.csv')
# filter to data from the fed only
fed_data = prattle_data[prattle_data.bank=='frc']

# helper functions

def convert_date(mydate):
    # converts string representing date to datetime object
    return datetime.datetime.strptime(mydate, "%Y-%m-%d")

# for grabbing dates and prices for a relevant equity
def get_data(etf_name,trading_start,trading_end='2015-07-20'):
    # using today as a default arg: assuming most of the ETFs I want to inspect
    # are still trading.
    stock_data = get_pricing(etf_name,
                        start_date = trading_start,
                        end_date = trading_end,
                        fields = ['close_price'],
                        frequency = 'daily')
    stock_data['date'] = stock_data.index
    # drop nans. For whatever reason, nans were causing the kf to return a nan array.
    stock_data = stock_data.dropna()
    # the dates are just those on which the prices were recorded
    dates = stock_data['date']
    dates = [convert_date(str(x)[:10]) for x in dates]
    prices = stock_data['close_price']
    return dates, prices

# for grabbing the prattle data on which the etf has been trading.
def get_prattle(trading_start,trading_end='2015-07-20'):
    # filter down to the relevant time period
    data_prattle = fed_data[fed_data.date > trading_start]
    data_prattle = data_prattle[trading_end > data_prattle.date]
    dates_prattle = data_prattle['date']
    dates_prattle = [convert_date(str(x)[:10]) for x in dates_prattle]
    scores_prattle = data_prattle['score']
    return dates_prattle, scores_prattle

# Initialize a Kalman Filter.
# Using kf to filter does not change the values of kf, so we don't need to ever reinitialize it.
kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 0,
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)

# Look at the Fed Data since 1999-01-01.
dates_fed_1999, scores_fed_1999 = get_prattle('1999-01-01')

# Filter Rolling Means
scores_fed_1999_means, _ = kf.filter(scores_fed_1999.values)

# Overlay Plots
fig, ax1 = pyplot.subplots()
# Use a scatterplot instead of a line plot because a line plot would be far too noisy.
ax1.scatter(dates_fed_1999,scores_fed_1999,c='gray',label='Fed Sentiment')
pyplot.xlabel('Date')
pyplot.ylabel('Fed Sentiment')
pyplot.ylim([-3,3])
pyplot.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(dates_fed_1999,scores_fed_1999_means, c='red', label='Fed Sentiment MA')
pyplot.ylabel('Fed Sentiment MA')
pyplot.ylim([-3,3])
pyplot.legend(loc=1)

pyplot.show()

def roll_volatility(timeseries,n):
    # skip the first n/7 datapoints, since we're looking back over n days.
    skip = int(round(n/7)+1)
    out = [None]*skip
    sk = skip+1
    for i in range(sk,len(timeseries)+1):
        sub_series = timeseries[i-sk:i]
        std = numpy.std(sub_series)
        out.append(std)
    return out

fed_vol_1999_90 = roll_volatility(scores_fed_1999,90)
fed_vol_1999_180 = roll_volatility(scores_fed_1999,180)

# let's plot it!
pyplot.plot(dates_fed_1999,fed_vol_1999_90,label='90 Day Window')
pyplot.plot(dates_fed_1999,fed_vol_1999_180,c='black',label='180 Day Window')
pyplot.xlabel('Date')
pyplot.ylabel('Fed Sentiment Volatility')
pyplot.legend()
pyplot.show()

# grab the SPY dataset and Kalman-filter it
dates_spy, prices_spy = get_data('SPY', '2002-03-03')
prices_spy_means, _ = kf.filter(prices_spy.values)

# First, we need some helper functions.

def add_days(s,x):
    """ takes a date in string format and adds exactly n days to it"""
    end = convert_date(s) + datetime.timedelta(days=x)
    return end.strftime('%Y-%m-%d')

def compute_volatility(opens,closes):
    """ calculates volatility as per 
        https://en.wikipedia.org/wiki/Volatility_(finance)#Mathematical_definition """
    # first: calculate the daily logarithmic rate of return
    daily_log_returns = [math.log(opens[x]/closes[x]) for x in range(len(opens))]
    # then get the standard deviation
    sigma_period = numpy.std(daily_log_returns)
    # now adjust to time period
    volatility = sigma_period / math.sqrt(len(opens))
    return volatility

def get_volatility(stock,n,today):
    """grabs the volatility over timespan of n days"""
    stock_data = get_pricing(stock, 
                             start_date = add_days(today, -n), 
                             end_date = today,
                             fields = ['open_price', 'close_price'],
                             frequency = 'daily')
    volatility = compute_volatility(stock_data.open_price, stock_data.close_price)
    return volatility
    
def etf_rolling_volatility(stock,n,dates,start_date):
    out = []
    nthday = convert_date(start_date)+ datetime.timedelta(days=n)
    for index,day in enumerate(list(dates)):
        # skip first n datapoints
        if day + datetime.timedelta(days=-1) < nthday:
            out.append(None)
        else:
            # adding days again is kind of an ugly (inefficient) solution
            dd = add_days(start_date,index)
            vol = get_volatility(stock,n,dd)
            out.append(vol)
    return out

# calculate the rolling volatilities for SPY
spy_vol_90 = etf_rolling_volatility('SPY',90,dates_spy,'2002-03-03')
spy_vol_180 = etf_rolling_volatility('SPY',180,dates_spy,'2002-03-03')

# let's plot it!
pyplot.plot(dates_spy,spy_vol_90,label='90 Day Window')
pyplot.plot(dates_spy,spy_vol_180,c='black',label='180 Day Window')
pyplot.xlabel('Date')
pyplot.ylabel('SPY Volatility')
pyplot.legend()
pyplot.show()

# first: get data for the right time period
dates_fed_2002, scores_fed_2002 = get_prattle('2002-03-03')

fed_vol_2002_90 = roll_volatility(scores_fed_2002,90)
fed_vol_2002_180 = roll_volatility(scores_fed_2002,180)

fig, ax1 = pyplot.subplots()
ax1.plot(dates_spy,spy_vol_90, c='green',label='SPY Volatility')
pyplot.xlabel('Date')
pyplot.ylabel('SPY Volatility')
pyplot.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(dates_fed_2002, fed_vol_2002_90, c='black', label='Fed Sentiment Volatility')
pyplot.ylabel('Fed Sentiment Volatility')
pyplot.title('90 Day Window')
pyplot.legend(loc=1)

pyplot.show()

fig, ax1 = pyplot.subplots()
ax1.plot(dates_spy,spy_vol_180, c='green',label='SPY Volatility')
pyplot.xlabel('Date')
pyplot.ylabel('SPY Volatility')
pyplot.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(dates_fed_2002, fed_vol_2002_180, c='black', label='Fed Sentiment Volatility')
pyplot.ylabel('Fed Sentiment Volatility')
pyplot.title('180 Day Window')
pyplot.legend(loc=1)

pyplot.show()

# First, we've got to reduce SPY from weekly to daily.
# The fed measurement is always taken on a sunday, so matching up the two datasets exactly 
# is somewhat laborious. We'll just take every 5th datapoint from SPY so we get one datapoint per week.

spy_vol_90_weekly = spy_vol_90[::5]
dates_spy_weekly = dates_spy[::5]

spy_vol_180_weekly = spy_vol_180[::5]

# Noting that the Fed dataset ends in January 2015, the spy_vol_x_weekly dataset has length 674 and
# the fed_vol_x_02 dataset has length 637. We truncate the spy_vol dataset at the end.

compare_spy_90 = spy_vol_90_weekly[:637]
compare_spy_180 = spy_vol_180_weekly[:637]

print scipy.stats.spearmanr(compare_spy_90,fed_vol_2002_90)
print scipy.stats.spearmanr(compare_spy_180,fed_vol_2002_180)
# printouts are in form (correlation coefficient, p-value)

# grab the DIA dataset.
dates_dia, prices_dia = get_data('DIA', '2002-03-03')
dia_vol_90 = etf_rolling_volatility('DIA',90,dates_dia,'2002-03-03')

# Overlay Plots
fig, ax1 = pyplot.subplots()
ax1.plot(dates_spy,prices_spy, c='green',label='SPY Price')
pyplot.xlabel('Date')
pyplot.ylabel('SPY Price')
pyplot.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(dates_dia, prices_dia, c='black', label='DIA Price')
pyplot.ylabel('DIA Price')
pyplot.legend(loc=1)

pyplot.show()

fig, ax1 = pyplot.subplots()
ax1.plot(dates_spy,spy_vol_90, c='green',label='SPY Volatility')
pyplot.xlabel('Date')
pyplot.ylabel('SPY Volatility')
pyplot.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(dates_dia, dia_vol_90, c='black', label='DIA Volatility')
pyplot.ylabel('DIA Volatility')
pyplot.legend(loc=1)

pyplot.show()

# grab the GLD dataset and relevant prattle data.
dates_gld, prices_gld = get_data('GLD', '2004-11-18')
dates_gld_prattle, scores_gld_prattle = get_prattle('2004-11-18')

# Fed sentiment volatility for the relevant time period
gld_prattle_vol_90 = roll_volatility(scores_gld_prattle,90)
gld_prattle_vol_180 = roll_volatility(scores_gld_prattle,180)

# Gold volatility for the relevant time period
gld_vol_90 = etf_rolling_volatility('GLD',90,dates_gld,'2004-11-18')
gld_vol_180 = etf_rolling_volatility('GLD',180,dates_gld,'2004-11-18')

# Let's plot the volatilities for GLD
pyplot.plot(dates_gld,gld_vol_90,label='90 Day Window')
pyplot.plot(dates_gld,gld_vol_180,c='black',label='180 Day Window')
pyplot.xlabel('Date')
pyplot.ylabel('GLD Volatility')
pyplot.legend()
pyplot.show()

fig, ax1 = pyplot.subplots()
ax1.plot(dates_gld,gld_vol_90, c='green',label='GLD Volatility')
pyplot.xlabel('Days Elapsed (Actual Span: November 2004 to July 2015)')
pyplot.ylabel('GLD Volatility')
pyplot.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(dates_gld_prattle, gld_prattle_vol_90, c='black', label='Fed Sentiment Volatility')
pyplot.ylabel('Fed Sentiment Volatility')
pyplot.title('90 Day Window')
pyplot.legend(loc=1)

pyplot.show()

fig, ax1 = pyplot.subplots()
ax1.plot(dates_gld,gld_vol_180, c='green',label='GLD Volatility')
pyplot.xlabel('Days Elapsed (Actual Span: November 2004 to July 2015)')
pyplot.ylabel('GLD Volatility')
pyplot.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(dates_gld_prattle, gld_prattle_vol_180, c='black', label='Fed Sentiment Volatility')
pyplot.ylabel('Fed Sentiment Volatility')
pyplot.title('180 Day Window')
pyplot.legend(loc=1)

pyplot.show()

# grab the TLT dataset and relevant prattle data.
dates_tlt, prices_tlt = get_data('TLT', '2002-06-24')
dates_tlt_prattle, scores_tlt_prattle = get_prattle('2002-06-24')

# Fed sentiment volatility for the relevant time period
tlt_prattle_vol_90 = roll_volatility(scores_tlt_prattle,90)
tlt_prattle_vol_180 = roll_volatility(scores_tlt_prattle,180)

# TLT volatility for the relevant time period
tlt_vol_90 = etf_rolling_volatility('TLT',90,dates_tlt,'2002-06-24')
tlt_vol_180 = etf_rolling_volatility('TLT',180,dates_tlt,'2002-06-24')

# Let's plot the volatilities for TLT
pyplot.plot(dates_tlt,tlt_vol_90,label='90 Day Window')
pyplot.plot(dates_tlt,tlt_vol_180,c='black',label='180 Day Window')
pyplot.xlabel('Date')
pyplot.ylabel('TLT Volatility')
pyplot.legend()
pyplot.show()

fig, ax1 = pyplot.subplots()
ax1.plot(dates_tlt,tlt_vol_90, c='green',label='TLT Volatility')
pyplot.xlabel('Date')
pyplot.ylabel('TLT Volatility')
pyplot.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(dates_tlt_prattle, tlt_prattle_vol_90, c='black', label='Fed Sentiment Volatility')
pyplot.ylabel('Fed Sentiment Volatility')
pyplot.title('90 Day Window')
pyplot.legend(loc=1)

pyplot.show()

fig, ax1 = pyplot.subplots()
ax1.plot(dates_tlt,tlt_vol_180, c='green',label='TLT Volatility')
pyplot.xlabel('Date')
pyplot.ylabel('TLT Volatility')
pyplot.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(dates_tlt_prattle, tlt_prattle_vol_180, c='black', label='Fed Sentiment Volatility')
pyplot.ylabel('Fed Sentiment Volatility')
pyplot.title('180 Day Window')
pyplot.legend(loc=1)

pyplot.show()

tlt_vol_90_weekly = tlt_vol_90[::5]
dates_tlt_weekly = dates_tlt[::5]
tlt_vol_180_weekly = tlt_vol_180[::5]

compare_tlt_90 = tlt_vol_90_weekly[:637]
compare_tlt_180 = tlt_vol_180_weekly[:637]

print scipy.stats.spearmanr(compare_tlt_90,fed_vol_2002_90)
print scipy.stats.spearmanr(compare_tlt_180,fed_vol_2002_180)
# printouts are in form (correlation coefficient, p-value)

def find_offset(ts1,ts2,window):
    """ Finds the offset between two equal-length timeseries that maximizies correlation. 
        Window is # of days by which we want to left- or right-shift.
        N.B. You'll have to adjust the function for negative correlations."""
    l = len(ts1)
    if l!=len(ts2):
        raise Exception("Error! Timeseries lengths not equal!")
    max_i_spearman = -1000
    max_spearman = -1000
    spear_offsets = []
    
    # we try all possible offsets from -window to +window.
    # we record the spearman correlation for each offset.
    for i in range(window,0,-1):
        series1 = ts1[i:]
        series2 = ts2[:l-i]
        # spearmanr is a correlation test
        spear = scipy.stats.spearmanr(series1,series2)[0]
        spear_offsets.append(spear)
        
        if spear > max_spearman:
            # update best correlation
            max_spearman = spear
            max_i_spearman = -i

    for i in range(0,window):
        series1 = ts1[:l-i]
        series2 = ts2[i:]
        spear = scipy.stats.spearmanr(series1,series2)[0]
        spear_offsets.append(spear)
        if spear > max_spearman:
            max_spearman = spear
            max_i_spearman = i

    print "Max Spearman:", max_spearman, " At offset: ", max_i_spearman
    pyplot.plot(range(-window,window),spear_offsets, c='green', label='Spearman Correlation')
    pyplot.xlabel('Offset Size (Number of Business Days)')
    pyplot.ylabel('Spearman Correlation')
    pyplot.legend(loc=3)
    pyplot.show()

print "TLT 90 Day"
find_offset(compare_tlt_90,fed_vol_2002_90,150)
print "TLT 180 Day"
find_offset(compare_tlt_90,fed_vol_2002_180,150)

