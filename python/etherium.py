# Imports
get_ipython().run_line_magic('matplotlib', 'inline')
import quandl
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import backtesters
reload(backtesters) # to make sure changes are reloaded
from backtesters import MABacktester # see code in backtesters module

# Get ETH prices for GDAX exchange
eth = quandl.get("GDAX/ETH_BTC") 
eth = eth.shift(1).dropna() # GDAX has daily OPEN, shift to get daily CLOSE
eth.columns = [u'Last', u'High', u'Low', u'Volume'] # rename columns

# Use same time periods and dates as medium article
training_cutoff_date = "2017-05-19" 
training_cutoff_i = eth.index.get_loc(training_cutoff_date)
eth = eth.loc[:'2018-05-13'] # data up to 2018-05-13

# Chart of ETH price in BTC
eth['Last'].plot(figsize=(14,10));

# We also need BTCUSD prices for some of the return comparisons
btc = quandl.get("BCHAIN/MKPRU") 
btc = btc.shift(-1) # data set has daily open, we want daily close
btc = btc.loc['2010-08-17':].fillna(method = 'ffill') 
btc.columns = ['Last'] # for consistency

# Test different MA lookbacks over the training period
ms_range = [1]
ml_range = np.arange(5,100)
returns = []

for ms in ms_range:
    for ml in ml_range:
        results = MABacktester(eth['Last'].iloc[:training_cutoff_i], ms=ms,ml=ml,ema=False, long_only=False).results() 
        returns.append((ml,np.round(results['Strategy'],1)))

# Visualise the results and compare with buy and hold return
r = np.array(returns)
plt.figure(figsize=(12,8))
plt.plot(r[:,0], r[:,1], label = "Strategy return at various lookbacks");
plt.hlines(results['Market'],0,110, color='r')
plt.text(70, results['Market']+5, 'Buy and hold return (%s%%)' % results['Market'], fontsize = 12, color = 'red')
plt.legend()
plt.title('Strategy returns over training period')
plt.xlabel('Lookback period')
plt.ylabel('Return');

# Find the top few
returns.sort(key = lambda x: x[1])
returns[::-1][:10]

# Choose 18 day lookback and view buys and sells over the training period
train = MABacktester(eth['Last'].iloc[:training_cutoff_i], ms=1,ml=18,ema=False, long_only=False)
print train
train.plot(figsize = (14,10))

# Training period results
print train
train.results()

# Test period results for same 18 day lookback
test = MABacktester(eth['Last'].iloc[training_cutoff_i:], ms=1,ml=18,ema=False, long_only=False)
print test
test.results()

# Full period results using 18 day lookback
full = MABacktester(eth['Last'], ms=1,ml=18,ema=False, long_only=False)
print full
full.results()

# Show buys and sells over the trst period
print test
test.plot(figsize = (14,10))

# Heatmap of monthly returns using MA strategy
full.plot_heatmap(target="strategy", figsize=(14,10))

# Heatmap pf monthy returns for buy and hold
full.plot_heatmap(target="market", figsize=(14,10))

# drawdowns for MA strategy
full.drawdowns(target='strategy',cutoff=20)

# drawdowns for buy and hold
full.drawdowns(target='market',cutoff=20)

# Plot equity curve of strategy versus market (buy and hold)
full.plot_equity_curve(figsize=(12,10))

# Comparison of all returns over the test period

start_i = training_cutoff_i
end_i = -1
start_date = eth.iloc[start_i].name
end_date = eth.iloc[end_i].name
ETHBTC_end = eth['Last'].iloc[end_i]
ETHBTC_start = eth['Last'].iloc[start_i]
BTCUSD_end = btc['Last'].loc[end_date]
BTCUSD_start = btc['Last'].loc[start_date]
BTC_strategy_at_end = test.results()['Strategy'] / 100 + 1
BTC_market_at_end  = test.results()['Market'] / 100 + 1

print "Start %s End %s" % (start_date, end_date)
print "-" * 35
print "Return of HODL ETH in USD %.2f%%" % ( ( (ETHBTC_end * BTCUSD_end) / (ETHBTC_start * BTCUSD_start) - 1) *100)
print "Return of HODL BTC in USD %.2f%%" % ((BTCUSD_end / BTCUSD_start - 1) *100)
print "Return of strategy in USD %.2f%%" % ((BTC_strategy_at_end * BTCUSD_end / BTCUSD_start - 1) *100)
print "-" * 35
print "Return of HODL eth in BTC %.2f%%" % ((ETHBTC_end / ETHBTC_start - 1) *100)
print "Cross check %.2f%%" % ((BTC_market_at_end * BTCUSD_end / BTCUSD_start - 1) *100)

# Statistical significance
# null hypothesis is that strategy is no different from buy and hold
# calculate prob of buy and hold achieving the same or better result
# try this using a distibution with the same moments
sims = 5000
np.random.seed(0) # want same random results each time
test_stat =  train.results()['Strategy']
end = []
for i in np.arange(sims):
    simulated = np.random.choice(train.market_ret().dropna(),training_cutoff_i,replace = True)
    end.append((np.exp(simulated.cumsum())[-1] - 1) * 100)

count = 0.0
for i in end:
    if i >= test_stat:
        count += 1
print "p value %.1f%%" % (count / len(end) * 100)

np.random.seed(0) # want same random results each time
dates = pd.date_range(start_date, periods = training_cutoff_i)
simulated = pd.Series(np.random.choice(train.market_ret(),training_cutoff_i,replace = True), index= dates)
simprices = simulated.cumsum().apply(np.exp)
plt.plot(simprices);

# Statistical significance
# null hypothesis is that strategy does not capture the autcorrelations so would just do as well on a similar shaped distn
# calculate prob of strategy applied to simulated results achieving the same or better return
np.random.seed(0) # want same random results each time
test_stat = train.results()['Strategy']
end = []
dates = pd.date_range(start_date, periods = training_cutoff_i)
for i in np.arange(sims):
    simulated = pd.Series(np.random.choice(train.market_ret(),training_cutoff_i,replace = True), index= dates)
    simprices = simulated.cumsum().apply(np.exp)
    simbacktest = MABacktester(simprices, ms = 1, ml = 18, long_only = False)
    end.append(simbacktest.results()['Strategy'])

count = 0.0
for i in end:
    if i >= test_stat:
        count += 1
print "p value %.1f%%" % (count / len(end) * 100)

# Statistical significance
# Choose the same number of longs and shorts but randomise where they occur
# What is the prob of this doing as well as the strategy did
np.random.seed(0) # want same random results each time
test_stat = train.results()['Strategy']
end = []
market = train.market_ret()
stance = train.stance()
for i in np.arange(sims):
    new_stance = pd.Series(np.random.choice(stance, size = len(stance), replace = False), index = stance.index)
    strategy = market * new_stance.shift(1) 
    ret = ((np.exp(strategy.cumsum()[-1]) - 1) * 100)
    end.append(ret)

count = 0.0
for i in end:
    if i >= test_stat:
        count += 1
print "p value %.1f%%" % (count / len(end) * 100)

