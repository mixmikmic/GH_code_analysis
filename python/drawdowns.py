# Imports
get_ipython().run_line_magic('matplotlib', 'inline')
import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import backtesters
reload(backtesters)
from backtesters import MABacktester # use this to calculate drawdown table

# Get price data 
bitcoin = quandl.get("BCHAIN/MKPRU")
bitcoin.columns = ['Close']
bitcoin = bitcoin.shift(-1) # data set has daily open, we want daily close
sp500 = pd.read_csv('^GSPC.csv', index_col = 0, parse_dates = [0])

nasdaq_full = pd.read_csv('NASDAQCOM.csv', index_col = 0, parse_dates = [0])
nasdaq_full.columns = ['Close']
nasdaq_full.index.name = 'Date'
nasdaq_full[nasdaq_full['Close'] == '.'] = np.NAN
nasdaq_full['Close'] = nasdaq['Close'].astype(float)
nasdaq_full['Close'].fillna(method = 'ffill', inplace = True)

# Remove the 0's and start on same date
bitcoin = bitcoin.loc['2010-08-17':]
sp500 = sp500.loc['2010-08-17':]
nasdaq = nasdaq_full.loc['2010-08-17':].copy()

for df in [bitcoin, sp500, nasdaq]:
    max_so_far = np.maximum.accumulate(df['Close'])
    df['Drawdown%'] = (max_so_far - df['Close']) / max_so_far  * 100

for i in [5,10,20,30,40,50,60]:
    print i,
    for df in [bitcoin, sp500, nasdaq]:   
        print "{:,.1f}%".format(df[df['Drawdown%'] > i].size / float(df.size) * 100),
    print

# Comparison of % drawdown over time
#%matplotlib qt5
fig, ax = plt.subplots(figsize=(12,10))
plt.title('Drawdowns')
bitcoin['zero'] = 0
#plt.plot(bitcoin['Drawdown%'])
plt.fill_between(bitcoin['Drawdown%'].index,bitcoin['Drawdown%'].values,0, alpha = 0.3, label="Bitcoin")
plt.fill_between(sp500['Drawdown%'].index,sp500['Drawdown%'].values,0, alpha = 0.3, label = "SP500")
#plt.fill_between(nasdaq['Drawdown%'].index,nasdaq['Drawdown%'].values,0, alpha = 0.3)
plt.legend(loc= "lower center", frameon=False)
#bitcoin['Drawdown%'].plot(label = "Bitcoin", legend = True)
#sp500['Drawdown%'].plot(label = "S&P500", legend = True)
#nasdaq['Drawdown%'].plot(label = "Nasdaq", legend = True)
plt.yticks(np.arange(0,100,10))
vals = ax.get_yticks()
ax.set_yticklabels(['{:3.0f}%'.format(x) for x in vals])
ax.invert_yaxis() # turn up side down
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
fig.savefig("drawdowns.png", dpi=100, bbox_inches='tight')

bitcoin_drawdowns = MABacktester(bitcoin['Close']).drawdowns(target='market')
bitcoin_drawdowns

# plot each of the drawdowns to see the patterns

extra_days = 15 # add some extra days on either side

for index, row in bitcoin_drawdowns.iterrows():    
    #if index > 0:
    #    break
    if row['dd'] < 25: # drawdowns of 25% or more
        break
    high = row['highd']
    recovery = row['recoveryd']
    low = row['lowd']
    start = high - timedelta(days=extra_days)
    end = recovery + timedelta(days=extra_days)
    if row['rdays'].days > 100:
        end += timedelta(days=extra_days)
    if row['rdays'].days > 500:
        end += timedelta(days=extra_days * 4)    
    fig, ax = plt.subplots(figsize=(16,10))
    ax.plot(bitcoin['Close'].loc[start:end].index, bitcoin['Close'].loc[start:end])       
    text = "%.1f%% drop over %d days\nRecovery took %d days" % (row['dd'],row['days'].days,row['rdays'].days)
    if index == 3:
        text = "%.1f%% drop over %d days\nBeen %d days so far" % (row['dd'],row['days'].days,row['rdays'].days)
    ax.plot([high, low, recovery], [bitcoin['Close'].loc[high], bitcoin['Close'].loc[low], bitcoin['Close'].loc[recovery]], 'x', color='Red', markersize=8)
    ax.text(0.5, 0.5,text, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    fig.savefig("drawdowns/bitcoin/" + high.strftime("%Y-%m-%d") + '.png', dpi=100, bbox_inches='tight')
    plt.close(fig)

# Do same for Nasdaq
nasdaq_full.plot(figsize=(16,10));

nasdaq_drawdowns = MABacktester(nasdaq_full['Close']).drawdowns(target='market')
nasdaq_drawdowns

# plot each of the drawdowns to see the patterns

extra_days = 15 # add some extra days on either side

for index, row in nasdaq_drawdowns.iterrows():    
    if index > 0:
        break
    if row['dd'] < 25: # drawdowns of 25% or more
        break
    high = row['highd']
    recovery = row['recoveryd']
    low = row['lowd']
    start = high - timedelta(days=extra_days * row['rdays'].days / 25)
    end = recovery + timedelta(days=extra_days * row['rdays'].days / 25)
    fig, ax = plt.subplots(figsize=(16,10))
    ax.plot(nasdaq['Close'].loc[start:end].index, nasdaq['Close'].loc[start:end])       
    text = "%.1f%% drop over %d days\nRecovery took %d days" % (row['dd'],row['days'].days,row['rdays'].days)
    ax.plot([high, low, recovery], [nasdaq['Close'].loc[high], nasdaq['Close'].loc[low], nasdaq['Close'].loc[recovery]], 'x', color='Red', markersize=8)
    ax.text(0.5, 0.5,text, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    #fig.savefig("drawdowns/nasdaq/" + high.strftime("%Y-%m-%d") + '.png', dpi=100, bbox_inches='tight')
    #plt.close(fig)

bitcoin_drawdowns.sort_values('highd', ascending = True, inplace = True)
bitcoin_drawdowns.set_index('highd', inplace = True)

# on a log scale the sequence of lows in each succesive cycle are roughly a straight line (so exp growth)
# same for the sequence of highs
np.log(bitcoin_drawdowns[['low','high']]).plot(figsize=(12,10));

