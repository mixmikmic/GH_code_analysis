# Imports
get_ipython().run_line_magic('matplotlib', 'inline')
import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import backtesters
reload(backtesters)
from backtesters import MABacktester # use this to calculate drawdown table

# Get price data 
bitcoin = quandl.get("BCHAIN/MKPRU")
bitcoin.columns = ['Close']
bitcoin = bitcoin.shift(-1) # data set has daily open, we want daily close
bitcoin = bitcoin.loc['2010-08-17':] # remove the 0's

# Get hashrate data
hashrate = pd.read_csv('data/btc_hash_rate.csv', index_col = 0, parse_dates = [0])

sns.set_style(style='dark')
hashrate.loc['2018'].plot(figsize=(14,10));

bitcoin_drawdowns = MABacktester(bitcoin['Close']).drawdowns(target='market')
bitcoin_drawdowns

def draw_chart(i, label_pos = [0.5, 0.5], save = False):    
    row = bitcoin_drawdowns.iloc[i]
    extra_days = 15 # add some extra days on either side
    high = row['highd']
    recovery = row['recoveryd']
    low = row['lowd']
    start = high - timedelta(days=extra_days)
    end = low + timedelta(days=extra_days)
    if row['rdays'].days > 100:
        end += timedelta(days=extra_days)
    if row['rdays'].days > 500:
        end += timedelta(days=extra_days * 4)
    if i == 3:
        end = '2018-06-12'
    fig, ax = plt.subplots(figsize=(16,10))
    ax.plot(bitcoin['Close'].loc[start:end].index, bitcoin['Close'].loc[start:end], label='Price')       
    text = "%.1f%% drop over %d days\nRecovery took %d days" % (row['dd'],row['days'].days,row['rdays'].days)
    if i == 3:
        text = "%.1f%% drop over %d days\nBeen %d days so far" % (row['dd'],row['days'].days,row['rdays'].days)
    ax.plot([high, low], [bitcoin['Close'].loc[high], bitcoin['Close'].loc[low]], 'x', color='Red', markersize=12)
    ax.legend(['Price'], loc='upper left')
    ax.text(label_pos[0], label_pos[1],text, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, size=14)
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(hashrate.loc[start:end], color='white', alpha=0.8, label='Hash Rate')
    ax.set_ylabel('Price', color='C0', size=12)
    ax2.set_ylabel('Hash Rate', color='gray', size = 12)
    ax2.legend(['Hash Rate'])
    plt.title('Bitcoin Price and Hash Rate', size = 18)
    plt.show()
    if save:
        fig.savefig("drawdowns/hashrate/" + high.strftime("%Y-%m-%d") + '.png', dpi=100, bbox_inches='tight')
    plt.close(fig)

draw_chart(1,[0.5,0.8], save = True)

# plot each of the drawdowns to see the patterns
# 1 tera hash = 1 trillion hashes / sec
for i in bitcoin_drawdowns:    
    draw_chart(i, save-True)

bitcoin_drawdowns.sort_values('highd', ascending = True, inplace = True)
bitcoin_drawdowns.set_index('highd', inplace = True)

# on a log scale the sequence of lows in each succesive cycle are roughly a straight line (so exp growth)
# same for the sequence of highs
np.log(bitcoin_drawdowns[['low','high']]).plot(figsize=(12,10));

