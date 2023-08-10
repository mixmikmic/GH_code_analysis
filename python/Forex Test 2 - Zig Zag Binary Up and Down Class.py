from zigzag import peak_valley_pivots, max_drawdown, compute_segment_returns, pivots_to_modes
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd

target_quote = 'EUR/GBP' # TARGET VARIABLE
# sample interval is of 30 seconds
df = pd.read_pickle('real_time_quotes.pandas')  
df = df.set_index(pd.DatetimeIndex(df['time']))
del df['time']
#df.resample?

# resample downsampling to 5T samples or 5*30s = 2:30
# 30S segundo, 1 T minuto
# ........................ 1T
#df = df.resample('30S', label='right', closed='right').pad()

df.head(4)

X = pd.DataFrame(df[target_quote]) # TARGET VARIABLE
f = plt.figure(figsize=(15,4))
X[target_quote].plot()

def create_binary_zig_zag_class(X, target_quote, pts_variation, plot=True):
    pivots = peak_valley_pivots(X[target_quote].values, pts_variation, -pts_variation)
    spivots = pd.Series(pivots, index=X.index)
    spivots = spivots[pivots != 0] # just the pivots point, when it changes to up or down
    X['pivots'] = pd.Series(pivots, index=X.index) # adding a collum
    
    if plot:
        f, axr = plt.subplots(2, sharex=True, figsize=(15,7))
        f.subplots_adjust(hspace=0)
        X[target_quote].plot(ax=axr[0])
        X.loc[spivots.index, target_quote].plot(ax=axr[0], style='.-')
        
    # make class of up (1) or down (0) trends from the zigzap indicator
    for i in range(len(spivots)-1):
        X.loc[ (X.index >= spivots.index[i]) & 
              (X.index < spivots.index[i+1]), 'du' ] = X['pivots'][spivots.index[i]]
    X.iat[-1, 2] = X.iat[-2, 2] # replicate in the end ups or downs
    X['du'] = -X['du']
    X.loc[:, 'du'].where( X['du'] > 0, 0, inplace=True) # where is bigger than zero dont change
    # 1 is up trend, 0 is down trend
    
    if plot:
        X['du'].plot(ax=axr[1])
        plt.ylim(-0.15, 1.15)
        plt.ylabel('up=1 down=0')
    
    del X['pivots']

create_binary_zig_zag_class(X, target_quote, 0.00006, plot=True)

def make_shift_binary_class(X, shift, plot=True):

    if plot:
        f, axr = plt.subplots(2, sharex=True, figsize=(15,4))
        X['du'][-40:].plot(style='.-k', ax=axr[0])
        axr[0].set_ylim(-0.15, 1.15)
    
    X['dup'] = X['du'].shift(-shift) #shift = 1 # shift -1 sample to the left
    
    if plot:
        X['dup'][-40:].plot(style='.-k', ax=axr[1])
        axr[1].set_ylim(-0.15, 1.15)
    
    del X['du']

make_shift_binary_class(X, 2)

X.tail(15)

X[target_quote+' '+'dup'] = X['dup']

X = X['dup']

X.to_pickle('target_variable_dup.pandas')



