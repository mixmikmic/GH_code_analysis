import pandas as pd
import ccxt
import time
from timeit import default_timer as timer
from matplotlib import pyplot
from decimal import Decimal as D
import numpy as np
from tqdm import tqdm_notebook, tqdm
from tqdm import tnrange as trange

this_exchange = ccxt.poloniex()

if False:
    for i in POLO.__dict__:
        if '__' not in i:
            if 'fetch' in i:
                print(i)

ticker = this_exchange.fetchTickers()

x = list(ticker)[0]
print(x)

coins_not_working = ['LTC/BTC', 'STR/BTC']
coins_to_use = ['ETH/BTC', 'XRP/BTC', 'DASH/BTC' , 'XMR/BTC',
                'BTS/BTC', 'DOGE/BTC', 'FCT/BTC', 'MAID/BTC', 'CLAM/BTC']
coins_collected = []
datasets = []
min_len = 10000000
start_data_gen = timer()
amount_of_coins = len(ticker)
# for index, pair in enumerate(ticker):
for index in trange(amount_of_coins):
    pair = list(ticker)[index]
    if pair in coins_to_use:
        if pair not in coins_collected:
            print(pair)
            # SLEEP!! DO NOT GET BANNED!
            time.sleep (this_exchange.rateLimit / 400)
            start_call = timer()
            chart = this_exchange.fetchOhlcv(pair, 
                                             timeframe = '5m', # polo only offers the 5m
                                             since = int(1515974400)
                                             )
            df = pd.DataFrame(
                chart, 
                columns=['time','open','high','low','close','vol']
            )
            df.pair = pair
            this_len = len(df)
            if this_len < min_len:
                min_len = this_len
            tqdm.write('Elements in Dataset: {}'.format(this_len))
            tqdm.write('Downloading this coin took: {:.2f} secs'.format(timer()-start_call))
            datasets.append(df)
            coins_collected.append(pair)
print('Min length of all sets is {} samples'.format(min_len))
print('Collected All Datasets. Took {:.2f} mins'.format(float(timer()-start_data_gen)/60))        

# the end time in all datasets is the same... so take the min_len to the end
new_datasets = []
for d in datasets:
    pair = d.pair
    this_len = len(d)
    print(pair, this_len)
    if this_len == min_len:
        print('this is the lowest len df and not changing')
        time_of_start = d['time'][0]
        time_of_end = d['time'][len(d['time'])-1]
        new_datasets.append(d) 
    elif this_len > min_len:
        trim_section_start = len(d['time']) - min_len
        time_of_start = d['time'][trim_section_start]
        time_of_end = d['time'][len(d['time'])-1]
        df = d[trim_section_start:]
        df.pair = pair
        new_datasets.append(df)    
    print('Start of set {}'.format(time_of_start))
    print('end of set {}'.format(time_of_end))
    print('New set len {}'.format(len(df)))
    

for set_ in new_datasets:
    print(set_.pair, len(set_))

pyplot.figure()
i = 1
for set_ in new_datasets:
    pyplot.subplot(len(new_datasets), 3, i)
    pyplot.plot(set_['close'])
    pyplot.axis('off')
    pyplot.title(set_.pair, y=.75, loc='left')
    i += 1
pyplot.show()

# WHAT DO I WANT FOR COLUMNS!?!?!?!?!
# Each input will have 20 values
# input1 = [[last][vol][last -60 elements]]
# input2 = [[low][high][mean*]]
# input3 = [[bollinger high][bollinger low][mean**]]
# ###
# input4 = [[twitter sent][twittervol]]
# ###
# output1 = [time, time+1, time+2] # doesnt matter what the input all the outputs will be the same and averaged over

for set_ in new_datasets:
    print(set_.pair)

cols = []
for i in range(60):
    cols.append('close_{}'.format(i))
    cols.append('vol_{}'.format(i))
print(cols)

working_df = pd.DataFrame(columns=cols)

# input1 = [[last][vol][1]]
dataset = np.zeros((2, 60, len(new_datasets), len(new_datasets[0])))
for index1, set_ in enumerate(new_datasets):
    set_for_all_coins = []
    for index2 in range(len(set_)):
        set_of_elements = []
        if index2 <= len(set_) - 60:
            elements = []
            for i in range(60):
                close = set_.loc[i + index2]['close']
                vol = set_.loc[i + index2]['vol']
                elements.append(close)
                elements.append(vol)
            set_of_elements.append(elements)
        set_for_all_coins.append(set_of_elements)
    dataset.append(set_for_all_coins, index1)

fakeset = np.zeros((2, 60, 9, len(new_datasets[0]) - 60))

x = new_datasets[0].loc[200000]['close']

start_time = timer()
# num of max rows is the num of max elements in any dataset
set_len = len(new_datasets[0]) - 60
everything = []
for index in trange(3333,
                   ascii=True,
                            desc="OneRow",
                            # dynamic_ncols=True,
                            smoothing=0,
                            leave=False,
                            # unit_scale=True
                   ):
    # each row has all the coins
    coins = []
    """
    for coin_range in trange(len(new_datasets),
                            ascii=True,
                            desc="coin",
                            # dynamic_ncols=True,
                            smoothing=1,
                            leave=False,
                            # unit_scale=True
                            ): """
    for coin_range in range(len(new_datasets)):
        coin = new_datasets[coin_range]
        cols = coin.columns
        coin = pd.DataFrame(coin, columns=cols)
        # need 60 elements for each frame of the master time index
        elements = []
        """
        for element_range in trange(60,
                                   ascii=True,
                                    desc="inner",
                                    # dynamic_ncols=True,
                                    smoothing=1,
                                    leave=False,
                                    # unit_scale=True
                                   ): """
        for element_range in range(60):
            # get the 2 things we need.
            close = coin.loc[int(len(new_datasets[0])-int(index+element_range))-1]['close']
            vol = coin.loc[int(len(new_datasets[0])-int(index+element_range))-1]['vol']
            elements.append([close, vol])
        coins.append(elements)
        
    if index % 1000 == 0:
        tqdm.write('Completed {} of {} rows of data... LOLZ..'.format(index, set_len))
        tqdm.write('Wow right... this has been running for {:.2f} mins.'.format(float(timer()-start_time)/60))
    everything.append(coins)



x = np.array(everything)

print(x.shape)

filename = 'testset.npy'
np.save(filename, x)

y = np.load(filename)

print(x.shape, y.shape)

import tensorflow as tf

"""

PHONE DIED... BACK IN 5...

is that what you meant by backwards?????????????

.... getting more coffeee....

"""










df = new_datasets[2]

print(df.columns)

close = pd.Series(df['close'])

raw_values = close.values
print(raw_values)

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        # print(i - interval)
        value = D(dataset[i]) - D(dataset[i - interval])
        diff.append(value)
    return pd.Series(diff)

def inverse_difference(history, yhat, interval=1):
    return yhat + history[interval-1]

def undiff(dataset, diff, interval=1):
    the_return = list()
    for i, e in enumerate(diff):
        if i >= interval:
            last_val = dataset[i-interval]
            next_val = D(last_val) + D(diff[i])
            # assert next_val == dataset[i]
            the_return.append(next_val)
    return pd.Series(the_return)

diff_series = difference(close, 1)

fixed_series = undiff(close, diff_series, 1)

print(fixed_series.head())

inverted = list()
for i in range(len(diff_series)):
    value = inverse_difference(close, diff_series[i], len(diff_series)-i)
    inverted.append(value)
inverted = pd.Series(inverted)

print(close.head())
print(diff_series.head())
print(inverted.head())



