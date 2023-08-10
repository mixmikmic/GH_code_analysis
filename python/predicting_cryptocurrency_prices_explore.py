import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np

# get market info for bitcoin from the start of 2016 to the current day
bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]

# convert the date string to the correct date format
bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))

# when Volume is equal to '-' convert it to 0
bitcoin_market_info.loc[bitcoin_market_info['Volume']=="-",'Volume']=0

# convert to int
bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('int64')

# look at the first few rows
bitcoin_market_info.head()

# get market info for ethereum from the start of 2016 to the current day
eth_market_info = pd.read_html("https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
# convert the date string to the correct date format
eth_market_info = eth_market_info.assign(Date=pd.to_datetime(eth_market_info['Date']))
# look at the first few rows
eth_market_info.head()

# getting the Bitcoin and Eth logos
import sys
from PIL import Image
import io

import urllib2 as urllib
bt_img = urllib.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")
eth_img = urllib.urlopen("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/256px-Ethereum_logo_2014.svg.png")

image_file = io.BytesIO(bt_img.read())
bitcoin_im = Image.open(image_file)

image_file = io.BytesIO(eth_img.read())
eth_im = Image.open(image_file)
width_eth_im , height_eth_im  = eth_im.size
eth_im = eth_im.resize((int(eth_im.size[0]*0.8), int(eth_im.size[1]*0.8)), Image.ANTIALIAS)

# rename columns to differentiate the column names for two coins
bitcoin_market_info.columns =[bitcoin_market_info.columns[0]]+['bt_'+i for i in bitcoin_market_info.columns[1:]]
eth_market_info.columns =[eth_market_info.columns[0]]+['eth_'+i for i in eth_market_info.columns[1:]]

def plot_coin_info(coin_market_info, prefix, logo_img, img_x, img_y):
    sns.set()
    fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]}, sharex=True, figsize=(12, 9))
    ax1.set_ylabel('Closing Price ($)',fontsize=12)
    ax2.set_ylabel('Volume ($ bn)',fontsize=12)
    ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
    ax1.set_xticklabels('')
    ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
    ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
    ax1.plot(coin_market_info['Date'].astype(datetime.datetime), coin_market_info['%s_Open' %(prefix)])
    ax2.bar(coin_market_info['Date'].astype(datetime.datetime).values, 
            coin_market_info['%s_Volume' %(prefix)].values, color='black')
    fig.tight_layout()
    fig.figimage(logo_img, img_x, img_y, zorder=3,alpha=.5)
    plt.show()

plot_coin_info(bitcoin_market_info, 'bt', bitcoin_im, 120, 120)

plot_coin_info(eth_market_info, 'eth', eth_im, 350, 200)

# only take records starting from 2016
market_info = pd.merge(bitcoin_market_info,eth_market_info, on=['Date'])
market_info = market_info[market_info['Date']>='2016-01-01']

for coins in ['bt_', 'eth_']: 
    kwargs = {coins+'day_diff': lambda x: (x[coins+'Close']-x[coins+'Open'])/x[coins+'Open']}
    market_info = market_info.assign(**kwargs)
market_info.head()

def draw_train_test(split_date, ax, prefix, y_label, logo_img, img_x, img_y):
    ax.plot(market_info[market_info['Date'] < split_date]['Date'].astype(datetime.datetime),
            market_info[market_info['Date'] < split_date]['%s_Close' %(prefix)], 
            color='#B08FC7', label='Training')
    ax.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
            market_info[market_info['Date'] >= split_date]['%s_Close' %(prefix)], 
            color='#8FBAC8', label='Test')
    ax.set_ylabel(y_label,fontsize=12)

    fig.figimage(logo_img.resize((int(logo_img.size[0]*0.65), int(logo_img.size[1]*0.65)), Image.ANTIALIAS), 
                 img_x, img_y, zorder=3, alpha=.5)

split_date = '2017-06-01'
sns.set()
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12, 8))
draw_train_test(split_date, ax1, 'bt', 'Bitcoin Price ($)', bitcoin_im, 220, 270)
draw_train_test(split_date, ax2, 'eth', 'Ethereum Price ($)', eth_im, 360, 40)

ax1.set_xticks([datetime.date(i,j,1) for i in range(2016,2019) for j in [1,7]])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(i,j,1) for i in range(2016,2019) for j in [1,7]])
ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y') for i in range(2016,2019) for j in [1,7]])

plt.tight_layout()
ax1.legend(bbox_to_anchor=(0.03, 1), loc=2, borderaxespad=0., prop={'size': 12})
plt.show()

def overlay_predict_actual(x, y_actual, y_pred, ax, y_label):
    ax.plot(x, y_actual, label='Actual')
    ax.plot(x, y_pred, label='Predict')
    ax.set_ylabel(y_label, fontsize=12)
    
def draw_pred_performance(x, bt_actual, bt_pred, eth_actual, eth_pred, title):
    sns.set()
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12, 8), sharex=True)
    overlay_predict_actual(x, bt_actual, bt_pred, ax1, 'Bitcoin Price ($)')
    overlay_predict_actual(x, eth_actual, eth_pred, ax2, 'Etherum Price ($)')
    
    xticks = []
    for year in range(2017, 2019):
        if year == 2017:
            months = range(6, 13)
        if year == 2018:
            months = range(1, 6)
        for m in months:
            xticks.append(datetime.date(year, m, 1))
    
    ax1.set_xticks(xticks)
    ax1.set_xticklabels('')
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([xtick.strftime('%b %Y') for xtick in xticks])

    ax1.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0., prop={'size': 12})
    ax1.set_title(title)
    
    fig.tight_layout()
    plt.show()

x = market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime)
bt_actual = market_info[market_info['Date']>= split_date]['bt_Close'].values
bt_pred = market_info[market_info['Date']>= datetime.datetime.strptime(split_date, '%Y-%m-%d') - 
                      datetime.timedelta(days=1)]['bt_Close'][1:].values
eth_actual = market_info[market_info['Date']>= split_date]['eth_Close'].values
eth_pred = market_info[market_info['Date']>= datetime.datetime.strptime(split_date, '%Y-%m-%d') - 
                       datetime.timedelta(days=1)]['eth_Close'][1:].values

draw_pred_performance(x, bt_actual, bt_pred, eth_actual, eth_pred, 'Simple Lag Model (Test Set)')

sns.set()
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 8))
ax1.hist(market_info[market_info['Date']< split_date]['bt_day_diff'].values, bins=100)
ax2.hist(market_info[market_info['Date']< split_date]['eth_day_diff'].values, bins=100)
ax1.set_title('Bitcoin Daily Price Changes')
ax2.set_title('Ethereum Daily Price Changes')
plt.show()

sns.set()
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 8))
ax1.hist(market_info[market_info['Date']>= split_date]['bt_day_diff'].values, bins=100, color='orange')
ax2.hist(market_info[market_info['Date']>= split_date]['eth_day_diff'].values, bins=100, color='orange')
ax1.set_title('Bitcoin Daily Price Changes')
ax2.set_title('Ethereum Daily Price Changes')
plt.show()

def random_walks(prefix, num_pred_days):
    r_walk_mean = np.mean(market_info[market_info['Date'] < split_date]['%s_day_diff' %(prefix)].values)
    r_walk_sd = np.std(market_info[market_info['Date'] < split_date]['%s_day_diff' %(prefix)].values)
    random_steps = np.random.normal(r_walk_mean, r_walk_sd, num_pred_days)
    
    return random_steps

#apply random walks
np.random.seed(202)
num_pred_days = (max(market_info['Date']).to_pydatetime() - datetime.datetime.strptime(split_date, '%Y-%m-%d')).days + 1

bt_random_steps = random_walks('bt', num_pred_days)
eth_random_steps = random_walks('eth', num_pred_days)

x = market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime)
bt_actual = market_info[market_info['Date']>= split_date]['bt_Close'].values
bt_pred = market_info[market_info['Date']>= datetime.datetime.strptime(split_date, '%Y-%m-%d') - 
                      datetime.timedelta(days=1)]['bt_Close'][1:].values * (1 + bt_random_steps)
eth_actual = market_info[market_info['Date']>= split_date]['eth_Close'].values
eth_pred = market_info[market_info['Date']>= datetime.datetime.strptime(split_date, '%Y-%m-%d') - 
                       datetime.timedelta(days=1)]['eth_Close'][1:].values * (1 + eth_random_steps)

draw_pred_performance(x, bt_actual, bt_pred, eth_actual, eth_pred, 'Simple Point Random Walk (Test Set)')

#apply random walks
np.random.seed(100)
num_pred_days = (max(market_info['Date']).to_pydatetime() - datetime.datetime.strptime(split_date, '%Y-%m-%d')).days + 1

bt_random_steps = random_walks('bt', num_pred_days)
eth_random_steps = random_walks('eth', num_pred_days)

x = market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime)
bt_actual = market_info[market_info['Date']>= split_date]['bt_Close'].values
eth_actual = market_info[market_info['Date']>= split_date]['eth_Close'].values

bt_random_walk = []
eth_random_walk = []
for n_step, (bt_step, eth_step) in enumerate(zip(bt_random_steps, eth_random_steps)):
    if n_step==0:
        bt_random_walk.append(market_info[market_info['Date']< split_date]['bt_Close'].values[0] * (bt_step+1))
        eth_random_walk.append(market_info[market_info['Date']< split_date]['eth_Close'].values[0] * (eth_step+1))
    else:
        bt_random_walk.append(bt_random_walk[n_step-1] * (bt_step+1))
        eth_random_walk.append(eth_random_walk[n_step-1] * (eth_step+1))

draw_pred_performance(x, bt_actual, bt_random_walk[::-1], eth_actual, eth_random_walk[::-1], 'Full Interval Random Walk')

