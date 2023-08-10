# Import Snap's share price from Google
# google 'pandas remote data access' --> pandas-datareader.readthedocs...

get_ipython().magic('matplotlib inline')
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

import datetime

start = datetime.datetime(2017, 3, 2) # the day Snap went public
end = datetime.date.today() # datetime.date.today

snap = web.DataReader("SNAP", 'google', start, end)

snap

snap.index.tolist()

# How did we do this before?

for index in snap.index:
     print('On day', index, 'Snap closed at', snap['Close'][index], 'and the volume was', snap['Volume'][index], '.')

# express Volume in millions
snap['Volume'] = snap['Volume']/10**6

snap

print('Today is {}.'.format(datetime.date.today()))

for index in snap.index:
     print('On {} Snap closed at ${} and the volume was {} million.'.format(index, snap['Close'][index], snap['Volume'][index]))

for index in snap.index:
     print('On {:.10} Snap closed at ${} and the volume was {:.1f} million.'.format(str(index), snap['Close'][index], snap['Volume'][index]))

fig, ax = plt.subplots() #figsize=(8,5))

snap['Close'].plot(ax=ax, grid=True, style='o', alpha=.6)
ax.set_xlim([snap.index[0]-datetime.timedelta(days=1), snap.index[-1]+datetime.timedelta(days=1)])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.vlines(snap.index, snap['Low'], snap['High'], alpha=.2, lw=.9)
ax.set_ylabel('SNAP share price', fontsize=14)
ax.set_xlabel('Date', fontsize=14)
plt.show()

start_w = datetime.datetime(2008, 6, 8)
oilwater = web.DataReader(['BP', 'AWK'], 'google', start_w, end)

oilwater.describe

type(oilwater[:,:,'AWK'])

water = oilwater[:, :, 'AWK']
oil = oilwater[:, :, 'BP']

#import seaborn as sns
#import matplotlib as mpl
#mpl.rcParams.update(mpl.rcParamsDefault)

plt.style.use('seaborn-notebook')
plt.rc('font', family='serif')

deepwater = datetime.datetime(2010, 4, 20)

fig, ax = plt.subplots(figsize=(8, 5))
water['Close'].plot(ax=ax, label='AWK', lw=.7) #grid=True,
oil['Close'].plot(ax=ax, label='BP', lw=.7) #grid=True, 
ax.yaxis.grid(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.vlines(deepwater, 0, 100, linestyles='dashed', alpha=.6)
ax.text(deepwater, 70, 'Deepwater catastrophe', horizontalalignment='center')
ax.set_ylim([0, 100])
ax.legend(bbox_to_anchor=(1.2, .9), frameon=False)
plt.show()

print(plt.style.available)

fig, ax = plt.subplots(figsize=(8, 5))
water['AWK_pct_ch'] = water['Close'].diff().cumsum()/water['Close'].iloc[0]
oil['BP_pct_ch'] = oil['Close'].diff().cumsum()/oil['Close'].iloc[0]
#water['Close'].pct_change().cumsum().plot(ax=ax, label='AWK')
water['AWK_pct_ch'].plot(ax=ax, label='AWK', lw=.7)
#oil['Close'].pct_change().cumsum().plot(ax=ax, label='BP')
oil['BP_pct_ch'].plot(ax=ax, label='BP', lw=.7)
ax.yaxis.grid(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.vlines(deepwater, -1, 3, linestyles='dashed', alpha=.6)
ax.text(deepwater, 1.2, 'Deepwater catastrophe', horizontalalignment='center')
ax.set_ylim([-1, 3])
ax.legend(bbox_to_anchor=(1.2, .9), frameon=False)
ax.set_title('Percentage change relative to {:.10}\n'.format(str(start_w)), fontsize=14, loc='left')
plt.show()

