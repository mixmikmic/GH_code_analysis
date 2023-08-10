get_ipython().magic('matplotlib inline')
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
DATA_DIR = Path('data', 'financial', 'raw')

twtr_df = pd.read_csv(DATA_DIR.joinpath('TWTR.csv'), parse_dates=['Date'])
fb_df = pd.read_csv(DATA_DIR.joinpath('FB.csv'), parse_dates=['Date'])

twtr_df.head()

fb_df.tail()

# Basic chart
fig, ax = plt.subplots()
ax.plot(fb_df['Date'], fb_df['Adj Close'], color='#5566DD')
ax.plot(twtr_df['Date'], twtr_df['Adj Close'], color='#88CCEE');

from datetime import datetime
fig, ax = plt.subplots()
ax.plot(fb_df['Date'], fb_df['Adj Close'], color='#5566DD')
ax.plot(twtr_df['Date'], twtr_df['Adj Close'], color='#88CCEE');
ax.set_xticks([datetime(2012, 1, 1), datetime(2013, 1, 1), datetime(2014, 1, 1), 
               datetime(2015, 1, 1), datetime(2016, 1, 1)]);

# use yearlocator
from matplotlib.dates import YearLocator
fig, ax = plt.subplots()
ax.plot(fb_df['Date'], fb_df['Adj Close'], color='#5566DD')
ax.plot(twtr_df['Date'], twtr_df['Adj Close'], color='#88CCEE');
ax.xaxis.set_major_locator(YearLocator());

fig, ax = plt.subplots()
ax.plot(fb_df['Date'], fb_df['Adj Close'], color='#5566DD')
ax.plot(twtr_df['Date'], twtr_df['Adj Close'], color='#88CCEE');
ax.xaxis.set_major_locator(YearLocator());
ax.set_yticks(range(0, 120, 30));

# manually set a y-limit so that there is space between 120 and the top of the chart
fig, ax = plt.subplots()
ax.plot(fb_df['Date'], fb_df['Adj Close'], color='#5566DD')
ax.plot(twtr_df['Date'], twtr_df['Adj Close'], color='#88CCEE');
ax.xaxis.set_major_locator(YearLocator());
ax.set_yticks(range(0, 130, 30))
ax.set_ylim(ymax=130);

# TODO: Add quarterly earnings reports



