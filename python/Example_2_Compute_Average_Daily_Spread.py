# Standard imports
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import datetime
get_ipython().magic('matplotlib inline')
# EveKit imports
from evekit.reference import Client
from evekit.util import convert_raw_time

# Settings for the day we want to compute
# We retreive type, region and station ID from the SDE
sde_client = Client.SDE.get()
type_query = "{values: ['Tritanium']}"
region_query = "{values: ['The Forge']}"
station_query = "{values: ['Jita IV - Moon 4 - Caldari Navy Assembly Plant']}"
type_id = sde_client.Inventory.getTypes(typeName=type_query).result()[0][0]['typeID']
region_id = sde_client.Map.getRegions(regionName=region_query).result()[0][0]['regionID']
station_id = sde_client.Station.getStations(stationName=station_query).result()[0][0]['stationID']
compute_date = convert_raw_time(1483228800000) # 2017-01-01 12:00 AM UTC
print("Using type_id=%d, region_id=%d, station_id=%d at %s" % (type_id, region_id, station_id, str(compute_date)))

# Let's develop code for computing spread from an order book retrieved from the Orbital Enterprises market data service
# 
# To start, we'll need the client for the service.  We'll also fetch the first book available on our compute date.
mdc_client = Client.MarketData.get()
sample_book = mdc_client.MarketData.book(typeID=type_id, regionID=region_id, date=str(compute_date) + " UTC").result()[0]
sample_book

# Orders are ordered with buys first, descending in price, followed by sells in ascending price
# Let's pull these out to simplify things
buy = [x for x in sample_book['orders'] if x['buy'] and x['locationID'] == station_id ]
sell = [x for x in sample_book['orders'] if not x['buy'] and x['locationID'] == station_id ]

# By construction, spread is now just the difference in price between the first buy order and 
# the first sell order.  However, there are some corner cases where one or both of these
# lists are empty.  In those cases, there isn't a spread.  Here's a function which 
# computes the spread
def compute_spread(buy, sell):
    if len(buy) == 0 or len(sell) == 0:
        return None
    return sell[0]['price'] - buy[0]['price']

compute_spread(buy, sell)

# All that remains is to compute the spread for all snapshots for the target day.
# We'll do that by iterating over 288 intervals from our start time, retrieving 
# the order book at each step.  This will require 288 calls to the market data service
# so this will take a few minutes to run.
current_time = compute_date
five_minute_delta = datetime.timedelta(minutes=5)
spreads = []
for _ in range(288):
    print("Computing spread for %s" % str(current_time), end="...")
    next_book = mdc_client.MarketData.book(typeID=type_id, regionID=region_id, date=str(current_time) + " UTC").result()[0]
    buy = [x for x in next_book['orders'] if x['buy'] and x['locationID'] == station_id ]
    sell = [x for x in next_book['orders'] if not x['buy'] and x['locationID'] == station_id ]
    next_spread = compute_spread(buy, sell)
    if next_spread is not None:
        spreads.append(next_spread)
    current_time += five_minute_delta
    print("done")

np.average(spreads)

# Now let's look at library support which eliminates or simplifies some of the steps above.
#
# As in the first example, we start with library functions that can be used to download
# book data to local storage.  Let's say it again so we don't forget: book data is quite large,
# make sure you're on a reasonably fast connection before you download lots of data
from evekit.online.Download import download_order_book_range

# The book downloader lets you filter to specific types and regions for a date range.
# In this example, we only need Tritanium in The Forge on our target date, so this should
# download relatively quickly.  We'll store the download in our local directory in "tree"
# format (i.e. YYYY/MM/DD/files...)
download_order_book_range([compute_date], ".", types=[type_id], regions=[region_id], config={'verbose': True, 'tree': True})

# Previously, we iterated through each snapshot for our target date.  We can do this 
# a bit more tersely using the OrderBook library and loading the data in a Pandas DataFrame.
# This call creates a DataFrame where each row is an individual order indexed by 
# the time of the snapshot containing the order.  The DataFrame also contains type_id
# and region_id columns which can be used to filter as needed.
from evekit.marketdata import OrderBook

order_book = OrderBook.get_data_frame(dates=[compute_date], types=[type_id], regions=[region_id], 
                                      config=dict(local_storage=".", tree=True, skip_missing=True, verbose=True))

order_book

# DataFrame operations allow us to compute average spread by concatenating a few basic operations
# But first, let's walk through it step by step so we can explain each operation.
#
# We start by eliminating all orders that aren't at our target station
filtered = order_book[order_book.location_id == station_id]

filtered

# Since spread is computed by snapshot, we need to recover the snapshots which we can
# do by grouping by index (recall that the index is essentially the snapshot time of
# each order)
#
groups = filtered.groupby(filtered.index)

groups

# Each group in the group object is a DataFrame representing a snapshot indexed by snapshot time
#
df = groups.get_group(filtered.index[0])

df

# We can implement a simple function to compute the spread given a group object DataFrame.
# Note that we still have to handle the case where there may be no buys or sells.  When
# we encounter such cases, we return a NaN which is conveniently ignored by the Pandas
# mean function.
#
def spread_df(df):
    if df[df.buy == True].index.size == 0 or df[df.buy == False].index.size == 0:
        return np.NaN

    return df[df.buy == False].sort_values('price', ascending=True).ix[0].price -            df[df.buy == True].sort_values('price', ascending=False).ix[0].price

spread_df(df)

# Finally, we can apply our spread function to the groups and compute the mean (which excludes NaN automatically)
groups.apply(spread_df).mean()

# Here's the whole thing as a one liner (except for the spread_df function)
#
order_book[order_book.location_id == station_id].groupby(order_book[order_book.location_id == station_id].index)                                                .apply(spread_df)                                                .mean()

