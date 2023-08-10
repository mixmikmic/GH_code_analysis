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

# We'll use The Forge region to check for liquid assert types.  The set of liquid types
# can vary from day to day.  So, we'll also specify a date range and focus on finding
# types which are liquid over an acceptable portion of our date range.  For this example,
# we'll use a 90 day date range backwards from our reference date.
#
# This cell initializes these settings.
#
sde_client = Client.SDE.get()
region_query = "{values: ['The Forge']}"
region_id = sde_client.Map.getRegions(regionName=region_query).result()[0][0]['regionID']
compute_date = convert_raw_time(1483228800000) # 2017-01-01 12:00 AM UTC
date_range = pd.date_range(compute_date - datetime.timedelta(days=90), compute_date)
print("Using region_id=%d from %s to %s" % (region_id, str(date_range[0]), str(date_range[-1])))

#
# WARNING: this cell will take some time to evaluate if your connection is slow
#
# We'll find liquid types by analyzing market history which we download to local storage
# for convenience.  This isn't mandatory and if you'd prefer to load everything online on demand,
# you can remove the "local_storage" argument from the market history functions below.
# However, we strongly recommend downloading history to local storage to allow you to iterate
# more easily as you experiment with different filters.
#
from evekit.online.Download import download_market_history_range
download_market_history_range(date_range, ".", dict(skip_missing=True, tree=True, verbose=True))

# We'll need all market types for this example.  We'll retrieve the set of market types from the SDE.
# The EveKit libraries include a convenience function for iterating over a large set of results
# (like the set of all market types).
#
market_types = Client.SDE.load_complete(sde_client.Inventory.getTypes, marketGroupID="{start: 0, end: 1000000000}")
market_types = [x['typeID'] for x in market_types]
len(market_types)

# Now we'll load market history into a DataFrame.  Note that we're loading for all types in a single
# region for 90 days so this may take a few minutes.  You can reduce the set of types if memory or 
# speed is an issue for your installation.
#
from evekit.marketdata import MarketHistory
market_history = MarketHistory.get_data_frame(dates=date_range, types=market_types, regions=[region_id], 
                                              config=dict(local_storage=".", tree=True, skip_missing=True, verbose=True))

# We'll want to experiment with various liquidity filters.  So we'll create a generic 
# liquidity evaluator which accepts a market history object and a generic filter.
# The evaluator will apply the filter to each asset type in each region and record which
# types the filter marks as liquid.  The result of the evaluator is a map from region to
# the set of liquid types in that region.
#
# The signature of the liquidity filter is:
#
# boolean liquidp(region_id, type_id, history)
#
# where the history DataFrame will be pre-filtered to the selected region and type.
# The filter should return True if the given type should be considered liquid in 
# the given region, and False otherwise.
#
def liquid_types(history, liquidp, verbose=False):
    # Result is map from region to set of liquid types for that region
    # Iterate through all types contained in the history object
    liquid_map = {}
    count = 0
    # Iterate through all regions and types
    for next_region in history.region_id.unique():
        liquid_set = set()
        by_region = history[history.region_id == next_region]
        for next_type in by_region.type_id.unique():
            by_type = by_region[by_region.type_id == next_type]
            if liquidp(next_region, next_type, by_type):
                liquid_set.add(next_type)
            count += 1
            if count % 1000 == 0 and verbose:
                print("Tested %d (region, type) pairs" % count)
        liquid_map[next_region] = liquid_set
    return liquid_map

# There are many ways to define liquidity.  For our first filter, we'll treat a type as liquid if:
#
# 1. Market history exists for a minimum number of days
# 2. Each day of market history meets an order count threshold
# 3. Each day of market history meets an ISK volume threshold
#
# It is convenient to express our filter as a functor (a function which returns a function) so that
# it is easier to experiment with different thresholds.
#
def liquidity_filter(min_days, min_order_count, min_isk_volume):
    def liquidp(region_id, type_id, history):
        return len(history) >= min_days and                len(history[history.order_count < min_order_count]) == 0 and                len(history[(history.avg_price * history.volume) < min_isk_volume]) == 0
    return liquidp

# Let's start with a simple parameterization.  We'll expect a liquid type to:
#
# 1. Be traded at least 70% of the days in the date range;
# 2. Have at least 100 orders a day; and,
# 3. Have at least 100m ISK of activity
#
day_min = int(len(date_range) * 0.7)
order_min = 100
volume_min = 100000000

# We can hand test our filter on a sample type.  For example, Tritanium (type 34) is almost certainly liquid over this range:
(liquidity_filter(day_min, order_min, volume_min))(region_id, 34, market_history[market_history.region_id == region_id]                                                                                [market_history.type_id == 34])

# Let's collect the list of all liquid types for this data.  This may take a few minutes
# to run due to the size of the data.
l_set = liquid_types(market_history, liquidity_filter(day_min, order_min, volume_min), verbose=True)

import pprint
pprint.pprint(l_set, compact=True)

# You can use the SDE if you want to see type names for these types as follows.
#
type_name_query = "{values:[" + ",".join(map(str, l_set[region_id])) + "]}"
type_name_results = Client.SDE.load_complete(sde_client.Inventory.getTypes, typeID=type_name_query)
pprint.pprint([x['typeName'] for x in type_name_results], compact=True)

# Our first filter selected across the entire date range.  However, many assets are more active on weekends.
# For example, here is Tritanium ISK volume with week days and week ends (friday through saturday) 
# rendered in different colors.
#
weekends = [x for x in date_range if x.weekday() in (4,5,6)]
weekdays = [x for x in date_range if x not in weekends]
tritanium = market_history[market_history.region_id == region_id][market_history.type_id == 34].copy()
tritanium['weekday_volume'] = tritanium.volume.ix[weekdays]
tritanium['weekend_volume'] = tritanium.volume.ix[weekends]
tritanium[['weekday_volume','weekend_volume']].plot(kind='bar', figsize=[18,5])

# Note that weekend volume is usually slightly higher than weekday volume.  We can write a liquidity
# filter which only considers weekends as shown below. 
#
def liquidity_filter_2(min_days, min_order_count, min_isk_volume):
    def liquidp(region_id, type_id, history):
        by_weekend = history.ix[[x for x in history.index.unique() if x.weekday() in (4, 5, 6)]]
        return len(by_weekend) >= min_days and                len(by_weekend[by_weekend.order_count < min_order_count]) == 0 and                len(by_weekend[(by_weekend.avg_price * by_weekend.volume) < min_isk_volume]) == 0
    return liquidp

# Since there are many fewer days under consideration, we'll need to adjust our day minimum,
# but otherwise we can keep the same parameters from before.
#
day_min = int(len(weekends) * 0.7)
order_min = 100
volume_min = 100000000

# And now we can apply the filter...
l_set_2 = liquid_types(market_history, liquidity_filter_2(day_min, order_min, volume_min), verbose=True)
pprint.pprint(l_set_2, compact=True)

# As we might expect, there are more liquid types when we only consider weekends:
print("Full Date Range: %d  Weekends Only: %d" % (len(l_set[region_id]), len(l_set_2[region_id])))
new_liquid_types = l_set_2[region_id].difference(l_set[region_id])
type_name_query = "{values:[" + ",".join(map(str, new_liquid_types)) + "]}"
type_name_results = Client.SDE.load_complete(sde_client.Inventory.getTypes, typeID=type_name_query)
print("New liquid types:")
pprint.pprint([x['typeName'] for x in type_name_results], compact=True)

