# Standard imports
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Include evekit libraries we need for this example
from evekit.reference import Client
from evekit.online import Download
from evekit.marketdata import MarketHistory

# Step 1 - lookup type and region info from the SDE
#
# We're going to show off a bit and include Domain as well as The Forge to show some interesting features.
#
# NOTE: the library version of the clients sets "also_return_response" to True for the Bravado client
# This means the result is a 2-tuple (result, response), hence the extra de-reference as compared to
# the original example.  If we wanted to do proper error handling, we should do something like:
#
# result, response = sde_client.Inventory.getTypes(typeName="{values: ['Tritanium']}").result()
# if response.status_code == 200:
#   type_id = result[0]['typeID']
# else:
#   ...error...
# 
sde_client = Client.SDE.get()
type_id = sde_client.Inventory.getTypes(typeName="{values: ['Tritanium']}").result()[0][0]['typeID']
region_id = sde_client.Map.getRegions(regionName="{values: ['The Forge']}").result()[0][0]['regionID']
domain_region_id = sde_client.Map.getRegions(regionName="{values: ['Domain']}").result()[0][0]['regionID']
print("Using type_id=%d, region_id=%d, domain_region_id=%d" % (type_id, region_id, domain_region_id))

# Step 2 - Construct a date range consisting of every day from a year ago until today.
#
import datetime
date_range = pd.date_range(datetime.date.today() - datetime.timedelta(days=365), datetime.date.today())

# Step 3 - Fetch daily market averages for every date in our date range and store the data in a Pandas DataFrame.
#
# This time, we'll let evekit do all the work for us, but to make it interesting we'll download part of the
# history to a local directory and tell evekit to use data from the local directory if it's available.
#
# First, download the first week of data in our date range
Download.download_market_history_range(date_range[0:7], ".", dict(skip_missing=True, tree=True, verbose=True))

# Step 3 (cont'd)
#
# Now we'll use the MarketHistory object to create a Pandas DataFrame with our desired range.
# We'll tell the constructor to look in our local storage first before retrieving data remotely.
# We'll also include an additional region to show how to filter from a larger result.
#
market_history = MarketHistory.get_data_frame(dates=date_range, types=[type_id], regions=[region_id, domain_region_id], 
                                              config=dict(local_storage=".", tree=True, skip_missing=True, verbose=True))

# The result is a DataFrame just as we constructed in the first example, indexed by date.
# Note that the result includes two regions this time.
market_history

# Step 4 - Graph the average price from the DataFrame.
#
# We can graph the same as before but this time we need to filter by region ID since we fetched two regions
# We can now produce two graphs.  Here's the first:
market_history[market_history.region_id == 10000002].avg_price.plot(title="Tritanium price in The Forge", figsize=[10,5])

# and here's the second
market_history[market_history.region_id == 10000043].avg_price.plot(title="Tritanium price in Domain", figsize=[10,5])

# and we can overlay them as well
market_history[market_history.region_id == 10000002].avg_price.plot(title="Tritanium price in The Forge vs. Domain", 
                                                                    figsize=[10,5],
                                                                    label="The Forge",
                                                                    legend=True)
market_history[market_history.region_id == 10000043].avg_price.plot(label="Domain", legend=True)

