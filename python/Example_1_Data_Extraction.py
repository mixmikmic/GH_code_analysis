# Just about every example will start with these imports.
# These are the basic tools we'll use in most examples
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# You should have installed Bravado before you started this example.
# If not, exit out of Jupyter and install bravado using:
#
# pip install --upgrade bravado
#
from bravado.client import SwaggerClient

# Step 1 - lookup type and region info from the SDE
#
# We need a SwaggerClient for the SDE which we can create as follows:
sde_client = SwaggerClient.from_url("https://evekit-sde.orbital.enterprises/latest/swagger.json", 
                                    config={'use_models': False, 'validate_responses' : False})

# Now let's make calls to look up Tritanium and The Forge
# You can view the latest SDE endpoint docs here: https://evekit.orbital.enterprises//#/sde/ui
#
# Look up type ID for Tritanium
sde_client.Inventory.getTypes(typeName="{values: ['Tritanium']}").result()

# Helpful tip - if you ever forget the documentation for a method, you can view docstring for a method as follows:
get_ipython().magic('pinfo sde_client.Inventory.getTypes')

# If we only wanted the typeID in the previous example, we could have done:
# ...result()[0]['typeID']
#
# Now we'll lookup The Forge
sde_client.Map.getRegions(regionName="{values: ['The Forge']}").result()

# Let's tidy this up and save the typeID and regionID we want
type_id = sde_client.Inventory.getTypes(typeName="{values: ['Tritanium']}").result()[0]['typeID']
region_id = sde_client.Map.getRegions(regionName="{values: ['The Forge']}").result()[0]['regionID']
print("Using typeID=%d and regionID=%d" % (type_id, region_id))

# Step 2 - Construct a date range consisting of every day from a year ago until today.
#
# This is trival with pandas
import datetime
date_range = pd.date_range(datetime.date.today() - datetime.timedelta(days=365), datetime.date.today())
date_range

# Step 3 - Fetch daily market averages for every date in our date range and store the data in a Pandas DataFrame.
#
# Before we dive into this, we need a SwaggerClient for the market data collection service
mdc_client = SwaggerClient.from_url("https://evekit-market.orbital.enterprises//swagger", 
                                    config={'use_models': False, 'validate_responses' : False})

# Let's test our client on the first date in our range
mdc_client.MarketData.history(typeID=type_id, regionID=region_id, date=str(date_range[0])).result()

# Let's see what happens if we request a missing date
mdc_client.MarketData.history(typeID=type_id, regionID=region_id, date="2030-01-01").result()

# That was nasty but instructive.  It means we'll need to catch HTTPNotFound when we're looking up data from the service.
# Actually, we'll be a little sloppy and just catch HTTPError instead which is the parent exception.
#
# Now we can use a simple loop to gather all the data.  This is a year's worth of calls so it may take a few minutes 
# depending on your connection speed.
#
from bravado.exception import HTTPError
market_history = []
for next in date_range:
    try:
        print(".", end="")
        next_data = mdc_client.MarketData.history(typeID=type_id, regionID=region_id, date=str(next)).result()
        market_history.append(next_data)
    except HTTPError:
        # Skip this date
        pass
print()
market_history

# Now we have our market history, we just need to turn it into a DataFrame.  This is easy to do, but first we need to
# figure out how to convert dates returned by the service into a date type Pandas can accept.
#
# Here's a one liner which will do the trick:
raw_time = market_history[0]['date']
datetime.datetime.utcfromtimestamp(raw_time//1000).replace(microsecond=raw_time%1000*1000)

# We'll make our one-line into a function which we'll use to construct our DataFrame
def convertRawTime(raw_time):
    return datetime.datetime.utcfromtimestamp(raw_time//1000).replace(microsecond=raw_time%1000*1000)

mh_frame = DataFrame(market_history, index=[convertRawTime(v['date']) for v in market_history])
mh_frame

# Step 4 - Graph the average price from the DataFrame.
#
# Finally, we can graph average price for the range we've selected using matplotlib
# If you want to learn more about configuring matplotlib, check out: http://matplotlib.org/
mh_frame.avgPrice.plot(title="Tritanium price in The Forge", figsize=[10,5])

