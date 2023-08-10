# This automatically time every cell's execution
get_ipython().system('pip install ipython-autotime')
get_ipython().run_line_magic('load_ext', 'autotime')

#Imports 
import numpy as np 
import pandas as pd
import math
from timeit import default_timer as timer

#GPU
import pygdf 

#
from collections import OrderedDict

#Load the Zip Code data
raw_data = pd.read_csv("./data/Zip_time_series.csv", parse_dates=True)

# We want the Date as a 'datetime' and not as a string
raw_data.Date=pd.to_datetime(raw_data.Date)

# Add two new fields by breaking the date into Year and Month
# only last day on month is used, so it can be ignored
raw_data['Year']  = raw_data.Date.dt.year
raw_data['Month'] = raw_data.Date.dt.month

# How many Records and Fields?
raw_data.shape

# Get only the fields we care about for this task: Year, Month, Median list prices
sale_data = raw_data[['Year', 'Month', 'RegionName', 'MedianListingPrice_1Bedroom', 
                         'MedianListingPrice_2Bedroom', 'MedianListingPrice_3Bedroom', 
                         'MedianListingPrice_4Bedroom', 'MedianListingPrice_5BedroomOrMore', 
                         'MedianListingPrice_AllHomes', 'MedianListingPrice_CondoCoop',
                         'MedianListingPrice_SingleFamilyResidence' ]]

sale_data.shape

# convert all NaN to 0
sale_data[np.isnan(sale_data)] = 0

gdf = pygdf.DataFrame.from_pandas(sale_data)

# Aggregating methods to apply to each column
aggs = OrderedDict()
aggs['MedianListingPrice_AllHomes'] = 'max'


topZip = gdf.groupby(by='RegionName').agg(aggs).sort_values(by='MedianListingPrice_AllHomes', ascending=False)

topZip.head()



