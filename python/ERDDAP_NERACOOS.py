from IPython.display import IFrame
IFrame('http://www.neracoos.org/erddap/tabledap/A01_met_all.graph', width='100%', height=450)

##Initialize

import urllib2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

start='2014-07-01T00:00:00Z'
stop='2014-07-11T00:00:00Z'

# Construct URL for CSV data:
url='http://www.neracoos.org/erddap/tabledap/A01_met_all.csv?station,time,air_temperature,wind_speed,wind_direction,longitude,latitude&time>=%s&time<=%s' % (start,stop)

# Load the CSV data directly into Pandas
df = pd.read_csv(url,index_col='time',parse_dates=True,skiprows=[1])  # skip the units row 

# List last ten records
df.tail(10)

df[['wind_speed','air_temperature']].plot(figsize=(12,4));



