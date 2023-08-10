get_ipython().magic('matplotlib inline')
import requests
import pandas as pd

import sys
path = '/home/usgs/github/cgsn-parsers/'
sys.path.append(path)
from cgsn_parsers.parsers.parse_mopak import Parser 

#Coastal Endurance - WA Shelf Surface Piercing Profiler Mooring
baseurl = "https://rawdata.oceanobservatories.org/files/CE07SHSM/D00003/cg_data/dcl11/mopak/"
fname = "20160505_220005.mopak.log"

# Coastal Pioneer - Central Surface Mooring
baseurl = "https://rawdata.oceanobservatories.org/files/CP01CNSM/D00006/cg_data/dcl11/mopak/"
fname = "20170116_150009.mopak.log"

# initialize the Parser object for METBK
mopak = Parser(baseurl + fname)

r = requests.get(mopak.infile, verify=True) # use verify=False for expired certificate

mopak.raw = r.content

len(mopak.raw), mopak.raw[-5:]

mopak.parse_data()

mopak.data.keys()

df = pd.DataFrame(mopak.data)
df['dt_utc'] = pd.to_datetime(df.time, unit='s')
df.set_index('dt_utc', drop=True, inplace=True)

# Later, can drop time, dcl_date_time_string, and maybe dt_utc (not the index)
df.shape, df.columns

df.dtypes

df.tail(10)

df[['acceleration_x', 'acceleration_y']].plot(figsize=(11,5));

df['angular_rate_z'].plot(figsize=(11,5));

