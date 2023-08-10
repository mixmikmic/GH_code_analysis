get_ipython().magic('matplotlib inline')
import requests
import pandas as pd
import os
import sys
from cgsn_parsers.parsers.parse_metbk import Parser as Parser_metbk

# add cgsn-parsers tools to the path (no package yet)
path = os.path.abspath('../')
sys.path.append(path)

#Coastal Endurance - WA Shelf Surface Piercing Profiler Mooring
baseurl = "https://rawdata.oceanobservatories.org/files/CE07SHSM/D00003/cg_data/dcl11/metbk/"
fname = "20160522.metbk.log"

# Coastal Pioneer - Central Surface Mooring
baseurl = "https://rawdata.oceanobservatories.org/files/CP01CNSM/D00006/cg_data/dcl11/metbk1/"
fname = "20170112.metbk1.log"

# initialize the Parser object for METBK
metbk = Parser_metbk(baseurl + fname)

r = requests.get(metbk.infile, verify=True) # use verify=False for expired certificate
metbk.raw = r.content.splitlines()

len(metbk.raw), metbk.raw[-5:]

metbk.parse_data()

metbk.data.keys()

df = pd.DataFrame(metbk.data)
df['dt_utc'] = pd.to_datetime(df.dcl_date_time_string, utc=True)
df.set_index('dt_utc', drop=False, inplace=True)

# Later, can drop time, dcl_date_time_string, and maybe dt_utc (not the index)
df.shape, df.columns

df.dtypes

df.tail(10)

df[['air_temperature', 'sea_surface_temperature']].plot(figsize=(11,5));

df['sea_surface_conductivity'].plot(figsize=(11,5));





