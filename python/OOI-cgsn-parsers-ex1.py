get_ipython().magic('matplotlib inline')
import requests
import pandas as pd

import sys
sys.path.insert(0, '/usr/mayorgadat/workmain/APL/DataSources/OOI/ooici/cgsn-parsers/parsers')

from parse_metbk import Parser as Parser_metbk

baseurl = "https://rawdata.oceanobservatories.org/files/CE07SHSM/D00003/cg_data/dcl11/metbk/"

fname = "20160522.metbk.log"

# initialize the Parser object for METBK
metbk = Parser_metbk(baseurl + fname)

# verify=False added b/c certificate is expired
r = requests.get(metbk.infile, verify=False)
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



