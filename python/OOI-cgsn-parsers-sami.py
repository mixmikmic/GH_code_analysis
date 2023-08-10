get_ipython().magic('matplotlib inline')
import requests
import pandas as pd
import sys
from cgsn_parsers.parsers.parse_phsen import Parser 

# add cgsn-parsers tools to the path (no package yet)
path = os.path.abspath('../')
sys.path.append(path)

#Coastal Endurance - WA Shelf Surface Piercing Profiler Mooring
baseurl = "https://rawdata.oceanobservatories.org/files/CE07SHSM/D00003/cg_data/dcl26/phsen1/"
fname = "20160926.phsen1.log"

# Coastal Pioneer - Central Surface Mooring
baseurl = "https://rawdata.oceanobservatories.org/files/CP01CNSM/D00006/cg_data/dcl26/phsen1/"
fname = "20170116.phsen1.log"

# initialize the Parser object for METBK
phsen = Parser(baseurl + fname)

r = requests.get(phsen.infile, verify=True) # use verify=False for expired certificate
phsen.raw = r.content

len(phsen.raw), phsen.raw[-5:]

phsen.parse_data()

phsen.data.keys()

df = pd.DataFrame(phsen.data)
df['dt_utc'] = pd.to_datetime(df.time, unit='s')
df.set_index('dt_utc', drop=True, inplace=True)
del df['dcl_date_time_string']

df.head()

# Later, can drop time, and maybe dt_utc (not the index)
df.shape, df.columns

df.dtypes

#extract a specific element from each list of light_measurements
df['light_0'] = [x[0] for x in df['light_measurements']]
df['light_end'] = [x[-1] for x in df['light_measurements']]

df['light_0'].plot(figsize=(11,5), grid='on');

