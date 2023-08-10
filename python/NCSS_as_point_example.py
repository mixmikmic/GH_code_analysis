get_ipython().magic('matplotlib inline')

import pandas as pd

url = 'http://data.ncof.co.uk/thredds/ncss/METOFFICE-NWS-AF-WAV-HOURLY?var=VHM0&var=VHM0_SW1&var=VHM0_WW&latitude=61.15&longitude=-9.5&time_start=2017-06-02T00%3A00%3A00Z&time_end=2017-06-17T23%3A00%3A00Z&accept=csv'

latitude = 61.15
longitude = -9.5
time_start = '2017-06-02T00:00:00Z'
time_end =   '2017-06-17T23:00:00Z'

url = 'http://data.ncof.co.uk/thredds/ncss/METOFFICE-NWS-AF-WAV-HOURLY?var=VHM0&var=VHM0_SW1&var=VHM0_WW&latitude={}&longitude={}&time_start={}&time_end={}&accept=csv'.format(latitude,longitude,time_start,time_end)

df = pd.read_csv(url, parse_dates=True, index_col=0, na_values=[-32767.0], 
        skiprows=[0], names=['Lon','Lat','SigHeight(m)','Swell(m)','Wind Waves(m)'])

# drop lon,lat columns
df = df.drop(df.columns[[0,1]], axis=1)

df.head()

# Ugh, NCSS apparently ignores the scale_factor. Shite!  I'll report this bug, but in the meantime...
df = df/100.

df.head()

df.plot(figsize=(12,4),grid='on');

