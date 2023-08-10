import pandas as pd

url = 'http://ooi-data.marine.rutgers.edu/erddap/search/advanced.csv?page=1&itemsPerPage=1000&searchFor=glider'
dft = pd.read_csv(url, usecols=['Title', 'Summary', 'Institution', 'Dataset ID'])  
dft.head()

start = '2000-01-01T00:00:00Z'
stop  = '2017-02-22T00:00:00Z'
lat_min =  39.
lat_max =  41.5
lon_min = -72.
lon_max = -69.
standard_name = 'sea_water_temperature'
endpoint = 'http://ooi-data.marine.rutgers.edu/erddap/search/advanced.csv'

import pandas as pd


base = (
    '{}'
    '?page=1'
    '&itemsPerPage=1000'
    '&searchFor='
    '&protocol=(ANY)'
    '&cdm_data_type=(ANY)'
    '&institution=(ANY)'
    '&ioos_category=(ANY)'
    '&keywords=(ANY)'
    '&long_name=(ANY)'
    '&standard_name={}'
    '&variableName=(ANY)'
    '&maxLat={}'
    '&minLon={}'
    '&maxLon={}'
    '&minLat={}'
    '&minTime={}'
    '&maxTime={}').format

url = base(
    endpoint,
    standard_name,
    lat_max,
    lon_min,
    lon_max,
    lat_min,
    start,
    stop
)

print(url)

dft = pd.read_csv(url, usecols=['Title', 'Summary', 'Institution','Dataset ID'])  

print('Datasets Found = {}'.format(len(dft)))
print(url)
dft

def download_df(glider_id):
    from pandas import DataFrame, read_csv
#    from urllib.error import HTTPError
    uri = ('http://ooi-data.marine.rutgers.edu/erddap/tabledap/{}.csv'
           '?trajectory,'
           'time,latitude,longitude,'
           'ctdpf_ckl_wfp_instrument_ctdpf_ckl_seawater_temperature'
           '&orderByMax("time")'
           '&time>={}'
           '&time<={}'
           '&latitude>={}'
           '&latitude<={}'
           '&longitude>={}'
           '&longitude<={}').format
    url = uri(glider_id,start,stop,lat_min,lat_max,lon_min,lon_max)
    print(url)
    # Not sure if returning an empty df is the best idea.
    try:
        df = read_csv(url, index_col='time', parse_dates=True, skiprows=[1])
    except:
        df = pd.DataFrame()
    return df

df = pd.concat(list(map(download_df, dft['Dataset ID'].values)))

print('Total Data Values Found: {}'.format(len(df)))

df

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature

bathym_1000 = NaturalEarthFeature(name='bathymetry_J_1000',
                                  scale='10m', category='physical')

fig, ax = plt.subplots(
    figsize=(9, 9),
    subplot_kw=dict(projection=ccrs.PlateCarree())
)
ax.coastlines(resolution='10m')
ax.add_feature(bathym_1000, facecolor=[0.9, 0.9, 0.9], edgecolor='none')

dx = dy = 0.5
ax.set_extent([lon_min-dx, lon_max+dx, lat_min-dy, lat_max+dy])

g = df.groupby('trajectory')
for glider in g.groups:
    traj = df[df['trajectory'] == glider]
    ax.plot(traj['longitude'], traj['latitude'], 'o', label=glider)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')

ax.legend();



