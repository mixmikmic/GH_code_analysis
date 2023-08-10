start = '2016-07-10T00:00:00Z'
stop  = '2017-02-10T00:00:00Z'
lat_min =  39.
lat_max =  41.5
lon_min = -72.
lon_max = -69.
standard_name = 'sea_water_temperature'

import pandas as pd

base = (
    'https://data.ioos.us/gliders/erddap/search/advanced.csv'
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
    standard_name,
    lat_max,
    lon_min,
    lon_max,
    lat_min,
    start,
    stop
)

dft = pd.read_csv(url, usecols=['Title', 'Summary', 'Institution'])  

print('Glider Datasets Found = {}'.format(len(dft)))
print(url)
dft

def download_df(glider_id):
    from pandas import DataFrame, read_csv
#    from urllib.error import HTTPError
    uri = ('https://data.ioos.us/gliders/erddap/tabledap/{}.csv'
           '?trajectory,wmo_id,time,latitude,longitude,depth,pressure,temperature'
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

# We need to add the "if 'All' not in glider_id" logic here.
    
dfs = list(map(download_df, dft['Title'].values))

df = pd.concat(dfs)

print('Total Data Values Found: {}'.format(len(df)))

df.head()

df.tail()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig, ax = plt.subplots(
    figsize=(9, 9),
    subplot_kw=dict(projection=ccrs.PlateCarree())
)
ax.coastlines(resolution='10m')
dx = dy = 0.5
ax.set_extent([lon_min-dx, lon_max+dx, lat_min-dy, lat_max+dy])

g = df.groupby('trajectory')
for glider in g.groups:
    traj = df[df['trajectory'] == glider]
    ax.plot(traj['longitude'], traj['latitude'], label=glider)

ax.legend();

fig = plt.figure(figsize=(10,10))
plt.scatter(df['temperature'],-df['pressure'],
            facecolors='none', edgecolors='black',s=10);
plt.ylabel('depth'); plt.xlabel('temperature');
plt.grid()



