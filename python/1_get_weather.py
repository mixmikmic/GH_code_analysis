import requests
import json
import time

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Plot Voronoi diagram for the stations to show approximate coverage of each.
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,voronoi_plot_2d

get_ipython().run_line_magic('matplotlib', 'inline')

stations = pd.read_csv('isd-history.csv',parse_dates=['BEGIN','END'])
# Weather records are queried by a concatenation of USAF and WBAN.
stations['station_id'] = stations.apply(lambda x: str(x['USAF'])+str(x['WBAN']), axis=1)
stations = stations.set_index('station_id')
stations.head()

utah_stations = stations[stations.STATE == 'UT']
utah_stations.head()

# Let's get stations that are valid for all of our time range. Not perfect, but we don't want to deal with interpolating missing data
start = pd.Timestamp(2010,1,1)
end = pd.Timestamp(2018,1,1)
valid_stations = utah_stations[(utah_stations.BEGIN < start) & (utah_stations.END > start)]
plt.figure()

lons = valid_stations.LON.values
lats = valid_stations.LAT.values
plt.plot(lons, lats,'ko')


vor = Voronoi(np.vstack((lons,lats)).T)
voronoi_plot_2d(vor,ax=plt.gca())
plt.gca().set_aspect(1)
valid_stations.head()

stations = valid_stations.index.values.tolist()
valid_stations.to_csv('utah_stations.csv')

url = 'https://www.ncdc.noaa.gov/access-data-service/api/v1/data'

params = {
    'dataset': 'global-hourly',
    'startDate': '2010-01-01T00:00:00',
    'endDate': '2018-02-01T00:00:00',
    'dataTypes':'AA1,AA1,AA2,AA3,TMP,VIS,WND,AJ1,AT1,AT2,AT3,AT4,AT5,AT6,AT7,AT8',
    'stations': stations,
    'format': 'json'
    
}

weather_data = pd.DataFrame()

for station in stations:
    params['stations'] = station
    print('Station:',station)
    res = requests.get(url,params=params)
    js = res.json()
    print(len(js))
    try:
        weather_data = weather_data.append(js)
    except:
        print ("Empty for station",station)
    time.sleep(0.5)

weather_data.head()
weather_data.to_csv('utah_weather_2010-2018_raw.csv')


def parseWindDir(x):
    # Wind direction, deg or 999
    d = x.split(',')[0]
    if d == '999':
        return np.nan
    return float(d)


def parseWindSpeed(x):
    s = x.split(',')[3]
    if s == '9999':
        return 0.0
    return float(s) / 10.0

def parseVisibility(x):
    v = x.split(',')[0]
    if v == '999999':
        return 16093.0
    return float(v)

def parseTemp(x):
    t = x.split(',')[0]
    if t == '+9999':
        return np.nan
    return float(t) / 10.0

def parseSnowDepth(x):
    try:
        t = x.split(',')[0]
    except:
        return 0.0

    if t == '9999':
        return 0.0
    return float(t) 

def parsePrecip(x):
    try:
        p = x.split(',')[1]
        return float(p) / 10.0
    except:
        return 0.0

weather_lut = {
    '01': 'fog',
    '02': 'fog',
    '03': 'thunder',
    '04': 'sleet/hail',
    '05': 'hail',
    '06': 'glaze',
    '07': 'dust',
    '08': 'smoke',
    '09': 'blowing_snow',
    '10': 'tornado',
    '11': 'winds',
    '12': 'spray',
    '13': 'mist',
    '14': 'drizzle',
    '15': 'freezing_drizzle',
    '16': 'rain',
    '17': 'freezing_rain',
    '18': 'snow',
    '19': 'unknown_precipitation',
    '21': 'ground_fog',
    '22': 'ice_fog'
}
def parseWeatherType(x):
    try:
        p = x.split(',')[1]
    except:
        return x
    return weather_lut[p]
        
    
weather_data['DATE'] = pd.to_datetime(weather_data["DATE"])
weather_data['wind_dir'] = weather_data.WND.apply(parseWindDir)
weather_data['wind_speed'] = weather_data.WND.apply(parseWindSpeed)
weather_data['visibility'] = weather_data.VIS.apply(parseVisibility)
weather_data['temperature'] = weather_data.TMP.apply(parseTemp)
weather_data['precip_01'] = weather_data.AA1.apply(parsePrecip)
weather_data['precip_02'] = weather_data.AA2.apply(parsePrecip)
weather_data['precip_03'] = weather_data.AA3.apply(parsePrecip)
weather_data['precip_depth'] = weather_data[['precip_01','precip_02','precip_03']].max(axis=1)
weather_data['snow_depth'] = weather_data.AJ1.apply(parseSnowDepth)
weather_data['AT1'] = weather_data.AT1.apply(parseWeatherType)
weather_data['AT2'] = weather_data.AT2.apply(parseWeatherType)
weather_data['AT3'] = weather_data.AT3.apply(parseWeatherType)
weather_data['AT4'] = weather_data.AT4.apply(parseWeatherType)
weather_data['AT5'] = weather_data.AT5.apply(parseWeatherType)
weather_data['AT6'] = weather_data.AT6.apply(parseWeatherType)
weather_data['AT7'] = weather_data.AT7.apply(parseWeatherType)
weather_data['AT8'] = weather_data.AT8.apply(parseWeatherType)

weather_types = weather_data[['AT1','AT2','AT3','AT4','AT5','AT6','AT7','AT8']]

weather_data['snowing'] = weather_types.apply(lambda x: x.isin(['snow'])).any(axis=1)
weather_data['raining'] = weather_types.apply(lambda x: x.isin(['rain','freezing_rain','drizzle','freezing_drizzle'])).any(axis=1)
weather_data['foggy'] = weather_types.apply(lambda x: x.isin(['fog','ground_fog','ice_fog'])).any(axis=1)
weather_data['thunderstorm'] = weather_types.apply(lambda x: x.str.contains('thunder')).any(axis=1)
weather_data['hailing'] = weather_types.apply(lambda x: x.isin(['sleet/hail','hail'])).any(axis=1)
weather_data['icy'] = weather_types.apply(lambda x: x.isin(['glaze','freezing_rain,freezing_drizzle','ice_fog','snow'])).any(axis=1)
weather_data['station_id'] = weather_data.STATION

# Drop parsed columns
weather_data = weather_data.drop(['WND','VIS','AA1','AA2','AA3','precip_01','precip_02','precip_03','AJ1'],axis=1)
weather_data = weather_data.drop(['TMP','AT1','AT2','AT3','AT4','AT5','AT6','AT7','AT8','SOURCE','REPORT_TYPE','QUALITY_CONTROL','STATION'],axis=1)

weather_data.head()

ts = weather_data.set_index('DATE').temperature.resample('1d').mean()
ts.plot()
ts = weather_data.set_index('DATE').snow_depth.resample('1d').mean()
plt.figure()
ts.plot()
ts = weather_data.set_index('DATE')[['icy','raining','foggy','hailing','snowing']].resample('1d').sum()
plt.figure()
ts.plot()

weather_data.to_csv('utah_weather_2010-2018.csv')
weather_data = pd.read_csv('utah_weather_2010-2018.csv',index_col=0)
weather_data.head()

weather_data['timestamp'] = pd.to_datetime(weather_data.DATE)

time_index = pd.DatetimeIndex(weather_data['timestamp'])

aggs = {
    'snowing': 'any',
    'raining': 'any',
    'foggy': 'any',
    'icy': 'any',
    'hailing': 'any',
    'thunderstorm':'any',
    'wind_speed': 'mean',
    'visibility': 'mean',
    'temperature': 'mean',
    'precip_depth':'mean',
    'snow_depth':'mean'
}

resamp = pd.DataFrame()
station_ids = list(set(weather_data.station_id.tolist()))
for _id in station_ids:
    idx = weather_data.station_id == _id
    ti = time_index[idx]

    wdfi = weather_data[idx].set_index(ti)
    floating = wdfi[['visibility','temperature','wind_speed','precip_depth','snow_depth']]
    binaries = wdfi[['snowing','raining','foggy','icy','hailing','thunderstorm']]
    b = binaries.resample('1h').rolling(24).apply(lambda x: x.any())
    f = floating.resample('1h').agg({
        'wind_speed': 'mean',
        'visibility': 'mean',
        'temperature': 'mean',
        'precip_depth':'mean',
        'snow_depth':'mean'
    })

    temp = pd.concat((f,b),axis=1)
    temp['station_id'] = _id
    resamp = resamp.append(temp)

for f in ['visibility','temperature','wind_speed','precip_depth','snow_depth']:
    resamp.loc[pd.isna(resamp[f]),f] = np.nanmedian(resamp[f])
for f in ['snowing','raining','foggy','icy','hailing','thunderstorm']:
    resamp.loc[pd.isna(resamp[f]),f] = 0

resamp.reset_index().set_index(['timestamp','station_id']).to_csv('utah_weather_2010-2018_grouped.csv')



