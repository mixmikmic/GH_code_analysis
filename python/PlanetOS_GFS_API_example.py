get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import dateutil.parser
import datetime
from urllib.request import urlopen, Request
import simplejson as json

longitude = 24.+36./60
latitude = 59+24./60
apikey = ''

#API_url = "http://data.planetos.com/api/data/dataset_physical_values/noaa_gfs_global_sflux_0.12d?lon={0}&lat={1}&count=10&verbose=true".format(longitude,latitude)
API_url = "http://api.planetos.com/v1/datasets/noaa_gfs_global_sflux_0.12d/point?lon={0}&lat={1}&count=10&verbose=true&apikey={2}".format(longitude,latitude,apikey)
request = Request(API_url)
response = urlopen(request)
API_data = json.loads(response.read())

print("{0:<50} {1}".format("Variable","Context"))
print()
for k,v in set([(j,i['context']) for i in API_data['entries'] for j in i['data'].keys() if 'wind' in j.lower()]):
    print("{0:<50} {1}".format(k,v))

time_axes = []
time_axes_precipitation = []
time_axes_wind = []
surface_temperature = []
air2m_temperature = []
precipitation_rate = []
wind_speed = []
for i in API_data['entries']:
    #print(i['axes']['time'])
    if i['context'] == 'reftime_time_lat_lon':
        surface_temperature.append(i['data']['Temperature_surface'])
        time_axes.append(dateutil.parser.parse(i['axes']['time']))
    if i['context'] == 'reftime_time1_lat_lon':
        if 'Precipitation_rate_surface_3_Hour_Average' in i['data']:
            precipitation_rate.append(i['data']['Precipitation_rate_surface_3_Hour_Average']*3*3600)
            time_axes_precipitation.append(dateutil.parser.parse(i['axes']['time']))
    if i['context'] == 'reftime_time_height_above_ground_lat_lon':
        air2m_temperature.append(i['data']['Temperature_height_above_ground'])
    if i['context'] == 'reftime_time_height_above_ground1_lat_lon':
        wind_speed.append(np.sqrt(i['data']['u-component_of_wind_height_above_ground']**2+i['data']['v-component_of_wind_height_above_ground']**2))
        time_axes_wind.append(dateutil.parser.parse(i['axes']['time']))

time_axes_precipitation = np.array(time_axes_precipitation)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(time_axes,surface_temperature,color='k',label='Surface temperature')
ax.plot(time_axes,air2m_temperature,color='r',label='2m temperature')
ax_r = ax.twinx()
ax_r.bar(time_axes_precipitation-datetime.timedelta(seconds=1800),precipitation_rate,width=0.1,alpha=0.4)
fig.autofmt_xdate()
plt.show()

API_url = "http://api.planetos.com/v1/datasets/noaa_gfs_global_sflux_0.12d/point?lon={0}&lat={1}&count=1000&verbose=true&apikey={2}&contexts=reftime_time_lat_lon,reftime_time1_lat_lon,reftime_time_height_above_ground_lat_lon".format(longitude,latitude,apikey)
request2 = Request(API_url)
response2 = urlopen(request2)
API_data2 = json.loads(response2.read())

reftimes = set()
for i in API_data2['entries']:
    reftimes.update([i['axes']['reftime']])
reftimes=list(reftimes)

reftimes

if len(reftimes)>1:
    reftime = reftimes[0] if dateutil.parser.parse(reftimes[0])<dateutil.parser.parse(reftimes[1]) else reftimes[1]
else:
    reftime = reftimes[0]

time_2mt = []
time_surft = []
time_precipitation = []
time_surfrad = []
surface_temperature = []
air2m_temperature = []
precipitation_rate = []
surfrad = []
for i in API_data2['entries']:
    #print(i['context'])
    if i['context'] == 'reftime_time_lat_lon' and i['axes']['reftime']==reftime:
        surface_temperature.append(i['data']['Temperature_surface']-273.15)
        time_surft.append(dateutil.parser.parse(i['axes']['time']))
    if i['context'] == 'reftime_time1_lat_lon' and i['axes']['reftime']==reftime:
        if 'Precipitation_rate_surface_3_Hour_Average' in i['data']:
            precipitation_rate.append(i['data']['Precipitation_rate_surface_3_Hour_Average']*3*3600)
            time_precipitation.append(dateutil.parser.parse(i['axes']['time']))
        if 'Downward_Short-Wave_Radiation_Flux_surface_3_Hour_Average' in i['data']:
            surfrad.append(i['data']['Downward_Short-Wave_Radiation_Flux_surface_3_Hour_Average'])
            time_surfrad.append(dateutil.parser.parse(i['axes']['time']))
    if i['context'] == 'reftime_time_height_above_ground_lat_lon' and i['axes']['reftime']==reftime:
        if 'Temperature_height_above_ground' in i['data']:
            air2m_temperature.append(i['data']['Temperature_height_above_ground']-273.15)
            time_2mt.append(dateutil.parser.parse(i['axes']['time']))

time_precipitation = np.array(time_precipitation)
surfrad=np.array(surfrad)
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
plt.plot(time_surft,surface_temperature,color='k',label='Surface temperature')
plt.plot(time_2mt,air2m_temperature,color='r',label='2m temperature')
lg = plt.legend(framealpha=0.2)
ax.set_ylabel('Temperature, Celsius')

ax_r = ax.twinx()
ax_r.bar(time_precipitation-datetime.timedelta(seconds=1800),precipitation_rate,width=0.1,alpha=0.4,label='precipitation')
ax_r.fill_between(time_surfrad,surfrad/np.amax(surfrad)*np.amax(precipitation_rate),color='gray',alpha=0.1,label='surface radiation')
ax_r.set_ylabel('Precipitation, mm 3hr')
lg.get_frame().set_alpha(0.5)

fig.autofmt_xdate()



