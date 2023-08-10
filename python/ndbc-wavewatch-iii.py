get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import dateutil.parser
import datetime
from urllib.request import urlopen, Request
import simplejson as json
from datetime import date, timedelta, datetime
import matplotlib.dates as mdates
from mpl_toolkits.basemap import Basemap

dataset_id = 'noaa_ndbc_stdmet_stations'
## stations with wave height available: '46006', '46013', '46029'
## stations without wave height: icac1', '41047', 'bepb6', '32st0', '51004'
## stations too close to coastline (no point to compare to ww3)'sacv4', 'gelo1', 'hcef1'
station = '46029'
apikey = 'INSERT-YOUR-API-KEY-HERE'

API_url = 'http://api.planetos.com/v1/datasets/%s/stations?apikey=%s' % (dataset_id, apikey)
request = Request(API_url)
response = urlopen(request)
API_data_locations = json.loads(response.read())
# print(API_data_locations)

m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,            llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
fig=plt.figure(figsize=(15,10))
m.drawcoastlines()
##m.fillcontinents()
for i in API_data_locations['station']:
    x,y=m(API_data_locations['station'][i]['SpatialExtent']['coordinates'][0],
          API_data_locations['station'][i]['SpatialExtent']['coordinates'][1])
    plt.scatter(x,y,color='r')
x,y=m(API_data_locations['station'][station]['SpatialExtent']['coordinates'][0],
          API_data_locations['station'][station]['SpatialExtent']['coordinates'][1])
plt.scatter(x,y,s=100,color='b')

## Find suitable reference time values
atthemoment = datetime.utcnow()
atthemoment = atthemoment.strftime('%Y-%m-%dT%H:%M:%S') 

before5days = datetime.utcnow() - timedelta(days=5)
before5days_long = before5days.strftime('%Y-%m-%dT%H:%M:%S')
before5days_short = before5days.strftime('%Y-%m-%d')

start = before5days_long
end = atthemoment

reftime_start = str(before5days_short) + 'T18:00:00'
reftime_end = reftime_start

API_url = "http://api.planetos.com/v1/datasets/{0}/point?station={1}&apikey={2}&start={3}&end={4}&count=1000".format(dataset_id,station,apikey,start,end)
print(API_url)

request = Request(API_url)
response = urlopen(request)
API_data_buoy = json.loads(response.read())

buoy_variables = []
for k,v in set([(j,i['context']) for i in API_data_buoy['entries'] for j in i['data'].keys()]):
    buoy_variables.append(k)

for i in API_data_buoy['entries']:
    #print(i['axes']['time'])
    if i['context'] == 'time_latitude_longitude':
        longitude = (i['axes']['longitude'])
        latitude = (i['axes']['latitude'])

print ('Latitude: '+ str(latitude))
print ('Longitude: '+ str(longitude))

API_url = 'http://api.planetos.com/v1/datasets/noaa_ww3_global_1.25x1d/point?lat={0}&lon={1}&verbose=true&apikey={2}&count=100&end={3}&reftime_start={4}&reftime_end={5}'.format(latitude,longitude,apikey,end,reftime_start,reftime_end)
request = Request(API_url)
response = urlopen(request)
API_data_ww3 = json.loads(response.read())
print(API_url)

ww3_variables = []
for k,v in set([(j,i['context']) for i in API_data_ww3['entries'] for j in i['data'].keys()]):
    ww3_variables.append(k)

print(ww3_variables)
print(buoy_variables)

buoy_model = {'wave_height':'Significant_height_of_combined_wind_waves_and_swell_surface',
              'mean_wave_dir':'Primary_wave_direction_surface',
             'average_wpd':'Primary_wave_mean_period_surface',
             'wind_spd':'Wind_speed_surface'}

def append_data(in_string):
    if in_string == None:
        return np.nan
    elif in_string == 'None':
        return np.nan
    else:
        return float(in_string)

ww3_data = {}
ww3_times = {}
buoy_data = {}
buoy_times = {}
for k,v in buoy_model.items():
    ww3_data[v] = []
    ww3_times[v] = []
    buoy_data[k] = []
    buoy_times[k] = []

for i in API_data_ww3['entries']:
    for j in i['data']:
        if j in buoy_model.values():
            ww3_data[j].append(append_data(i['data'][j]))
            ww3_times[j].append(dateutil.parser.parse(i['axes']['time']))
            
for i in API_data_buoy['entries']:
    for j in i['data']:
        if j in buoy_model.keys():
            buoy_data[j].append(append_data(i['data'][j]))
            buoy_times[j].append(dateutil.parser.parse(i['axes']['time']))
for i in ww3_data:
    ww3_data[i] = np.array(ww3_data[i])
    ww3_times[i] = np.array(ww3_times[i])

buoy_label = "NDBC Station %s" % station
ww3_label = "WW3 at %s" % reftime_start
for k,v in buoy_model.items():
    if np.abs(np.nansum(buoy_data[k]))>0:
        fig=plt.figure(figsize=(10,5))
        plt.title(k+'  '+v)
        plt.plot(ww3_times[v],ww3_data[v], label=ww3_label)
        plt.plot(buoy_times[k],buoy_data[k],'*',label=buoy_label)
        plt.legend(bbox_to_anchor=(1.5, 0.22), loc=1, borderaxespad=0.)
        plt.xlabel('Time')
        plt.ylabel(k)
        fig.autofmt_xdate()
        plt.grid()



