get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import dateutil.parser
import datetime
from urllib.request import urlopen, Request
import simplejson as json

location = 'Hawaii Oahu'
if location == 'Est':
    longitude = 24.+45./60
    latitude = 59+25/60.
elif location == 'Au':
    longitude = 149. + 7./60
    latitude = -35.-18./60
elif location == "Hawaii Oahu":
    latitude = 21.205
    longitude = -158.35

apikey = 'YOUR-API-KEY-GOES-HERE'
API_url = "http://api.planetos.com/v1/datasets/pacioos_swan_oahu/point?lon={0}&lat={1}&count=1000&verbose=false&apikey={2}".format(longitude,latitude,apikey)
print(API_url)
request = Request(API_url)
response = urlopen(request)
API_data = json.loads(response.read())

print("{0:<50} {1}".format("Variable","Context"))
print()
for k,v in set([(j,i['context']) for i in API_data['entries'] for j in i['data'].keys()]):
    print("{0:<50} {1}".format(k,v))

reftimes = set()
for i in API_data['entries']:
    reftimes.update([i['axes']['reftime']])
reftimes=list(reftimes)
if len(reftimes)>1:
    reftime = reftimes[0] if dateutil.parser.parse(reftimes[0])<dateutil.parser.parse(reftimes[1]) else reftimes[1]
else:
    reftime = reftimes[0]

reftime

varlist = ['mdir','mper','pdir','shgt','pper']

vardict = {}
for i in varlist:
    vardict['time_'+i]=[]
    vardict['data_'+i]=[]
for i in API_data['entries']:
    #print(i['context'])
    if i['context'] == 'reference_time_time_z_lat_lon' and i['axes']['reftime']==reftime:
        for j in i['data']:
            vardict['data_'+j].append(i['data'][j])
            vardict['time_'+j].append(dateutil.parser.parse(i['axes']['time']))

for i in varlist:
    fig = plt.figure(figsize=(15,3))
    ax = fig.add_subplot(111)
    plt.plot(vardict['time_'+i],vardict['data_'+i],color='r')
    ax.set_ylabel(i)

