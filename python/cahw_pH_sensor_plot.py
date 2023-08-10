import netrc
netrc = netrc.netrc()
remoteHostName = "ooinet.oceanobservatories.org"
info = netrc.authenticators(remoteHostName)
username = info[0]
token = info[2]

import matplotlib.pyplot as plt
import matplotlib.dates as dates
import netCDF4 as nc
import numpy as np

import requests
import datetime

subsite = 'RS01SBPS'
node = 'PC01A'
sensor = '4B-PHSENA102'
method = 'streamed'
stream = 'phsen_data_record'
beginDT = '2017-08-21T07:00:00.000Z'
# beginDT = (datetime.datetime.utcnow() - datetime.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
endDT = '2017-08-22T07:00:00.000Z'

base_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv'

data_request_url ='/'.join((base_url,subsite,node,sensor,method,stream))
params = {
    'beginDT':beginDT,
    'endDT':endDT,
    'limit':20000,   
}

r = requests.get(data_request_url, params=params,auth=(username, token))
data = r.json()

len(data)
data[0]

time = []
pH = []
hour = []

for i in range(len(data)):
    time.append(nc.num2date(data[i]['time'],'seconds since 1900-01-01').replace(microsecond=0))
    pH.append(data[i]['ph_seawater'])
    hour.append((data[i]['time']-data[0]['time'])/3600)

len(pH)

plt.plot(time, pH, marker=".", markersize=1, linestyle=None)
plt.ylabel('pH (total scale)')
plt.xlabel('Time')
plt.xticks(rotation=30)
#plt.tight_layout()
plt.show()

