import re
import time
import xarray as xr
import pandas as pd
import os
import datetime
import requests
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import netCDF4 as nc
from IPython import display

import netrc
netrc = netrc.netrc()
remoteHostName = "ooinet.oceanobservatories.org"
info = netrc.authenticators(remoteHostName)
username = info[0]
token = info[2]

subsite = 'RS01SBPS'
node = 'SF01A'
sensor = '2A-CTDPFA102'
method = 'streamed'
stream = 'ctdpf_sbe43_sample'
beginDT = '2017-08-21T07:00:00.000Z'
endDT = '2017-08-22T07:00:00.000Z'

base_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/'

data_request_url ='/'.join((base_url,subsite,node,sensor,method,stream))
params = {
    'beginDT':beginDT,
    'endDT':endDT,   
}
r = requests.get(data_request_url, params=params, auth=(username, token))
data = r.json()

data['allURLs'][1]
data['allURLs'][0]

get_ipython().run_cell_magic('time', '', "check_complete = data['allURLs'][1] + '/status.txt'\nfor i in range(1800): \n    r = requests.get(check_complete)\n    if r.status_code == requests.codes.ok:\n        print('request completed')\n        break\n    else:\n        time.sleep(1)")

url = data['allURLs'][0]
# url = 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/friedrich.knuth@rutgers.edu/20180221T052204-RS03AXPS-SF03A-2A-CTDPFA302-streamed-ctdpf_sbe43_sample/catalog.html'
tds_url = 'https://opendap.oceanobservatories.org/thredds/dodsC'
datasets = requests.get(url).text
urls = re.findall(r'href=[\'"]?([^\'" >]+)', datasets)
x = re.findall(r'(ooi/.*?.nc)', datasets)
for i in x:
    if i.endswith('.nc') == False:
        x.remove(i)
for i in x:
    try:
        float(i[-4])
    except:
        x.remove(i)
datasets = [os.path.join(tds_url, i) for i in x]

ds_ = xr.open_mfdataset(datasets)
ds = ds.swap_dims({'obs': 'time'})
ds

x = ds['time'].values
y = ds['seawater_pressure'].values
z = ds['corrected_dissolved_oxygen'].values
fig, ax = plt.subplots()
fig.set_size_inches(16, 6);
ax.invert_yaxis()
ax.grid()
ax.set_xlim(x[0],x[-1])
sc = plt.scatter(x, y, c=z)
cb = fig.colorbar(sc, ax=ax)
cb.update_ticks()
cb.formatter.set_useOffset(False)
plt.show()



