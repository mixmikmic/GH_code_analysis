username =''
token = ''

import netrc
netrc = netrc.netrc()
remoteHostName = "ooinet.oceanobservatories.org"
info = netrc.authenticators(remoteHostName)
username = info[0]
token = info[2]

import requests
import time

subsite = 'RS03ASHS'
node = 'MJ03B'
sensor = '07-TMPSFA301'
method = 'streamed'
stream = 'tmpsf_sample'
beginDT = '2014-09-27T01:01:01.000Z' #begin of first deployement
endDT = None

base_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/'

data_request_url ='/'.join((base_url,subsite,node,sensor,method,stream))
params = {
    'beginDT':beginDT,
    'endDT':endDT,   
}
r = requests.get(data_request_url, params=params, auth=(username, token))
data = r.json()

print(data['allURLs'][0])

print(data['allURLs'][1])

get_ipython().run_cell_magic('time', '', "check_complete = data['allURLs'][1] + '/status.txt'\nfor i in range(1800): \n    r = requests.get(check_complete)\n    if r.status_code == requests.codes.ok:\n        print('request completed')\n        break\n    else:\n        time.sleep(1)")

import re
import xarray as xr
import pandas as pd
import os

# url = data['allURLs'][0]
url = 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/ooidatateam@gmail.com/20180221T030103-RS03ASHS-MJ03B-07-TMPSFA301-streamed-tmpsf_sample/catalog.html'
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

datasets

ds = xr.open_mfdataset(datasets)
ds = ds.swap_dims({'obs': 'time'})
ds = ds.chunk({'time': 100})
ds

import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import numpy as np

ds['temperature12'].plot()
plt.show()

get_ipython().run_cell_magic('time', '', "from dask.diagnostics import ProgressBar\nwith ProgressBar():\n    df = ds['temperature12'].to_dataframe()\n    df = df.resample('min').mean()")

get_ipython().run_cell_magic('time', '', "plt.close()\nfig, ax = plt.subplots()\nfig.set_size_inches(16, 6)\ndf['temperature12'].plot(ax=ax)\ndf['temperature12'].resample('H').mean().plot(ax=ax)\ndf['temperature12'].resample('D').mean().plot(ax=ax)\nplt.show()")

get_ipython().run_cell_magic('time', '', 'time = []\ntime_pd = pd.to_datetime(ds.time.values.tolist())\nfor i in time_pd:\n    i = np.datetime64(i).astype(datetime.datetime)\n    time.append(dates.date2num(i)) ')

temperature = ds['temperature12'].values.tolist()

plt.close()
fig, ax = plt.subplots()
fig.set_size_inches(16, 6)

hb1 = ax.hexbin(time, temperature, bins='log', vmin=0.4, vmax=3, gridsize=(1100, 100), mincnt=1, cmap='Blues')
fig.colorbar(hb1)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
# ax.set_xlim(datetime.datetime(2015, 12, 1, 0, 0),datetime.datetime(2016, 7, 25, 0, 0))
# ax.set_ylim(2,11)
years = dates.YearLocator()
months = dates.MonthLocator()
yearsFmt = dates.DateFormatter('\n\n\n%Y')
monthsFmt = dates.DateFormatter('%b')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.xaxis.set_minor_locator(years)
ax.xaxis.set_minor_formatter(yearsFmt)
plt.tight_layout()
plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
plt.ylabel('Temperature $^\circ$C')
plt.xlabel('Time')
plt.show()



