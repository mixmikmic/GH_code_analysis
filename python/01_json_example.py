username = ''
token = ''

# import netrc
# netrc = netrc.netrc()
# remoteHostName = "ooinet.oceanobservatories.org"
# info = netrc.authenticators(remoteHostName)
# username = info[0]
# token = info[2]

import requests
import datetime

subsite = 'RS03ASHS'
node = 'MJ03B'
sensor = '07-TMPSFA301'
method = 'streamed'
stream = 'tmpsf_sample'
beginDT = '2017-09-04T10:01:01.000Z'
# beginDT = (datetime.datetime.utcnow() - datetime.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
endDT = '2017-09-05T10:01:01.000Z'

base_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/'

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

import matplotlib.pyplot as plt
import matplotlib.dates as dates
import netCDF4 as nc
import numpy as np

time = []
temperature01 = []

for i in range(len(data)):
    time.append(nc.num2date(data[i]['time'],'seconds since 1900-01-01').replace(microsecond=0))
    temperature01.append(data[i]['temperature01'])

plt.plot(time, temperature01, marker=".", markersize=1, linestyle=None)
plt.ylabel('Temperature $^\circ$C')
plt.xlabel('Time')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

temperature03 = []
for i in range(len(data)):
    temperature03.append(data[i]['temperature03'])

fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
ax1.plot(time, temperature01, marker=".", markersize=1, linestyle=None)
ax2.plot(time, temperature03, marker=".", markersize=1, linestyle=None)
ax1.set_ylabel('T01 $^\circ$C')
ax2.set_ylabel('T03 $^\circ$C')
plt.xlabel('Time')
plt.xticks(rotation=30)
plt.tight_layout()
# fig.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
plt.show()

subsite = 'RS03ASHS'
node = 'MJ03B'
sensor = '09-BOTPTA304'
method = 'streamed'
stream = 'botpt_nano_sample'
beginDT = '2017-09-04T10:01:01.000Z'
endDT = '2017-09-05T10:01:01.000Z'

data_request_url ='/'.join((base_url,subsite,node,sensor,method,stream))
params = {
    'beginDT':beginDT,
    'endDT':endDT,
    'limit':8640,   
}

r = requests.get(data_request_url, params=params,auth=(username, token))
data = r.json()

botpt_time = []
bottom_pressure = []
for i in range(len(data)):
    botpt_time.append(nc.num2date(data[i]['time'],'seconds since 1900-01-01').replace(microsecond=0))
    bottom_pressure.append(data[i]['bottom_pressure'])

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
ax1.plot(time, temperature01, marker=".", markersize=1, linestyle=None)
ax2.plot(time, temperature03, marker=".", markersize=1, linestyle=None)
ax3.plot(botpt_time, bottom_pressure, marker=".", markersize=1, linestyle=None)
ax1.set_ylabel('T01 $^\circ$C')
ax2.set_ylabel('T03 $^\circ$C')
ax3.set_ylabel('psia')
plt.xlabel('Time')
plt.xticks(rotation=30)
# plt.tight_layout()
# fig.subplots_adjust(hspace=0)
# plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
plt.show()

# Import Bokeh functions
import os
from bokeh.plotting import figure, output_file, reset_output, show, ColumnDataSource, save
from bokeh.layouts import column
from bokeh.models import BoxAnnotation
from bokeh.io import output_notebook # required to display Bokeh visualization in notebook

source = ColumnDataSource(
    data=dict(
        x=time,
        y1=temperature01,
        y2=temperature03,
        y3=bottom_pressure,
    )
)

s1 = figure(width=600,
           height=400,
           title='Temperature01',
           x_axis_label='Time (GMT)',
           y_axis_label='T01 °C',
           x_axis_type='datetime')

s1.line('x', 'y1', line_width=3, source=source)
s1.circle('x', 'y1', fill_color='white', size=4, source=source)

s2 = figure(width=600,
           height=400,
           title='Temperature01',
           x_axis_label='Time (GMT)',
           y_axis_label='T03 °C',
           x_axis_type='datetime')

s2.line('x', 'y2', line_width=3, source=source)
s2.circle('x', 'y2', fill_color='white', size=4, source=source)

s3 = figure(width=600,
           height=400,
           title='Bottom Pressure',
           x_axis_label='Time (GMT)',
           y_axis_label='psia',
           x_axis_type='datetime')

s3.line('x', 'y3', line_width=3, source=source)
s3.circle('x', 'y3', fill_color='white', size=4, source=source)

output_notebook()
show(column(s1, s2, s3))

# output_file(os.getcwd())
# save(s1, filename='temperature01.html')
# save(s2, filename='temperature03.html')
# save(s3, filename='bottom_pressure.html')



