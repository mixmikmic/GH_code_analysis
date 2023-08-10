username ='OOIAPI-I0UCP16V67ENKZ'
token = '4CUPODF4AL256S'

import requests
import time
subsite = 'RS01SBPS'
node = 'PC01A'
sensor = '05-ADCPTD102'
method = 'streamed'
stream = 'adcp_velocity_beam'
beginDT = '2017-08-21T00:00:00.000Z' #begin of first deployement
# endDT = None
endDT = '2017-08-22T23:59:59.000Z' #begin of first deployement

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

check_complete = data['allURLs'][1] + '/status.txt'
for i in range(1800): 
    r = requests.get(check_complete)
    if r.status_code == requests.codes.ok:
        print('request completed')
        break
    else:
        time.sleep(1)

import re
import xarray as xr
import pandas as pd
import os

url = data['allURLs'][0]
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

ds = xr.open_mfdataset(datasets)
ds = ds.swap_dims({'obs': 'time'})
ds = ds.chunk({'time': 100})
ds

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date

depth_vec = ds.bin_depths.values[0]

depth_max = max(depth_vec)
depth_min =min(depth_vec)
echo_intb1 = ds.corrected_echo_intensity_beam1.values.transpose()
echo_intb2 = ds.corrected_echo_intensity_beam2.values.transpose()
echo_intb3 = ds.corrected_echo_intensity_beam3.values.transpose()
echo_intb4 = ds.corrected_echo_intensity_beam4.values.transpose()
sz = echo_intb1.shape
print('Size of Dataset (pings,depth): ', sz)

depth_idx_start = np.searchsorted(depth_vec, depth_min, side="left")
depth_idx_end = np.searchsorted(depth_vec, depth_max, side="right")
if depth_idx_end>=depth_vec.shape[0]:
    depth_idx_end = depth_vec.shape[0]-1
    
depth_ticks_num = 5
y_ticks_spacing = sz[0]/(depth_ticks_num-1)
y_ticks = np.arange(depth_ticks_num)*y_ticks_spacing
depth_spacing = np.around((depth_max-depth_min)/(depth_ticks_num-1),decimals=1)
depth_label = np.around((np.arange(depth_ticks_num)*depth_spacing)+depth_min,decimals=1)

#num_ping_to_plot = 12000
num_ping_to_plot = sz[1]
num_ping_label = 10
x_ticks = np.int32(np.arange(0,num_ping_label)*num_ping_to_plot/num_ping_label)
x_ticks_label = [x.strftime('%H:%M') for x in pd.to_datetime(ds.time.values[x_ticks])]

fig = plt.figure(figsize=(15,5))
plt.imshow(echo_intb1[::-1,0:num_ping_to_plot],aspect='auto',vmax=np.ceil(np.amax(echo_intb1)),vmin=np.floor(np.amin(echo_intb1)),cmap='jet')
plt.xticks(x_ticks,x_ticks_label,fontsize=12)
plt.yticks(y_ticks,depth_label,fontsize=12)
plt.xlabel('Time (hour:min)',fontsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.colorbar()
plt.show()

fig = plt.figure(figsize=(20,10))
plt.figure(1)

plt.subplot(221)
plt.imshow(echo_intb1[::-1,0:num_ping_to_plot],aspect='auto',vmax=np.ceil(np.amax(echo_intb1)),vmin=np.floor(np.amin(echo_intb1)),cmap='jet')
plt.xticks(x_ticks,x_ticks_label,fontsize=12)
plt.yticks(y_ticks,depth_label,fontsize=12)
#plt.xlabel('Time (hour:min)',fontsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.title('EI Beam 1')
plt.colorbar()

plt.subplot(222)
plt.imshow(echo_intb2[::-1,0:num_ping_to_plot],aspect='auto',vmax=np.ceil(np.amax(echo_intb1)),vmin=np.floor(np.amin(echo_intb1)),cmap='jet')
plt.xticks(x_ticks,x_ticks_label,fontsize=12)
plt.yticks(y_ticks,depth_label,fontsize=12)
#plt.xlabel('Time (hour:min)',fontsize=14)
#plt.ylabel('Depth (m)',fontsize=14)
plt.title('EI Beam 2')
plt.colorbar()


plt.subplot(223)
plt.imshow(echo_intb3[::-1,0:num_ping_to_plot],aspect='auto',vmax=np.ceil(np.amax(echo_intb1)),vmin=np.floor(np.amin(echo_intb1)),cmap='jet')
plt.xticks(x_ticks,x_ticks_label,fontsize=12)
plt.yticks(y_ticks,depth_label,fontsize=12)
plt.xlabel('Time (hour:min)',fontsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.title('EI Beam 3')
plt.colorbar()

# logit
plt.subplot(224)
plt.imshow(echo_intb4[::-1,0:num_ping_to_plot],aspect='auto',vmax=np.ceil(np.amax(echo_intb1)),vmin=np.floor(np.amin(echo_intb1)),cmap='jet')
plt.xticks(x_ticks,x_ticks_label,fontsize=12)
plt.yticks(y_ticks,depth_label,fontsize=12)
plt.xlabel('Time (hour:min)',fontsize=14)
#plt.ylabel('Depth (m)',fontsize=14)
plt.title('EI Beam 4')
plt.colorbar()

plt.show()

fig,ax = plt.subplots(2,2,figsize=(16,8))
ds['echo_intensity_beam1'].T.plot(ax=ax[0,0])
ds['echo_intensity_beam2'].T.plot(ax=ax[0,1])
ds['echo_intensity_beam3'].T.plot(ax=ax[1,0])
ds['echo_intensity_beam4'].T.plot(ax=ax[1,1])
plt.show()

def mean_weighted_depth(ds, echo_variable_name):
    mwd = []
    bd = ds.bin_depths.values
    eb = ds[echo_variable_name].values
    for t in range(eb.shape[0]):
        cur_ping = sum(eb[t,:]*bd[t,:])/sum(eb[t,:])
        mwd.append(cur_ping)
    mwd = pd.DataFrame(mwd,ds.time.values)
    return mwd

echo_variables = ['corrected_echo_intensity_beam1','corrected_echo_intensity_beam2','corrected_echo_intensity_beam3','corrected_echo_intensity_beam4']
eb1 = mean_weighted_depth(ds,echo_variables[0])
eb2 = mean_weighted_depth(ds,echo_variables[1])
eb3 = mean_weighted_depth(ds,echo_variables[2])
eb4 = mean_weighted_depth(ds,echo_variables[3])

fig = plt.figure(figsize=(15,20))
plt.figure(1)

plt.subplot(411)
plt.plot(eb1)
plt.plot(eb1.resample('60S').mean())
plt.xlabel('Date (MM-DD HH)')
plt.ylabel('Mean-Weighted Depth')
plt.title('Beam 1')
plt.legend(['1.3 Hz Sampling', '60 Second Mean'])


plt.subplot(412)
plt.plot(eb2)
plt.plot(eb2.resample('60S').mean())
plt.xlabel('Date (MM-DD HH)')
plt.ylabel('Mean-Weighted Depth')
plt.title('Beam 2')
plt.legend(['1.3 Hz Sampling', '60 Second Mean'])


plt.subplot(413)
plt.plot(eb3)
plt.plot(eb3.resample('60S').mean())
plt.xlabel('Date (MM-DD HH)')
plt.ylabel('Mean-Weighted Depth')
plt.title('Beam 3')
plt.legend(['1.3 Hz Sampling', '60 Second Mean'])


plt.subplot(414)
plt.plot(eb4)
plt.plot(eb4.resample('60S').mean())
plt.xlabel('Date (MM-DD HH)')
plt.ylabel('Mean-Weighted Depth')
plt.title('Beam 4')
plt.legend(['1.3 Hz Sampling', '60 Second Mean'])
plt.show()

eb_mean = (eb1+eb2+eb3+eb4)/4
fig = plt.figure(figsize=(15,5))
plt.figure(1)
plt.plot(eb_mean.resample('300S').mean())

