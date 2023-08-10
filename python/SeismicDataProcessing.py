import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import copy
import pickle
from datetime import datetime
from obspy import UTCDateTime
import sys

#print(sys.path)

from obspy.clients.fdsn import Client

#client_name='IRIS'
client_name='http://service.iris.edu'

client=Client(client_name)

print(client)

from obspy.clients.fdsn.header import URL_MAPPINGS

for key in sorted(URL_MAPPINGS.keys()):
    print("{0:<7} {1}".format(key,  URL_MAPPINGS[key]))

### Parameters

#net_code='NV'
net_code='NV,OO'

starttime = UTCDateTime("2015-01-01")
endtime = UTCDateTime("2015-01-02")

### Read network data

inventory = client.get_stations(network=net_code,
                                starttime=starttime,
                                endtime=endtime)

print(inventory)

### Plot inventory
plt.ioff()
inventory.plot(color_per_network=True) ### Plot global
inventory.plot(projection='local',color_per_network=True) ### Plot local
plt.show()

inventory = client.get_stations(network='OO',station='AXAS1,AXAS2',channel='*Z',
                                starttime=starttime,
                                endtime=endtime,level="channel")
print(inventory)

inventory_select = inventory.select(channel="*Z", station="AXAS1")
print(inventory_select)

inventory = client.get_stations(network='OO',station='AXAS1',channel='EHZ',
                                starttime=starttime,
                                endtime=endtime,level="response")

inventory.plot_response(min_freq=0.001)
plt.show()

inventory = client.get_stations(network='OO',station='AXAS1',channel='*',
                                starttime=starttime,
                                endtime=endtime,level="response")

#help(inventory.write) # to check available format
inventory.write('station.xml',format='STATIONXML')

starttime = UTCDateTime("2015-01-22T10:34:21")
duration = 20

st = client.get_waveforms(network='OO',station='AX*',location="*",channel='E*Z',
                                starttime=starttime,
                                endtime=starttime+duration)

print(type(st))
print(st)
print(st.print_gaps())

trace=st[0]
print(type(trace))
print(trace.stats)
print(trace.times())
print(trace.data)

st.plot()

starttime = UTCDateTime("2015-01-22T00:00:00")

st = client.get_waveforms(network='OO',station='AXAS2',location="*",channel='E*Z',
                                starttime=starttime,
                                endtime=starttime+86400)

trace=st[0]
trace.plot(type='dayplot',color=['k', 'r', 'b', 'g'],interval=60) ## interval is in minutes and can be changed

st_filled_gaps=None
st_filled_gaps=copy.deepcopy(st)
#print(st_filled_gaps[0].stats)
#print(st_filled_gaps[1].stats)
st_filled_gaps[0].stats.sampling_rate=200  # allow merging
st_filled_gaps.merge(method=0, fill_value=0)

trace=st_filled_gaps[0]
trace.plot(type='dayplot',color=['k', 'r', 'b', 'g'],interval=60) ## interval is in minutes and can be changed

starttime = UTCDateTime("2015-01-22T10:34:25")
duration = 20

st = client.get_waveforms(network='OO',station='AXAS1',location="*",channel='E*Z',
                                starttime=starttime,
                                endtime=starttime+duration)

st.merge()
trace=st[0]

trace.plot()

#### Apply different to the trace

trace_filter=copy.copy(trace)
trace_filter.filter('bandpass', freqmin=5.0, freqmax=50,corners=3, zerophase=True)
trace_filter.plot()

trace_filter=copy.copy(trace)
trace_filter.filter('highpass', freq=50.0, corners=2, zerophase=True)
trace_filter.plot()

trace_filter=copy.copy(trace)
trace_filter.filter('lowpass', freq=5.0, corners=2, zerophase=True)
trace_filter.plot()

starttime = UTCDateTime("2015-01-22T00:00:00")
duration = 60*10 # Let's take 10 minutes

st = client.get_waveforms(network='OO',station='AXAS1',location="*",channel='E*Z',
                                starttime=starttime,
                                endtime=starttime+duration)

st.merge()
trace=st[0]

trace_filter=copy.copy(trace)
trace_filter.filter('bandpass', freqmin=5.0, freqmax=50,corners=3, zerophase=True)

trace_filter.plot()

#fig,ax=plt.subplots()

cmap= matplotlib.cm.get_cmap('jet')

trace_filter.spectrogram(cmap=cmap,wlen=20,per_lap=0.9)

starttime = UTCDateTime("2015-01-22T10:34:00")
duration = 60 # Let's take 10 minutes

st = client.get_waveforms(network='OO',station='AXAS1',location="*",channel='E*Z',
                                starttime=starttime,
                                endtime=starttime+duration,attach_response=True)

trace=st[0]

trace_sensi=trace.copy()
trace_sensi.remove_sensitivity()  # This removes the multiplication factor applied to the seismogram
trace.plot()
trace_sensi.plot()
pre_filt = [0.5, 1, 50, 70]
trace.remove_response(output="DISP",plot=True,water_level=0)

trace_filter=trace.copy()

trace_filter.filter('bandpass', freqmin=5.0, freqmax=50,corners=3)
trace_filter.plot()


st = client.get_waveforms(network='OO',station='AXAS1',location="*",channel='E*Z',
                                starttime=starttime,
                                endtime=starttime+duration,attach_response=True)
st.write('mytrace.mseed',format='MSEED')

from obspy import read

new_st=read('mytrace.mseed',format='MSEED')

print(new_st)

