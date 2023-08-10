get_ipython().magic('matplotlib inline')
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 12, 8

from obspy import read_inventory
# real-world StationXML files often deviate from the official schema definition
# therefore file-format autodiscovery sometimes fails and we have to force the file format
inventory = read_inventory("./data/station_PFO.xml", format="STATIONXML")
print(type(inventory))

get_ipython().system('head data/station_BFO.xml')

print(inventory)

network = inventory[0]
print(network)

station = network[0]
print(station)

channel = station[0]
print(channel)

print(channel.response)

from obspy import read
st = read("./data/waveform_PFO.mseed")
print(st)

inv = read_inventory("./data/station_PFO.xml", format="STATIONXML")
st.attach_response(inv)

print(st[0].stats)

st.plot()
st.remove_response()
st.plot()

st = read("./data/waveform_PFO.mseed")
st.attach_response(inv)
st.remove_response(water_level=60, pre_filt=(0.01, 0.02, 8, 10), output="DISP")
st.plot()

