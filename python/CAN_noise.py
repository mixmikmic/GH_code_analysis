# Preparation for programming
# Make sure to execute this cell first!
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')                  # do not show warnings
from __future__ import print_function
from scipy import interpolate, signal
from time import *
from obspy import *
from obspy.core import read, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.cross_correlation import xcorr_pick_correction
import numpy as np
import matplotlib.pylab as plt
import os
import glob
import wave
import struct
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['lines.linewidth'] = 1

# Getting the waveforms

client = Client("IRIS")
t = UTCDateTime("2004-12-26T00:58:53.0")
starttime = t-(9*24*3600) # 10 days before the Sumatra earthquake
endtime = t # the original time of the Sumatra earthquake

st = client.get_waveforms("G", "CAN", "*", "LHZ", starttime, endtime, attach_response=True)
print(st)

st.plot()

# Plotting signals
tr = st[0]
trace_data = tr.data
plt.plot(tr.data)

# Remove obvious earthquake 
t1 = UTCDateTime("2004-12-23T14:59:30.9")
starttime1 = t1-(90*3600) # 90 hours before the New Zealand earthquake
endtime1 = t1 # the original time of the New Zealand earthquake

st1 = client.get_waveforms("G", "CAN", "*", "LHZ", starttime1, endtime1, attach_response=True)
print(st)

st1.plot()

# Plotting signals
tr1 = st1[0]
trace_data = tr1.data
plt.plot(tr1.data)



