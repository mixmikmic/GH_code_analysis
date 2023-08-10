# Standard python numerical analysis imports:
#!pip install gwpy
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import h5py
import json
import sys

pyversion = sys.version_info.major
if pyversion == 2: 
    import urllib2
else:
    import urllib.request
    
import os
from gwpy.timeseries import TimeSeries

# the IPython magic below must be commented out in the .py file, since it doesn't work there.
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

try: os.mkdir('./data')
except: pass

# -- Handy function to download data file, and return the filename
def download(url):
    filename = os.path.join('data', url.split('/')[-1])
    print('Downloading ' + url )
    if pyversion == 2: 
        r = urllib2.urlopen(url).read()
        f = open(filename, 'w')   # write it to the right filename
        f.write(r)
        f.close()
    else:
        urllib.request.urlretrieve(url, filename)  
    print("File download complete")
    return filename

## Make a spectrogram of a hardware injection
gps_inj = 1128668183
url_inj = 'https://losc.ligo.org/archive/data/O1/1128267776/H-H1_LOSC_4_V1-1128665088-4096.hdf5'
fn_inj = download(url_inj)

gps = gps_inj
fn = fn_inj
inj_data = TimeSeries.read(fn, format='hdf5.losc', start=gps-16, end=gps+16)
# -- Follow example at https://gwpy.github.io/docs/stable/examples/timeseries/qscan.html
plot = inj_data.q_transform().crop(gps-.15, gps+.05).plot(figsize=[8, 4])
ax = plot.gca()
ax.set_epoch(gps)
ax.set_yscale('log')
ax.set_xlabel('Time [milliseconds]')
ax.set_ylim(20, 500)
ax.grid(True, axis='y', which='both')
plot.add_colorbar(cmap='viridis', label='Normalized energy')

# -- Koi fish glitch
gps = 1132401286.330
url = 'https://losc.ligo.org/archive/data/O1/1131413504/H-H1_LOSC_4_V1-1132400640-4096.hdf5'
fn = download(url)

data = TimeSeries.read(fn, format='hdf5.losc', start=gps-16, end=gps+16)

# -- Follow example at https://gwpy.github.io/docs/stable/examples/timeseries/qscan.html
plot = data.q_transform().crop(gps-.15, gps+.05).plot(figsize=[8, 4])
ax = plot.gca()
ax.set_epoch(gps)
ax.set_yscale('log')
ax.set_xlabel('Time [milliseconds]')
ax.set_ylim(20, 500)
ax.grid(True, axis='y', which='both')
plot.add_colorbar(cmap='viridis', label='Normalized energy')

# -- Scratchy glitch
gps = 1128779800.440
url = 'https://losc.ligo.org/archive/data/O1/1128267776/H-H1_LOSC_4_V1-1128779776-4096.hdf5'
fn = download(url)

data = TimeSeries.read(fn, format='hdf5.losc', start=gps-16, end=gps+16)

# -- Follow example at https://gwpy.github.io/docs/stable/examples/timeseries/qscan.html
plot = data.q_transform().crop(gps-.15, gps+.05).plot(figsize=[8, 4])
ax = plot.gca()
ax.set_epoch(gps)
ax.set_yscale('log')
ax.set_xlabel('Time [milliseconds]')
ax.set_ylim(20, 500)
ax.grid(True, axis='y', which='both')
plot.add_colorbar(cmap='viridis', label='Normalized energy')

