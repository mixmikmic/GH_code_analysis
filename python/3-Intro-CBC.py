# Standard python numerical analysis imports:
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

# the IPython magic below must be commented out in the .py file, since it doesn't work there.
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# -- Handy function to download data file, and return the filename
def download(url):
    filename = url.split('/')[-1]
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

download('https://losc.ligo.org/s/sample_code/readligo.py')
download('https://losc.ligo.org/s/events/BBH_events_v3.json')
download('https://losc.ligo.org/s/sample_code/readligo.py')
fn_H1 = download('https://losc.ligo.org/s/events/GW150914/H-H1_LOSC_4_V2-1126259446-32.hdf5')
fn_L1 = download('https://losc.ligo.org/s/events/GW150914/L-L1_LOSC_4_V2-1126259446-32.hdf5')
fn_template = download('https://losc.ligo.org/s/events/GW150914/GW150914_4_template.hdf5')

import readligo as rl
strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')

