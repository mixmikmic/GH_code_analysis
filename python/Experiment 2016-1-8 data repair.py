import numpy as np
import scipy.io
import itertools
import os
import h5py
import pyret
#import binary     # in igor >> recording
import pdb
import string
#from jetpack.signals import peakdet
from scipy.signal import find_peaks_cwt

# This is a bit of magic to make matplotlib figures appear inline in the
# notebook rather than in a new window.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, title, imshow

# note that nonposx(y) for log plots will no longer work with this package
#import mpld3
#mpld3.enable_notebook()

from pylab import rcParams
rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

data_dir = os.path.expanduser('~/experiments/data/16-01-08/')
h5_file = '16-1-8.h5'

f = h5py.File(data_dir+h5_file, 'r+')

f['data'].shape

channel_means = np.mean(f['data'], axis=1)

our_channel_means = channel_means[4:]
our_channel_means.shape

array = 'mcs'

f['data'].attrs['array'] = array

f['data'].attrs['channel-means'] = our_channel_means

f.close()



