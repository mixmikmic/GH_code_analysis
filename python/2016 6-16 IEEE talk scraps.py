import numpy as np
from sklearn.decomposition import PCA
import scipy.io
import itertools
import os
import h5py
import pyret.visualizations as pyviz
import pyret.filtertools as ft
import pyret.spiketools as st
import jetpack
from experiments.iotools import read_channel # from niru-analysis github
from experiments.photodiode import find_peaks, find_start_times
# import binary     # in igor >> recording
import pdb
import string
# from jetpack.signals import peakdet
from scipy.signal import find_peaks_cwt
from os.path import expanduser

# This is a bit of magic to make matplotlib figures appear inline in the
# notebook rather than in a new window.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, title, imshow

# note that nonposx(y) for log plots will no longer work with this package
import mpld3
#mpld3.enable_notebook()

from pylab import rcParams
rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

bipolars = [45, 75, 80, 150, 175]
ganglions = [190, 210, 220, 240, 245]
features = [150, 160, 161, 163, 155, 160, 153]

hist_b, bins_b = np.histogram(bipolars, bins=np.linspace(40, 250, 20))
hist_g, bins_g = np.histogram(ganglions, bins=np.linspace(40, 250, 20))
hist_f, bins_f = np.histogram(features, bins=np.linspace(40, 250, 20))

from deepretina.io import despine

plt.bar(bins_b[:-1], hist_b, width=11, color='k', alpha=0.7)
plt.bar(bins_g[:-1], hist_g, width=11, color='g', alpha=0.7, bottom=hist_b)
plt.bar(bins_f[:-1], hist_f, width=11, color='m', alpha=0.7, bottom=hist_b)
plt.axis('off')



