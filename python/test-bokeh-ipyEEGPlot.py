# load the eegvis libaries and some others we might use

from __future__ import print_function, division, unicode_literals
import os.path
import pandas as pd
import numpy as np
import h5py

import ipywidgets
from IPython.display import display

import eegvis.stacklineplot
import eegvis.montageview as montageview
import eegvis.stackplot_bokeh as sbokplot
from bokeh.io import output_notebook, push_notebook
#import bokeh.plotting as bplt
#from bokeh.plotting import show
output_notebook()
ARCHIVEDIR = r'../../eeg-hdfstorage/notebooks/archive'

#hdf = h5py.File('./archive/YA2741BS_1-1+.eeghdf') # 5mo boy 
hdf = h5py.File(os.path.join(ARCHIVEDIR,'YA2741G2_1-1+.eeghdf')) # absence 10yo

rec = hdf['record-0']
years_old = rec.attrs['patient_age_days']/365

signals = rec['signals']
labels = rec['signal_labels']
electrode_labels = [str(s,'ascii') for s in labels]
ref_labels = montageview.standard2shortname(electrode_labels)
numbered_electrode_labels = ["%d:%s" % (ii, str(labels[ii], 'ascii')) for ii in range(len(labels))]

inr = sbokplot.IpyEEGPlot(signals, 15, electrode_labels=electrode_labels, fs=rec.attrs['sample_frequency'])
inr.show()

smallerplot = sbokplot.IpyEEGPlot(signals, 15, electrode_labels=electrode_labels, fs=rec.attrs['sample_frequency'], showchannels=(0,21))
smallerplot.show()

smallerplot.ch_start

smallerplot.ch_stop

import bokeh

bokeh.__version__



