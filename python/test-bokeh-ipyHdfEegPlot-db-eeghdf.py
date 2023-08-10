# %load explore-eeghdf-files-basics.py
# Here is an example of how to do basic exploration of what is in the eeghdf file. I show how to discover the fields in the file and to plot them.
# 
# I have copied the stacklineplot from my python-edf/examples code to help with display. Maybe I will put this as a helper or put it out as a utility package to make it easier to install.

from __future__ import print_function, division, unicode_literals
import os.path
import pandas as pd
import numpy as np
import eeghdf
from pprint import pprint

import ipywidgets
from IPython.display import display

# import eegvis.stacklineplot
import eegvis.montageview as montageview
import eegvis.stackplot_bokeh as sbokplot
from bokeh.io import output_notebook, push_notebook
import bokeh.plotting as bplt
from bokeh.plotting import show
output_notebook()

ARCHIVEDIR = r'../../eeg-hdfstorage/data'
EEGFILE = os.path.join(ARCHIVEDIR, 'spasms.eeghdf')

#hdf = h5py.File('./archive/YA2741BS_1-1+.eeghdf') # 5mo boy 
print(EEGFILE)
hf = eeghdf.Eeghdf_ver2(EEGFILE)# absence 10yo

hf.electrode_labels

hf.shortcut_elabels



tmp = sbokplot.IpyHdfEegPlot(hdf,page_width_seconds=15, showchannels=(0,19)) # doing this just to make the labels

db = montageview.DoubleBananaMontageView(rec_labels=hf.shortcut_elabels)
inr = sbokplot.IpyHdfEegPlot(hf.hdf,page_width_seconds=15, montage=db)

# inr.all_montages.append(db)
#inr.current_montage = db

inr.show()

# you can manipulate the EEG plot directly
inr.loc_sec = 600
inr.update()

inr.current_montage.name

hf.hdf.filename

import bokeh
bokeh.__version__



