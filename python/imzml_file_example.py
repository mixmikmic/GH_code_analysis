# Basic packages
import numpy as np
import os
import sys
import re
import time

# Plotting
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Terminal-specific path definitions for finding data
def isneeded(x):
    if x not in sys.path:
        sys.path.append(x)

isneeded('/Users/curt/openMSI_SVN/openmsi-tk/')
isneeded('/Users/curt/openMSI_localdata/')

# Use openmsi-tk / bastet code for imzML parsing
from omsi.dataformat.imzml_file import *

# Function to find most intense pixel and most intense m/z, 
#  will only plot if matplotlib is already imported
def plotMostIntense(cube, mzvec):
    """
    cube is a 3D numpy array with dimensions x, y, and m/z
    mzvec is a vector of same length as cube.shape[2]
    """
    mxx, mxy = np.unravel_index(cube[:, :, :].sum(axis=2).argmax(), dims=cube.shape[:-1])
    mxmz = cube[:, :, :].sum(axis=(0, 1)).argmax()
    f, ax = plt.subplots(1, 2)
    ax[0].plot(mzvec, cube[mxx, mxy, :])
    ax[0].set_title('spectrum for pixel ' + str(mxx) + ', ' + str(mxy))
    
    ax[1].imshow(cube[:, :, mxmz], interpolation='none', cmap=cm.Greys_r)
    ax[1].set_title('image for m/z ' + str(mzvec[mxmz]))
    return mxx, mxy, mxmz

# Reading tiny example file
fname = '/Users/curt/openMSI_localdata/imzml/example_files/Example_Continuous.imzML'
start = time.time()
ec = imzml_file(basename=fname)
stop = time.time()
print 'Parsing file %s took %s seconds.' % (fname, stop-start)
plotMostIntense(ec.data, ec.mz_all)

# Reading "real" example file
fname = '/Users/curt/openMSI_localdata/imzml/s042_continuous/S042_Continuous.imzML'
start = time.time()
s42c = imzml_file(basename=fname)
stop = time.time()
print 'Parsing file %s took %s seconds.' % (fname, stop-start)
plotMostIntense(s42c.data, s42c.mz_all)

# Reading "tiny" example processed-mode file: this works
fname = '/Users/curt/openMSI_localdata/imzml/example_files/Example_Processed.imzML'
start = time.time()
ep = imzml_file(basename=fname)
stop = time.time()
print 'Parsing file %s took %s seconds.' % (fname, stop-start)
plotMostIntense(ep.data, ep.mz_all)

# Reading "tiny" example processed-mode file: this does not work
fname = '/Users/curt/openMSI_localdata/imzml/s043_processed/S043_Processed.imzML'
start = time.time()
s43p = imzml_file(basename=fname)
stop = time.time()
print 'Parsing file %s took %s seconds.' % (fname, stop-start)

# plotMostIntense(s43p.data, s43p.mz_all)

