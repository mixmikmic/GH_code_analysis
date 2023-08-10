get_ipython().magic('pylab inline')
# Matplotlib default settings
rcdef = plt.rcParams.copy()
pylab.rcParams['figure.figsize'] = 12, 10
pylab.rcParams['xtick.major.size'] = 8.0
pylab.rcParams['xtick.major.width'] = 1.5
pylab.rcParams['xtick.minor.size'] = 4.0
pylab.rcParams['xtick.minor.width'] = 1.5
pylab.rcParams['ytick.major.size'] = 8.0
pylab.rcParams['ytick.major.width'] = 1.5
pylab.rcParams['ytick.minor.size'] = 4.0
pylab.rcParams['ytick.minor.width'] = 1.5
rc('axes', linewidth=2)

import numpy as np
from astropy.io import fits 
from __future__ import division 
from astropy import units as u

import cubehelix  # Cubehelix color scheme
import copy

import os.path
from pyraf import iraf

# GALFIT output file 
galfitFile1 = 'red_21572_Icut_1comp.fits'
galfitFile2 = 'red_21572_Icut_2comp.fits'

galOutData1 = fits.open(galfitFile1)
galOutData2 = fits.open(galfitFile2)

# Basic structure 
galOutData1.info()

# Read in the Multi-Extension Data 
galOri1 = galOutData1[1].data
galMod1 = galOutData1[2].data
galRes1 = galOutData1[3].data

galOri2 = galOutData2[1].data
galMod2 = galOutData2[2].data
galRes2 = galOutData2[3].data

# Header information for the model image
headMod1 = galOutData1[2].header
headMod2 = galOutData2[2].header

# Show an example header from Galfit model image
headMod2

aa = headMod2['1_MAG']

bb = float((aa.split())[0])





