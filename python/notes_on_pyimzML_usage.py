# Basic packages
import numpy as np
import os
import sys
import re

#Plotting
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Modulex for reading XML, mzML, and imzML
import pymzml
from pyimzml.ImzMLParser import ImzMLParser

#A pymzml reading of imzML files

foo = pymzml.run.Reader('example_files/Example_Continuous.imzML')
for spec in foo:
    print foo.spectrum
    

# pyimzML parsing

p = ImzMLParser('example_files/Example_Continuous.imzML')

# getting coordinates
print 'p.coordinates: \n', p.coordinates, '\n \n \n'   # provides a list of x, y indices of pixels

# get a dictionary with useful metadata about the file
print 'p.imzmldict: \n', p.imzmldict, '\n \n \n'

# make two plots
f, ax = plt.subplots(1, 2, figsize=(12, 6))

# plot a spectrum
ind = 0
mz, intens = p.getspectrum(ind)
ax[0].plot(mz, intens, '-o')
ax[0].set_xlabel('m/z, Da')
ax[0].set_ylabel('intensity, AU')
ax[0].set_title('The spectrum of pixel %s at spatial location %s' % (ind, p.coordinates[ind]))

# plot an ion image
mzmax = mz[np.asarray(intens).argmax()]
im = p.getionimage(mzmax)
ax[1].imshow(im, cmap=cm.Greys_r, interpolation="none")
ax[1].set_title('The ion image of m/z %s which is index %s' % (mzmax, np.asarray(intens).argmax()))

# print index of mzmax for each pixel to show that this is cube-like data
for ind, loc in enumerate(p.coordinates):
    mz, intens = p.getspectrum(ind)
    print 'Index of m/z=%s in pixel %s for location %s is %s' % (mzmax, ind, loc, np.abs((np.asarray(mz) - mzmax)).argmin())
    
# print other elements of p

p.intensityLengths
    

# function to print an attribute of the parser
def printAttribute(obj, attr):
    print attr + ':\n', getattr(obj, attr), '\n\n\n'
    
# function to filter out private methods from dir() results
def vdir(obj):
    return [x for x in dir(obj) if not x.startswith('__')]

# print all the attributes of the parser in a legible way
for attr in vdir(p):
    printAttribute(p, attr)

