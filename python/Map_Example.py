import sys
from os.path import *
import os

from astropy.io import fits
import astropy.units as u

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from pylab import figure, cm
get_ipython().magic('matplotlib inline')

import numpy as np

import nustar_pysolar as nustar

#infile = '../data/Sol_16208/20201001001/event_cl/nu20201001001A06_chu12_N_cl_sunpos.evt'
infile = '../data/Sol_16208/20201001001/event_cl/nu20201001001B06_chu3_N_cl_sunpos.evt'

hdulist = fits.open(infile)
evtdata=hdulist[1].data
hdr = hdulist[1].header
print("Loaded: ", len(evtdata['X']), " counts.")
print("Effective exposure: ", hdr['EXPOSURE'], ' seconds')
hdulist.close()

cleanevt = nustar.filter.event_filter(evtdata)

nustar_map = nustar.map.make_sunpy(cleanevt, hdr)

# Make the new filename:
(sfile, ext)=splitext(infile)
outfile=sfile+'_map.fits'

# Remove output file if necessary
if isfile(outfile):
  print(outfile, 'exists! Removing old version...')
  os.remove(outfile)
nustar_map.save(outfile, filetype='auto')


rangex = u.Quantity([500*u.arcsec, 1500 * u.arcsec])
rangey = u.Quantity([-500 * u.arcsec, 500 * u.arcsec])

nustar_map.plot_settings['norm'] = colors.LogNorm(1.0, nustar_map.max())
nustar_map.plot_settings['cmap'] = cm.get_cmap('BrBG')

nustar_submap = nustar_map.submap(rangex, rangey)


plt.subplots(figsize=(10, 10))
nustar_submap.plot()
plt.colorbar()
nustar_submap.draw_limb(color='r')





