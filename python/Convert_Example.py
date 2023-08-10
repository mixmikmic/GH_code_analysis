import sys
from os.path import *
import os

# For loading the NuSTAR data
from astropy.io import fits

# Load the NuSTAR python libraries
from nustar_pysolar import convert, utils

#infile = '/Users/bwgref/science/solar/july_2016/data/20201002001/event_cl/nu20201002001B06_chu3_N_cl.evt'

infile = '/Users/bwgref/science/solar/data/Sol_16208/20201002001/event_cl/nu20201002001B06_chu3_N_cl.evt'

hdulist = fits.open(infile)
evtdata = hdulist[1].data 
hdr = hdulist[1].header
hdulist.close()

reload(convert)
(newdata, newhdr) = convert.to_solar(evtdata, hdr)

# # Make the new filename:
(sfile, ext)=splitext(infile)

outfile=sfile+'_sunpos.evt'

# Remove output file if necessary
if isfile(outfile):
    print(outfile, 'exists! Removing old version...')
    os.remove(outfile)

fits.writeto(outfile, newdata, newhdr)

convert.convert_file(infile)



