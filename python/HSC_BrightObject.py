import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from astropy.table import Table
from dataCleanUp import dataCleanUp
import flatmaps as fm
from flatMask import createMask
from createMaps import createMeanStdMaps,createCountsMap

HSCdatapath= '/global/cscratch1/sd/damonge/HSC/'
HSCFiles= os.listdir(HSCdatapath)
HSCFiles= ['HSC_WIDE_GAMA15H_forced.fits', 'HSC_WIDE_GAMA15H_random.fits'] # now random have is primary

HSCFiles= [HSCdatapath+f for f in HSCFiles]
HSCFiles

HSCdata= {}
for filename in HSCFiles:
    key= filename.split('WIDE_')[1].split('.fits')[0]
    dat = Table.read(filename, format='fits')
    HSCdata[key] = dat.to_pandas()
    
HSCFieldTag= key.split('_')[0]  # just the field tag.

# clean up
for key in HSCdata:
    print key
    HSCdata[key]= dataCleanUp(HSCdata[key])

resolution=0.006
fsg0= fm.FlatMapInfo([212.5,222.],[-2.,2.], dx=resolution,dy=resolution)
mask,fsg=createMask(HSCdata['GAMA15H_random']['ra'],HSCdata['GAMA15H_random']['dec'],
                    [HSCdata['GAMA15H_random']['iflags_pixel_bright_object_center'],
                     HSCdata['GAMA15H_random']['iflags_pixel_bright_object_any']],fsg0,0.003)
fsg.view_map(mask)
print fsg.get_dims(),fsg0.get_dims()

tab=HSCdata['GAMA15H_forced']
mpd=createCountsMap(tab['ra'][tab['icmodel_mag']-tab['a_i']<25.],tab['dec'][tab['icmodel_mag']-tab['a_i']<25.],fsg)
fsg.view_map(mpd)
fsg.view_map(mpd*mask)



