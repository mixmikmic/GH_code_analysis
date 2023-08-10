# generic imports
import os
import numpy as np
import matplotlib
from matplotlib import style
style.use('ggplot')
matplotlib.use('nbagg')
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')

# these environment variables are required for Pandeia to work properly.  edit them to reflect where
# you have them installed locally.
os.environ['pandeia_refdata'] = "/Users/pickering/STScI/JWST_workshop/JWSTUserTraining2016/pandeia_data"
os.environ['PYSYN_CDBS'] = "/Users/pickering/STScI/JWST_workshop/JWSTUserTraining2016/cdbs.23.1.rc3"

# pandeia imports

# utility functions for building calculations and sources.
from pandeia.engine.calc_utils import build_default_calc, build_default_source

# utility functions for I/O between python structures and JSON files. these functions wrap python's core JSON
# functionality with the logic required to serialize NumPy data properly so that calculation outputs, for example,
# can be written to disk and saved.
from pandeia.engine.io_utils import read_json, write_json

# this is the core function that takes the calculation input, peforms the ETC calculation, and returns the results.
from pandeia.engine.perform_calculation import perform_calculation

# convenience functions to plot pandeia results 
def twod_plot(results, kind=None):
    if kind is None:
        print("Valid kinds of 2D plots are %s" % str(list(results['2d'].keys())))
        return None
    else:
        if kind not in results['2d']:
            print("Invalid kind of 2D plot: %s" % kind)
            print("Valid kinds of 2D plots are %s" % str(list(results['2d'].keys())))
            return None
        t = results['transform']
        if results['information']['calc_type'] == 'image':
            xmin = t['x_min'] 
            xmax = t['x_max']
            aspect = 1.0
        elif results['information']['calc_type'] == 'multiorder':
            xmin = t['x_min'] 
            xmax = t['x_max']
            aspect = 0.5
        elif results['information']['calc_type'] == 'slitless':
            mid = t['wave_det_size']/2.0
            xmin = -t['x_step'] * mid
            xmax = t['x_step'] * mid
            aspect = 0.75
        else:
            xmin = t['wave_det_min']
            xmax = t['wave_det_max']
            aspect = 0.75
        ymin = t['y_min']
        ymax = t['y_max']
        extent = [xmin, xmax, ymin, ymax]
        implot = plt.imshow(results['2d'][kind], interpolation='nearest', extent=extent, aspect=aspect*(xmax-xmin)/(ymax-ymin))
        cb = plt.colorbar(orientation='horizontal')
        plt.show()
def oned_plot(results, kind=None):
    if kind is None:
        print("Valid kinds of 1D plots are %s" % str(list(results['1d'].keys())))
        return None
    else:
        if kind not in results['1d']:
            print("Invalid kind of 1D plot: %s" % kind)
            print("Valid kinds of 1D plots are %s" % str(list(results['1d'].keys())))
            return None
        plt.plot(results['1d'][kind][0], results['1d'][kind][1])
        plt.show()

# make a default NIRCam LW imaging calculation and adjust the brightness of the source
c = build_default_calc(telescope='jwst', instrument='nircam', mode='lw_imaging')
c['scene'][0]['spectrum']['normalization']['norm_flux'] = 0.001  # make source 1 uJy

# run the calculation
r = perform_calculation(c)

# look at the scalar outputs
r['scalar']

# check to see if there were any warnings
r['warnings']

# have a look at the 2D S/N map
twod_plot(r, kind='snr')

# have a look at the focal plane count rate in e-/sec/um
oned_plot(r, kind='fp')

# do a loop through some filters and print how the extracted flux changes
filters = ['f250m', 'f277w', 'f323n']
for f in filters:
    c['configuration']['instrument']['filter'] = f
    r = perform_calculation(c)
    print(r['scalar']['extracted_flux'])

# write the results to disk using write_json(). the NumPy data gets converted to JSON arrays so this isn't the
# the most efficient method, but it's simple and largely transparent.
write_json(r, "blah.json")

# read the data that was written and plot one of the results to show how JSON can be used to store calculation results
# for later analysis
data = read_json("blah.json")
twod_plot(data, kind='snr')

# it's also possible to generate calculation outputs that produce FITS objects for the 2D and 3D data products
raw_r = perform_calculation(c, dict_report=False)  # outputs raw Report instance
r = raw_r.as_dict()  # produces dict() format as described in the engine output API document
fits_r = raw_r.as_fits()  # like the dict format, but with the 2D and 3D NumPy objects converted to FITS objects
fits_r['2d']['snr'].writeto("snr.fits")

get_ipython().system('ls')

