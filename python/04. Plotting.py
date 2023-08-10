import matplotlib
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from astropy.io import fits
import aplpy

gc = aplpy.FITSFigure('../data/502nmos.fits')
gc.show_grayscale()

gc = aplpy.FITSFigure('../data/502nmos.fits')
gc.show_colorscale()
gc.add_grid()
gc.tick_labels.set_font(size='small')

import pandas as pd

objects = pd.read_csv('../data/objsearch.txt',sep='|',skiprows=24)

ra, dec = np.asarray(objects['RA(deg)']), np.asarray(objects['DEC(deg)'])

m51 = aplpy.FITSFigure('../data/MESSIER_051-I-103aE-dss1.fits')
m51.show_colorscale(cmap='gist_heat')
m51.show_contour('../data/MESSIER_051-I-20cm-pkwb1984.fits', colors="white", levels=3)
m51.show_markers(ra, dec, edgecolor='green', facecolor='none', marker='*', s=100, alpha=0.8)

aplpy.make_rgb_cube(['../data/673nmos.fits','../data/656nmos.fits','../data/502nmos.fits'], '../data/nmod_cube.fits')

aplpy.make_rgb_image('../data/nmod_cube.fits','../data/output/nmod.png')

f = aplpy.FITSFigure('../data/nmod_cube_2d.fits')
f.show_rgb('../data/output/nmod.png')

aplpy.make_rgb_image('../data/nmod_cube.fits','../data/output/nmod.png', stretch_r='arcsinh', stretch_g='arcsinh', stretch_b='arcsinh')

f = aplpy.FITSFigure('../data/nmod_cube_2d.fits')
f.show_rgb('../data/output/nmod.png')

f.add_scalebar(5/60.)
f.scalebar.set_corner('top')
f.scalebar.set_length(17/600.)
f.scalebar.set_label('0.1 parsec')

f.tick_labels.set_xformat('hhmmss')
f.tick_labels.set_yformat('hhmmss')

