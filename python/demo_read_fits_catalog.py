get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from __future__ import (print_function, division, absolute_import)

import os
import glob
import copy
import sys
import warnings

import numpy as np

# Matplotlib related
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

from astropy.table import     Table,     Column,     vstack,     unique
    
from astropy.utils.metadata import MergeConflictWarning
warnings.filterwarnings('ignore', category=MergeConflictWarning, 
                        append=True)
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.console import ProgressBar

table_g09 = Table.read('../hsc/hsc_s16a_fastlane_g09_short.fits', format='fits')

table_g09

print("# There are %d objects in Wide-GAMA09 region" % len(table_g09))

table_g09.colnames

fig = plt.figure(figsize=(8, 7))
ax1 = fig.add_subplot(111)

ax1.scatter(table_g09['ra'], 
            table_g09['dec'], s=15, alpha=0.5,
            label=r'$\mathrm{HSC}$')

ax1.set_xlim(128.5, 142.5)
ax1.set_ylim(-1.8, 4.8)

ax1.legend(fontsize=20)

ax1.set_xlabel('$\mathrm{RA\ (deg)}$', size=30)
ax1.set_ylabel('$\mathrm{Dec\ (deg)}$', size=30)

fig = plt.figure(figsize=(8, 7))
ax1 = fig.add_subplot(111)

ax1.hist(table_g09['z_best'], alpha=0.9, 
         label=r'$\mathrm{z\_best}$')

ax1.set_xlim(0.25, 0.55)

ax1.legend(fontsize=20)

ax1.set_xlabel(r'$\mathrm{Redshift}$', size=30)
ax1.set_ylabel(r'$\mathrm{Number}$', size=30)

rcmodel_mag_real = table_g09['rcmodel_mag'] - table_g09['a_r']

table_g09.add_column(Column(rcmodel_mag_real, name='rcmodel_mag_real'))

plt.scatter(table_g09['rcmodel_mag'], table_g09['rcmodel_mag_real'])

table_g09.write('../hsc/temp.ftis', format='fits', 
                overwrite=True)

