get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import xarray as xr
import numpy as np
import marc_analysis as ma

from ipywidgets import interact

test_ds = xr.open_dataset("medium_sample.nc",)
test_ds = test_ds.isel(time=0, lat=slice(-60, 60))

print(test_ds)

field = 'T'
test_data = test_ds[field]

# Compute the 3D pressure field given the hybrid coordinate system
pres_levs = ma.hybrid_to_pressure(test_ds)

# Select pressure levels for interpolation
new_pres_levs = 100.*np.array([300., 500., 750., 850., 960.])
# new_pres_levs = 100.*np.array(np.logspace(2, 3, 50))[-5:]

# Compute interpolated data
new_data_np = ma.interp_to_pres_levels(test_data, pres_levs, new_pres_levs,
                                      'numpy')
# new_data_sp = ma.interp_to_pres_levels(test_data, pres_levs, new_pres_levs,
#                                       'scipy')

# As a safety check, copy attributes into interpolated data
nd = ma.copy_attrs(test_data, new_data_np)
new_data_np.lev.attrs

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, LogLocator
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import seaborn as sns
sns.set(style='ticks', context='talk')

@interact(ilat=[0, len(test_ds.lat)-1], ilon=[0, len(test_ds.lon)-1])
def plot_column(ilat, ilon):
    
    interp_np_column = new_data_np.isel(lat=ilat, lon=ilon)
    orig_column = test_data.isel(lat=ilat, lon=ilon)
    pres_column = pres_levs.isel(lat=ilat, lon=ilon)

    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot(111)

    ax.plot(orig_column, pres_column/100., color='k', 
            ms=3, marker='o', lw=1)
    ax.plot(interp_np_column, new_pres_levs/100.,
            linestyle='none', marker='d', color='r')
        
    ax.semilogy()
    ax.set_ylim(1035, 50)
    for p in new_pres_levs/100.:
        ax.hlines(p, *ax.get_xlim(), linestyle='dashed', lw=1)
    
    
    ax.set_xlabel("{} ({})".format(test_data.long_name, 
                                   test_data.units))
    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%d")
    )
    ax.yaxis.set_major_locator(MaxNLocator(10))
    ax.set_ylabel("Pressure (hPa)")
    

field = 'Q'
test_data = test_ds[field]

heights = test_ds.Z3*1e-3
new_height_levs = np.arange(1, 10.)
print(new_height_levs)

# Compute interpolated data
new_data_np = ma.interp_by_field(test_data, heights, new_height_levs)

# As a safety check, copy attributes into interpolated data
nd = ma.copy_attrs(test_data, new_data_np)
new_data_np.attrs.update(dict(
    long_name='altitude', units='km', positive='up',
    standard_name='altitude'
))

@interact(ilat=[0, len(test_ds.lat)-1], ilon=[0, len(test_ds.lon)-1])
def plot_column(ilat, ilon):
    
    interp_np_column = new_data_np.isel(lat=ilat, lon=ilon)
    orig_column = test_data.isel(lat=ilat, lon=ilon)
    heights_column = heights.isel(lat=ilat, lon=ilon)

    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot(111)

    ax.plot(orig_column, heights_column, color='k', 
            ms=3, marker='o', lw=1)
    ax.plot(interp_np_column, new_height_levs,
            linestyle='none', marker='d', color='r')

    for p in new_height_levs:
        ax.hlines(p, *ax.get_xlim(), linestyle='dashed', lw=1)
    
    ax.set_ylim(0, 17)
    ax.set_xlabel("{} ({})".format(test_data.long_name, 
                                   test_data.units))
    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%d")
    )
#     ax.set_ylabel("{} ({})".format(new_data_np.lev.long_name, 
#                                    new_data_np.lev.units))
    

