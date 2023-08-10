name = '2015-11-20-cartopy-example'
title = 'Cartopy example'
tags = 'maps, matplotlib, xarray, cartopy'
author = 'Denis Sergeev'

from nb_tools import connect_notebook_to_post
from IPython.core.display import HTML

html = connect_notebook_to_post(name, title, tags, author)

get_ipython().magic('matplotlib inline')

import warnings
warnings.filterwarnings('ignore')

from __future__ import print_function, division

import cartopy.util # Cartopy utilities submodule
import cartopy.crs as ccrs # Coordinate reference systems
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sgeom
import xarray # instead we could have used the plain netcdf4 module to read the data

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER # Fancy formatting
import matplotlib as mpl
from matplotlib.transforms import offset_copy

mpl.rcParams['mathtext.default'] = 'regular' # Makes Math text regular and not LaTeX-style

city = dict(lat=52.628333, lon=1.296667, name=u'Norwich')

da = xarray.open_dataset('../data/data.nc')

print(da)

cyclic_data, cyclic_lons = cartopy.util.add_cyclic_point(da.vo.data, coord=da.longitude.data)
lats = da.latitude.data

cyclic_data = cyclic_data*1e3

clevs = np.arange(-0.5,0.55,0.05)

print(clevs)

bbox = [-10, 10, 45, 60]

# Define figure of size (25,10) and an axis 'inside' with an equirectangular projection
fig, ax = plt.subplots(figsize=(20,8), subplot_kw=dict(projection=ccrs.PlateCarree()))
# Set the map extent using the bbox defined above
ax.set_extent(bbox)
# Draw coast line with the highest resolution available (unless you use a different source)
ax.coastlines('10m')

#
# Create a filled contour plot
#
# For the mappable array we use the first slice in time and the first level in the vertical coordinate, 
# hence: cyclic_data[0,0,...]
# 
c = ax.contourf(cyclic_lons, lats, cyclic_data[0,0,...], 
                clevs, # Contour levels defined above
                cmap=plt.cm.RdBu_r, # A standard diverging colormap, from Blue to Red
                extend='both' # To make pointy colorbar
               )
#
# Create a colorbar
#
cb = plt.colorbar(c, # connect it to the contour plot
                  ax=ax, # put it in the same axis
                  orientation='horizontal',
                  shrink=0.25, # shrink its size by 4
                  pad = 0.05 # shift it up
                 )
# Colorbar label: units of vorticity
cb.set_label('$10^{-3}$ '+da.vo.units, fontsize=18, fontweight='bold')
# Increase the font size of tick labels
cb.ax.tick_params(labelsize=15)

# Title of the plot
ax.set_title(da.vo.long_name, fontsize=24)

#
# Grid lines
#
gl = ax.gridlines(crs=ccrs.PlateCarree(), # using the same projection
                  draw_labels=True, # add labels
                  linewidth=2, color='gray', alpha=0.5, linestyle='--') # grid line specs
# Remove labels above and on the right of the map (note that Python allows the double equality)
gl.xlabels_top = gl.ylabels_right = False
# Format the labels using the formatters imported from cartopy
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

#
# Norwich
#
# Mark our location with a green asterisk
ax.plot(city['lon'], city['lat'], linestyle='None', marker='*', ms=10, mfc='g', mec='g')
# Add a label next to the marker using matplotlib.transform submodule
geodetic_transform = ccrs.Geodetic()._as_mpl_transform(ax)
text_transform = offset_copy(geodetic_transform, units='dots', x=50, y=20)
ax.text(city['lon'], city['lat'], city['name'],
         verticalalignment='center', horizontalalignment='right',
         transform=text_transform,
         bbox=dict(facecolor='g', alpha=0.5, boxstyle='round'))

#
# Additional axis
#
sub_ax = plt.axes([0.55, 0.75, 0.25, 0.25], # Add an inset axis at (0.55,0.75) with 0.2 for width and height
                  projection=ccrs.Orthographic(central_latitude=45.0) # Not a simple axis, but another cartopy geoaxes instance
                 )
# The whole globe
sub_ax.set_global()
# Paint it
sub_ax.stock_img()
# Using shapely module, create a bounding box to denote the map boundaries
extent_box = sgeom.box(bbox[0], bbox[2], bbox[1], bbox[3])
sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), color='none', edgecolor='blue', linewidth=2)

# Tighten the figure layout
fig.tight_layout()
# Print a success message
print("Here's our map!")

#fig.savefig('cartopy_rules.jpg')

HTML(html)

