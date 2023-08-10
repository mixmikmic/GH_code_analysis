get_ipython().magic('matplotlib inline')

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
import matplotlib.tri as tri

import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import pyugrid
import iris
import warnings

from ciso import zslice

#url = 'http://crow.marine.usf.edu:8080/thredds/dodsC/FVCOM-Nowcast-Agg.nc'
url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cubes = iris.load_raw(url)

var = cubes.extract_strict('sea_water_potential_temperature')[-1, ...]  # Last time step.

lon = var.coord(axis='X').points
lat = var.coord(axis='Y').points

var

# calculate the 3D z values using formula terms by specifying this derived vertical coordinate
# with a terrible name 
z3d = var.coord('sea_surface_height_above_reference_ellipsoid').points

# read the 3D chuck of data
var3d = var.data

# specify depth for fixed z slice
z0 = -25
isoslice = zslice(var3d, z3d, z0)

# For some reason I cannot tricontourf with NaNs.
isoslice = ma.masked_invalid(isoslice)
vmin, vmax = isoslice.min(), isoslice.max()
isoslice = isoslice.filled(fill_value=-999)

def make_map(projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(9, 13),
                           subplot_kw=dict(projection=projection))
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.coastlines('50m')
    return fig, ax

# use UGRID conventions to locate lon,lat and connectivity array
ugrid = pyugrid.UGrid.from_ncfile(url)

lon = ugrid.nodes[:, 0]
lat = ugrid.nodes[:, 1]
triangles = ugrid.faces[:]

triang = tri.Triangulation(lon, lat, triangles=triangles)

fig, ax = make_map()
extent = [lon.min(), lon.max(),
          lat.min(), lat.max()]
ax.set_extent(extent)

levels = np.linspace(vmin, vmax, 20)

kw = dict(cmap='jet', alpha=1.0, levels=levels)
cs = ax.tricontourf(triang, isoslice, **kw)
kw = dict(shrink=0.5, orientation='vertical')
cbar = fig.colorbar(cs, **kw)

