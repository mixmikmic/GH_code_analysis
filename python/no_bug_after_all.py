

get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import datetime as dt
import netCDF4
import numpy as np

import cartopy.crs as ccrs
from cartopy.io.img_tiles import MapQuestOpenAerial, MapQuestOSM, OSM

import iris
import pyugrid

iris.FUTURE.netcdf_promote = True

# UMASSD/SMAST FVCOM Simulations in support of Water Quality for MWRA
url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/mwra/fvcom'
var = 'sea_water_salinity'    # use standard_name if it exists
levs = np.arange(28.,33.5,.1)   # contour levels for plotting
bbox = [-71.5, -70, 41.5, 43]
klev = 0   # level 0 is top in FVCOM

# time relative to now
start = dt.datetime.utcnow() + dt.timedelta(hours=6)
# or specific time (UTC)
start = dt.datetime(1998,3,2,15,0,0)

cube = iris.load_cube(url,var)    # Iris uses the standard_name or long_name to access variables

ug = pyugrid.UGrid.from_ncfile(url)

cube.mesh = ug
cube.mesh_dimension = 1  # (0:time,1:node)
lon = cube.mesh.nodes[:,0]
lat = cube.mesh.nodes[:,1]
nv = cube.mesh.faces
triang = tri.Triangulation(lon,lat,triangles=nv)

nc = netCDF4.Dataset(url).variables
time_var = nc['time']
itime = netCDF4.date2index(start,time_var,select='nearest')

zcube = cube[itime, klev, :]

geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))

fig = plt.figure(figsize=(8,8))
tiler = MapQuestOpenAerial()
ax = plt.axes(projection=tiler.crs)

#ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(bbox,geodetic)
ax.add_image(tiler, 8)

#ax.coastlines()
plt.tricontourf(triang, zcube.data, levels=levs, transform=geodetic)
plt.colorbar()
plt.tricontour(triang, zcube.data, colors='k',levels=levs, transform=geodetic)
tvar = cube.coord('time')
tstr = tvar.units.num2date(tvar.points[itime])
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
plt.title('%s: %s: %s' % (zcube.attributes['title'],var,tstr));



