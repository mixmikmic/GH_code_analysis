get_ipython().magic('matplotlib inline')
from __future__ import (absolute_import, division, print_function)
import numpy as np
import matplotlib.tri as tri
import datetime as dt
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import iris
iris.FUTURE.netcdf_promote = True
import pyugrid

# specify UGRID compliant OPeNDAP Data URL

#ADCIRC
#url = 'http://comt.sura.org/thredds/dodsC/data/comt_1_archive/inundation_tropical/UND_ADCIRC/Hurricane_Rita_2D_final_run_without_waves'

#ESTOFS
url = 'http://coastalmodeldev.data.noaa.gov/thredds/dodsC/aggregAtlanticESTOFS'

#FVCOM
#url = 'http://comt.sura.org/thredds/dodsC/data/comt_1_archive/inundation_tropical/USF_FVCOM/Hurricane_Rita_2D_final_run_without_waves'

#SELFE
#url = 'http://comt.sura.org/thredds/dodsC/data/comt_1_archive/inundation_tropical/VIMS_SELFE/Hurricane_Rita_2D_final_run_without_waves'

# set parameters
bbox = [-95, -85, 27, 32]                  # set the bounding box [lon_min, lon_max, lat_min, lat_max]
var = 'sea_surface_height_above_geoid'     # standard_name (or long_name, if no standard_name)
levs = np.arange(-0.5,.55,.05)                # set the contour levels
#start = dt.datetime(2005, 9, 24, 5, 0, 0)  # time in UTC
start = dt.datetime.utcnow() + dt.timedelta(hours=6)

cube = iris.load_cube(url,var)

print(cube)

ug = pyugrid.UGrid.from_ncfile(url)

# What's in there?
print('There are {0} nodes'.format(ug.nodes.shape[0]))
print('There are {0} faces'.format(ug.faces.shape[0]))

cube.mesh = ug

lon = cube.mesh.nodes[:,0]
lat = cube.mesh.nodes[:,1]
nv = cube.mesh.faces

triang = tri.Triangulation(lon,lat,triangles=nv)

tvar = cube.coord('time')
itime = tvar.nearest_neighbour_index(tvar.units.date2num(start))

zcube = cube[itime]

plt.figure(figsize=(16,6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(bbox)
ax.coastlines(resolution='10m')
plt.tricontourf(triang, zcube.data, levels=levs)
plt.colorbar(fraction=0.046, pad=0.04)
plt.tricontour(triang, zcube.data, colors='k',levels=levs)
tstr = tvar.units.num2date(tvar.points[itime])
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
plt.title('{}: {}: {}'.format(var,tstr,zcube.attributes['title']));



