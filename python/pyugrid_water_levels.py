import matplotlib.tri as tri
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

import cartopy.crs as ccrs
get_ipython().magic('matplotlib inline')

import iris
import pyugrid

iris.FUTURE.netcdf_promote = True

#ADCIRC
#url =  'http://comt.sura.org/thredds/dodsC/data/comt_1_archive/inundation_tropical/UND_ADCIRC/Hurricane_Ike_3D_final_run_with_waves'
#FVCOM
#url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc'
#SELFE
url = 'http://comt.sura.org/thredds/dodsC/data/comt_1_archive/inundation_tropical/VIMS_SELFE/Hurricane_Ike_2D_final_run_with_waves'

ug = pyugrid.UGrid.from_ncfile(url)

print "There are %i nodes"%ug.nodes.shape[0]
print "There are %i faces"%ug.faces.shape[0]

cube = iris.load_cube(url,'sea_surface_height_above_geoid')

print cube

cube.mesh = ug
cube.mesh_dimension = 1  # (0:time,1:node)

lon = cube.mesh.nodes[:,0]
lat = cube.mesh.nodes[:,1]
nv = cube.mesh.faces

triang = tri.Triangulation(lon,lat,triangles=nv)

ind = -1 # last time index
zcube = cube[ind]

plt.figure(figsize=(12,12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-90, -60, 5, 50])
ax.coastlines()
levs = np.arange(-1,5,.2)
plt.tricontourf(triang, zcube.data, levels=levs)
plt.colorbar()
plt.tricontour(triang, zcube.data, colors='k',levels=levs)
tvar = cube.coord('time')
tstr = tvar.units.num2date(tvar.points[ind])
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
plt.title('%s: Elevation (m): %s' % (zcube.attributes['title'],tstr));



