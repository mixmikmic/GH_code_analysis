from netCDF4 import Dataset
import numpy as np
import pysgrid

url = ('http://geoport-dev.whoi.edu/thredds/dodsC/clay/usgs/users/zdefne/run076/his/00_dir_roms_display.ncml')

nc = Dataset(url)
sgrid = pysgrid.load_grid(nc)
sgrid  # We need a better __repr__ and __str__ !!!

get_ipython().run_cell_magic('time', '', 'lons, lats = np.mgrid[-74.38:-74.26:600j, 39.45:39.56:600j]\n\npoints = np.stack((lons, lats), axis = -1)\n                \nprint points.shape')

get_ipython().run_cell_magic('time', '', "time_idx = 0\ndepth_idx = -1\n\ninterp_u = sgrid.interpolate_var_to_points(points, sgrid.u, slices=[time_idx, depth_idx])\ninterp_v = sgrid.interpolate_var_to_points(points, sgrid.v, slices=[time_idx, depth_idx])\n\n#other function signatures:\ninterp_u2 = sgrid.interpolate_var_to_points(points, sgrid.u[time_idx, depth_idx])\ninterp_u3 = sgrid.interpolate_var_to_points(points, nc.variables['u'], slices=[time_idx, depth_idx])\nprint np.all(interp_u == interp_u2)\nprint np.all(interp_u == interp_u3)")

#rotation is still ugly...
from pysgrid.processing_2d import rotate_vectors, vector_sum
from pysgrid.processing_2d import vector_sum

ind = sgrid.locate_faces(points)
ang_ind = ind + [1, 1]
angles = sgrid.angles[:][ang_ind[:, 0], ang_ind[:, 1]]
u_rot, v_rot = rotate_vectors(interp_u, interp_v, angles)
u_rot = u_rot.reshape(600,-1) # reshape for pcolormesh
v_rot = v_rot.reshape(600,-1)

uv_vector_sum = vector_sum(u_rot, v_rot)

get_ipython().magic('matplotlib inline')

#Credit to Filipe for the plotting cells below

import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def make_map(projection=ccrs.PlateCarree(), figsize=(12, 12)):
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw=dict(projection=projection))
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return fig, ax

mscale = 1
vscale = 15
scale = 0.04
lon_data = lons
lat_data = lats

fig, ax = make_map()

kw = dict(scale=1.0/scale, pivot='middle', width=0.003, color='black')
q = plt.quiver(lon_data[::vscale, ::vscale], lat_data[::vscale, ::vscale],
               u_rot[::vscale, ::vscale], v_rot[::vscale, ::vscale], zorder=2, **kw)

cs = plt.pcolormesh(lon_data[::mscale, ::mscale],
                    lat_data[::mscale, ::mscale],
                    uv_vector_sum[::mscale, ::mscale], zorder=1, cmap=plt.cm.rainbow)
ax.coastlines('10m');
plt.title('SGrid interpolation')
plt.show()




