import iris

iris.FUTURE.netcdf_promote = True

url = ('http://omgsrv1.meas.ncsu.edu:8080/thredds/dodsC/fmrc/'
       'sabgom/SABGOM_Forecast_Model_Run_Collection_best.ncd')

cubes = iris.load(url)

print(cubes)

cube = cubes.extract_strict('sea_surface_height')

print(cube)

temp = cubes.extract_strict('sea_water_potential_temperature')

# Surface at the last time step.
T = temp[-1, -1, ...]

# Random profile at the last time step.
t_profile = temp[-1, :, 160, 220]

t_profile.coords(axis='Z')

get_ipython().magic('matplotlib inline')

import numpy.ma as ma
import iris.quickplot as qplt

T.data = ma.masked_invalid(T.data)

qplt.pcolormesh(T)

l = qplt.plot(t_profile)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(3, 7))

t = t_profile.data
z = t_profile.coord('sea_surface_height_above_reference_ellipsoid').points

l, = ax.plot(t, z)

import pyugrid

url = 'http://crow.marine.usf.edu:8080/thredds/dodsC/FVCOM-Nowcast-Agg.nc'

ugrid = pyugrid.UGrid.from_ncfile(url)

ugrid.build_edges()

lon = ugrid.nodes[:, 0]
lat = ugrid.nodes[:, 1]
triangles = ugrid.faces[:]

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def make_map(projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(8, 6),
                           subplot_kw=dict(projection=projection))
    ax.coastlines(resolution='50m')
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return fig, ax


fig, ax = make_map()

kw = dict(marker='.', linestyle='-', alpha=0.25, color='darkgray')
ax.triplot(lon, lat, triangles, **kw)
ax.coastlines()
ax.set_extent([-88, -79.5, 24, 32])

import pysgrid

url = ('http://geoport.whoi.edu/thredds/dodsC/'
       'coawst_4/use/fmrc/coawst_4_use_best.ncd')

sgrid = pysgrid.load_grid(url)

sgrid.edge1_coordinates, sgrid.edge1_dimensions, sgrid.edge1_padding

u_var = sgrid.u

u_var.center_axis, u_var.node_axis

v_var = sgrid.v
v_var.center_axis, v_var.node_axis

u_var.center_slicing, v_var.center_slicing

from netCDF4 import Dataset

nc = Dataset(url)
u_velocity = nc.variables[u_var.variable]
v_velocity = nc.variables[v_var.variable]

v_idx = 0  # Bottom.
time_idx = -1  # Last time step.

u = u_velocity[time_idx, v_idx, u_var.center_slicing[-2], u_var.center_slicing[-1]]
v = v_velocity[time_idx, v_idx, v_var.center_slicing[-2], v_var.center_slicing[-1]]

# Average at the center.
from pysgrid.processing_2d import avg_to_cell_center

u = avg_to_cell_center(u, u_var.center_axis)
v = avg_to_cell_center(v, v_var.center_axis)

# **Rotate the grid.
from pysgrid.processing_2d import rotate_vectors

angles = nc.variables[sgrid.angle.variable][sgrid.angle.center_slicing]
u, v = rotate_vectors(u, v, angles)

# Compute the speed.
from pysgrid.processing_2d import vector_sum

speed = vector_sum(u, v)

lon_var_name, lat_var_name = sgrid.face_coordinates

sg_lon = getattr(sgrid, lon_var_name)
sg_lat = getattr(sgrid, lat_var_name)

lon = sgrid.center_lon[sg_lon.center_slicing]
lat = sgrid.center_lat[sg_lat.center_slicing]

def is_monotonically_increasing(arr, axis=0):
    return np.all(np.diff(arr, axis=axis) > 0)


def is_monotonically_decreasing(arr, axis=0):
    return np.all(np.diff(arr, axis=axis) < 0)


def is_monotonic(arr):
    return (is_monotonically_increasing(arr) or
            is_monotonically_decreasing(arr))


def extent_bounds(arr, bound_position=0.5, axis=0):
    if not is_monotonic(arr):
        msg = "Array {!r} must be monotonic to guess bounds".format
        raise ValueError(msg(arr))

    x = arr.copy()
    x = np.c_[x[:, 0], (bound_position * (x[:, :-1] + x[:, 1:])), x[:, -1]]
    x = np.r_[x[0, :][None, ...], (bound_position * (x[:-1, :] + x[1:, :])), x[-1, :][None, ...]]

    return x

import numpy as np

# For plotting reasons we will subsample every 10th point here
# 100 times less data!
sub = 10

lon = lon[::sub, ::sub]
lat = lat[::sub, ::sub]
u, v = u[::sub, ::sub], v[::sub, ::sub]
speed = speed[::sub, ::sub]

x = extent_bounds(lon)
y = extent_bounds(lat)

def make_map(projection=ccrs.PlateCarree(), figsize=(9, 9)):
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw=dict(projection=projection))
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return fig, ax

scale = 0.06

fig, ax = make_map()

kw = dict(scale=1.0/scale, pivot='middle', width=0.003, color='black')
q = plt.quiver(lon, lat, u, v, zorder=2, **kw)

plt.pcolormesh(x, y, speed, zorder=1, cmap=plt.cm.rainbow)

c = ax.coastlines('10m')
ax.set_extent([-73.5, -62.5, 38, 46])

