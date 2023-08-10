import iris

iris.FUTURE.netcdf_promote = True

url = 'http://icdc.cen.uni-hamburg.de/thredds/dodsC/ftpthredds/woce/wghc_params.nc'

cubes = iris.load_raw(url)
print(cubes)

tpoten = cubes.extract_strict('Tpoten')
sig0 = cubes.extract_strict('Sig0')

tpoten = tpoten.extract(
    iris.Constraint(longitude=lambda cell: 334 <= cell < 334.5)
)

sig0 = sig0.extract(
    iris.Constraint(longitude=lambda cell: 334 <= cell < 334.5)
)

y = tpoten.coord(axis='Y').points
z = tpoten.coord(axis='Z').points

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from oceans.colormaps import cm


cmap = cm.odv

fig, ax = plt.subplots(figsize=(19, 7.25))
cs = ax.pcolormesh(y, z, tpoten.data, cmap=cmap)
ax.set_ylim(0, 2000)
ax.set_xlim(-60, 60)
ax.invert_yaxis()
ax.set_ylabel('depth (m)')
ax.set_xlabel('latitude (degrees)')

fig.colorbar(cs, orientation='vertical', shrink=0.95, fraction=0.15, extend='both')

levels = [24.5, 25, 26, 26.5, 27, 27.5]
clabel = ax.contour(y, z, sig0.data, colors='w', levels=levels)
t = ax.clabel(clabel, fmt=r'%.1f')

import palettable


cmap = palettable.cmocean.sequential.Thermal_9.mpl_colormap

fig, ax = plt.subplots(figsize=(19, 7.25))
cs = ax.pcolormesh(y, z, tpoten.data, cmap=cmap)
ax.set_ylim(0, 2000)
ax.set_xlim(-60, 60)
ax.invert_yaxis()
ax.set_ylabel('depth (m)')
ax.set_xlabel('latitude (degrees)')

fig.colorbar(cs, orientation='vertical', shrink=0.95, fraction=0.15, extend='both')

levels = [24.5, 25, 26, 26.5, 27, 27.5]
clabel = ax.contour(y, z, sig0.data, colors='w', levels=levels)
t = ax.clabel(clabel, fmt=r'%.1f')

cmap = palettable.mycarta.Cube1_20.mpl_colormap

fig, ax = plt.subplots(figsize=(19, 7.25))
cs = ax.pcolormesh(y, z, tpoten.data, cmap=cmap)
ax.set_ylim(0, 2000)
ax.set_xlim(-60, 60)
ax.invert_yaxis()
ax.set_ylabel('depth (m)')
ax.set_xlabel('latitude (degrees)')

fig.colorbar(cs, orientation='vertical', shrink=0.95, fraction=0.15, extend='both')

levels = [24.5, 25, 26, 26.5, 27, 27.5]
clabel = ax.contour(y, z, sig0.data, colors='w', levels=levels)
t = ax.clabel(clabel, fmt=r'%.1f')

