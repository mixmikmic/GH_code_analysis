url = ('https://data.ioos.us/thredds/dodsC/deployments/rutgers/'
       'ru29-20150623T1046/ru29-20150623T1046.nc3.nc')

import iris

iris.FUTURE.netcdf_promote = True

glider = iris.load(url)

print(glider)

temp = glider.extract_strict('sea_water_temperature')
salt = glider.extract_strict('sea_water_salinity')
dens = glider.extract_strict('sea_water_density')

print(temp)

import numpy.ma as ma

T = temp.data.squeeze()
S = salt.data.squeeze()
D = dens.data.squeeze()

x = temp.coord(axis='X').points.squeeze()
y = temp.coord(axis='Y').points.squeeze()
z = temp.coord(axis='Z')
t = temp.coord(axis='T')

vmin, vmax = z.attributes['actual_range']

z = ma.masked_outside(z.points.squeeze(), vmin, vmax)
t = t.units.num2date(t.points.squeeze())

location = y.mean(), x.mean()  # Track center.
locations = list(zip(y, x))  # Track points.

import folium

tiles = ('http://services.arcgisonline.com/arcgis/rest/services/'
         'World_Topo_Map/MapServer/MapServer/tile/{z}/{y}/{x}')

m = folium.Map(
    location,
    tiles=tiles,
    attr='ESRI',
    zoom_start=4
)

folium.CircleMarker(locations[0], fill_color='green', radius=10).add_to(m)
folium.CircleMarker(locations[-1], fill_color='red', radius=10).add_to(m)

line = folium.PolyLine(
    locations=locations,
    color='orange',
    weight=8,
    opacity=0.6,
    popup='Slocum Glider ru29 Deployed on 2015-06-23'
).add_to(m)

m

import numpy as np

# Find the deepest profile.
idx = np.nonzero(~T[:, -1].mask)[0][0]

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

ncols = 3
fig, (ax0, ax1, ax2) = plt.subplots(
    sharey=True, sharex=False, ncols=ncols, figsize=(3.25*ncols, 5)
)

kw = dict(linewidth=2, color='cornflowerblue', marker='.')
ax0.plot(T[idx], z[idx], **kw)
ax1.plot(S[idx], z[idx], **kw)
ax2.plot(D[idx]-1000, z[idx], **kw)


def spines(ax):
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')


[spines(ax) for ax in (ax0, ax1, ax2)]

ax0.set_ylabel('Depth (m)')
ax0.set_xlabel('Temperature ({})'.format(temp.units))
ax0.xaxis.set_label_position('top')

ax1.set_xlabel('Salinity ({})'.format(salt.units))
ax1.xaxis.set_label_position('top')

ax2.set_xlabel('Density ({})'.format(dens.units))
ax2.xaxis.set_label_position('top')

ax0.invert_yaxis()

import numpy as np
import seawater as sw
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def distance(x, y, units='km'):
    dist, pha = sw.dist(x, y, units=units)
    return np.r_[0, np.cumsum(dist)]


def plot_glider(x, y, z, t, data, cmap=plt.cm.viridis,
                figsize=(9, 3.75), track_inset=False):

    fig, ax = plt.subplots(figsize=figsize)
    dist = distance(x, y, units='km')
    z = np.abs(z)
    dist, z = np.broadcast_arrays(dist[..., np.newaxis], z)
    cs = ax.pcolor(dist, z, data, cmap=cmap, snap=True)
    kw = dict(orientation='vertical', extend='both', shrink=0.65)
    cbar = fig.colorbar(cs, **kw)

    if track_inset:
        axin = inset_axes(ax, width="25%", height="30%", loc=4)
        axin.plot(x, y, 'k.')
        start, end = (x[0], y[0]), (x[-1], y[-1])
        kw = dict(marker='o', linestyle='none')
        axin.plot(*start, color='g', **kw)
        axin.plot(*end, color='r', **kw)
        axin.axis('off')

    ax.invert_yaxis()
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Depth (m)')
    return fig, ax, cbar

import cmocean

fig, ax, cbar = plot_glider(x, y, z, t, S,
                            cmap=cmocean.cm.haline, track_inset=False)
cbar.ax.set_xlabel('(g kg$^{-1}$)')
cbar.ax.xaxis.set_label_position('top')
ax.set_title('Salinity')

fig, ax, cbar = plot_glider(x, y, z, t, T,
                            cmap=cmocean.cm.thermal, track_inset=False)
cbar.ax.set_xlabel(r'($^\circ$C)')
cbar.ax.xaxis.set_label_position('top')
ax.set_title('Temperature')

fig, ax, cbar = plot_glider(x, y, z, t, D-1000,
                            cmap=cmocean.cm.dense, track_inset=False)
cbar.ax.set_xlabel(r'(kg m$^{-3}$C)')
cbar.ax.xaxis.set_label_position('top')
ax.set_title('Density')

print('Data collected from {} to {}'.format(t[0], t[-1]))

