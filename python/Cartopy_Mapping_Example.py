get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def make_map(projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(9, 13),
                           subplot_kw=dict(projection=projection))
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return fig, ax

extent = [-180, -135, 45, 80]

from cartopy.feature import NaturalEarthFeature

coast = NaturalEarthFeature(category='physical', scale='10m',
                            facecolor='none', name='coastline')

fig, ax = make_map(projection=ccrs.PlateCarree())

#ax.set_extent(extent)

feature = ax.add_feature(coast, edgecolor='black')

fig,ax = make_map(projection=ccrs.PlateCarree())

ax.coastlines(resolution='50m')

import cartopy.feature as cfeature

fig, ax = make_map(projection=ccrs.PlateCarree())
ax.set_extent(extent)

shp = cfeature.shapereader.Reader('data/simple_coastlines/Alaska.shp')
for record, geometry in zip(shp.records(), shp.geometries()):
    ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='lightgray',
                      edgecolor='black')

fig, ax = make_map(projection=ccrs.PlateCarree())
ax.set_extent(extent)

shp = cfeature.shapereader.Reader('data/coastlines/Alaska.shp')
for record, geometry in zip(shp.records(), shp.geometries()):
    ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='lightgray',
                      edgecolor='black')

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(extent)

shp = cfeature.shapereader.Reader('data/coastlines/Alaska.shp')
for record, geometry in zip(shp.records(), shp.geometries()):
    ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='white',
                      edgecolor='black')



