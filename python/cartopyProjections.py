from random import randint
def randhex():
    r = lambda: randint(0,255)
    return('#%02X%02X%02X' % (r(),r(),r()))
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.style.use('bmh')

import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
_DEFAULT_PROJECTION = ccrs.Mercator().GOOGLE
get_ipython().run_line_magic('matplotlib', 'inline')

## uncomment the line below if you want fancy high-resolution plots
## which will take a long time to load
#%config InlineBackend.figure_format = 'retina'

shapename = 'admin_1_states_provinces_lakes_shp' # specified at http://naturalearthdata.com

# Download (or find locally) the shapefile.
states_shp = shapereader.natural_earth(resolution='50m',
                                       category='cultural',
                                       name=shapename)

states_shp

fig = plt.figure(figsize=(15, 15)) 
ax = plt.axes(projection=_DEFAULT_PROJECTION)
ax.set_extent([-180, -46.5, 8, 60], ccrs.Geodetic()) # North America, in Geodetic CRS
ax.set_title("North America – Shapefile")
for state in shapereader.Reader(states_shp).geometries(): # plot each geometry with different color. 
    ax.add_geometries(state, ccrs.PlateCarree(), facecolor=randhex(), edgecolor='black')

import cartopy.feature as feature

states_feature = feature.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes_shp', 
                                             scale='50m',
                                             facecolor='none', 
                                            edgecolor='black')

fig = plt.figure(figsize=(15, 15)) 
ax = plt.axes(projection=_DEFAULT_PROJECTION)
ax.set_extent([-180, -46.5, 8, 60], ccrs.Geodetic()) # North America, in Geodetic CRS
ax.set_title("North America – Feature")
ax.add_feature(states_feature)

fig = plt.figure(figsize=(15, 15)) 
ax = plt.axes(projection=_DEFAULT_PROJECTION)
ax.set_extent([-180, -46.5, 8, 60], ccrs.Geodetic()) # North America, in Geodetic CRS
ax.set_title("North America – Feature")
ax.add_feature(states_feature)
ax.add_feature(feature.RIVERS)
ax.add_feature(feature.BORDERS, edgecolor='lime')

fig = plt.figure(figsize=(8, 8)) 
ax = plt.axes(projection=ccrs.Mercator().GOOGLE)
ax.set_extent([-180, -46.5, 8, 60], ccrs.Geodetic()) # North America, in Geodetic CRS
ax.set_title("North America – LambertAzimuthalEqualArea")
ax.add_feature(states_feature)
ax.gridlines(linestyle=":")
ax.tissot(color='orange', alpha=0.4)

fig = plt.figure(figsize=(8, 8)) 
ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea())
ax.set_extent([-180, -46.5, 8, 60], ccrs.Geodetic()) # North America, in Geodetic CRS
ax.set_title("North America – LambertAzimuthalEqualArea")
ax.add_feature(states_feature)
ax.gridlines(linestyle=":")
ax.tissot(color='orange', alpha=0.4)

fig = plt.figure(figsize=(8, 8)) 
ax = plt.axes(projection=ccrs.AlbersEqualArea())
ax.set_extent([-180, -46.5, 8, 60], ccrs.Geodetic()) # North America, in Geodetic CRS
ax.set_title("North America – AlbersEqualArea")
ax.add_feature(states_feature)
ax.gridlines(linestyle=":")
ax.tissot(color='orange', alpha=0.4)

fig = plt.figure(figsize=(8, 8)) 
ax = plt.axes(projection=ccrs.LambertConformal())
ax.set_extent([-180, -46.5, 8, 60], ccrs.Geodetic()) # North America, in Geodetic CRS
ax.set_title("North America – LambertConformal")
ax.add_feature(states_feature)
ax.gridlines(linestyle=":")
ax.tissot(color='orange', alpha=0.4)

WASHINGTON_NORTH = 2926
WASHINGTON_SOUTH = 2927
SEATTLE_BOUNDS = [-122.4596959,-122.2244331,47.4919119,47.734145]
WASHINGTON_BOUNDS = [-124.849,-116.9156,45.5435,49.0024]
SEATTLE_CENTER = (-122.3321, 47.6062)

fig = plt.figure(figsize=(8, 8)) 
ax = plt.axes(projection=ccrs.epsg(WASHINGTON_NORTH))
#ax.set_extent(<NO EXTENT>) # not setting bounds means we can see the full extent of the projected space.
ax.set_title("Washington – North (epsg:2926)")
ax.add_feature(states_feature)
ax.annotate('Seattle', xy=SEATTLE_CENTER, xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='red',
            ha='left', va='center')
ax.gridlines(linestyle=":")
ax.tissot(lats=range(43, 51), lons=range(-124, -116), alpha=0.4, rad_km=20000, color='orange')
plt.show()

fig = plt.figure(figsize=(8, 8)) 
ax = plt.axes(projection=ccrs.epsg(WASHINGTON_SOUTH))
#ax.set_extent(<NO EXTENT>) # not setting bounds means we can see the full extent of the projected space.
ax.set_title("Washington – South (epsg:2927)")
ax.annotate('Seattle', xy=SEATTLE_CENTER, xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='red',
            ha='left', va='center')
ax.add_feature(states_feature)
ax.tissot(lats=range(43, 51), lons=range(-124, -116), alpha=0.4, rad_km=20000, color='orange')
ax.gridlines(linestyle=":")


fig = plt.figure(figsize=(8, 8)) 
ax = plt.axes(projection=_DEFAULT_PROJECTION)
ax.set_extent(WASHINGTON_BOUNDS) # not setting bounds means we can see the full extent of the projected space.
ax.set_title("Washington – Web Mercator Reference (epsg:3857)")
ax.annotate('Seattle', xy=SEATTLE_CENTER, xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='red',
            ha='left', va='center')
ax.add_feature(states_feature)
gl = ax.gridlines(linestyle=":", draw_labels=True)
ax.tissot(lats=range(43, 51), lons=range(-124, -116), alpha=0.4, rad_km=20000, color='orange')

gl.xlabels_top = False
plt.show()

get_ipython().system(' unzip -o ../data/nps.zip ')

import geopandas as gpd 
parks = gpd.read_file("nps/")
print("%d parks in file" % len(parks))

fig = plt.figure(figsize=(15, 15)) 
ax = plt.axes(projection=_DEFAULT_PROJECTION)
ax.set_extent([-180, -46.5, 8, 60], ccrs.Geodetic()) # North America, in Geodetic CRS
ax.set_title("North America – National Parks")
ax.add_feature(states_feature, linewidth=0.1)
ax.add_feature(feature.RIVERS)
ax.add_geometries(parks.geometry.buffer(0), crs=ccrs.PlateCarree(), facecolor='red')

fig = plt.figure(figsize=(15,15)) 
ax = plt.axes(projection=ccrs.epsg(WASHINGTON_NORTH))
ax.set_title("Northern Washington – National Parks")
ax.add_feature(states_feature, linewidth=0.1)
ax.add_feature(feature.RIVERS)
ax.add_geometries(parks.geometry.buffer(0), crs=ccrs.PlateCarree(), facecolor='red')

ALASKA_ALBERS = 3338
fig = plt.figure(figsize=(15, 15)) 
ax = plt.axes(projection=ccrs.epsg(ALASKA_ALBERS))
ax.set_title("Alaska – National Parks")
ax.add_feature(states_feature, linewidth=0.1)
ax.add_feature(feature.RIVERS)
ax.add_geometries(parks.geometry.buffer(0), crs=ccrs.PlateCarree(), facecolor='red')



