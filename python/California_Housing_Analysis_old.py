get_ipython().run_line_magic('matplotlib', 'inline')

import cartopy
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
# Adjust size of figures
plt.rcParams['figure.figsize'] = (10.0, 8.0)

# Define parameters
# Select features by index position
# Run this cell for data description with list of feature names
feature_ids = [0, 1, 2, 3, 4, 5, 6, 7]
test_size = 0.2
test_randomstate = 1
ll_latlon = [38.9, -124.6]
ur_latlon = [42.4, -119.8]

# Download/load california housing data
calh = fetch_california_housing()

# Extract selected features from input dataset
if len(feature_ids) == 1:
    calh_X = calh.data[:, np.newaxis, feature_ids[0]]
else:
    calh_X = calh.data[:, feature_ids]
calh_y = calh.target
# Filter rows of feature array, X, based on ll_latlon and ur_latlon

# Extract separate feature array with only latitude and longitude (needed?)
calh_latlon = calh.data[:, [6, 7]]
#print(calh_latlon[0,:])
#print(calh_X.shape)
print("Description: \n{}\n".format(calh.DESCR))
print("Feature names (in order): \n{}\n".format(calh.feature_names))


# Split the data and target into training/testing sets
calh_X_train, calh_X_test, calh_y_train,     calh_y_test = train_test_split(calh_X,         calh.target, test_size=test_size, random_state=test_randomstate
)

# Create linear regression object for modeling
#intercept = 0
#coef = [1000]
#model = linear_model.LinearRegression()
#model.intercept_ = intercept
#model.coef_ = np.array(coef)
# Generate model data using intercept and coef
#calh_y_mod = model.predict(calh_X_test)

# Create linear regression object for inversion
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(calh_X_train, calh_y_train)

# Make predictions using the testing set
calh_y_pred = regr.predict(calh_X_test)

# The coefficients
print("Intercept: {}".format(regr.intercept_))
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: {0:.2f}".format(         mean_squared_error(calh_y_test, calh_y_pred))
)
# Explained variance score: 1 is perfect prediction
print("Variance (R^2) score: {0:.2f}".format(         r2_score(calh_y_test, calh_y_pred))
)

# Plot outputs
plt_featid = 6
plt.scatter(calh_X_train[:,plt_featid], calh_y_train,  color='black')
plt.scatter(calh_X_test[:,plt_featid], calh_y_test,  color='red')
plt.scatter(calh_X_test[:,plt_featid], calh_y_pred, color='blue')
#plt.plot(calh_X_test, calh_y_mod, color='red', linewidth=3)
#print(calh_y_pred)
#plt.xticks(())
#plt.yticks(())

plt.show()

# Define parameters and get lat/lon limits
projection = "merc"
#projection = "cyl"
resolution = "f"
area_thresh = 1
lat_buf = 1
lon_buf = 1
plt_type = "scatter"
gridsize = 20
mincnt = 1
cmap = 'YlOrBr'

if len(ll_latlon) != len(ur_latlon) or len(ll_latlon) != 2:
    ll_latlon = [             calh_latlon[:,0].min() - lat_buf,             calh_latlon[:,1].min() - lon_buf
    ]
    ur_latlon = [             calh_latlon[:,0].max() + lat_buf,             calh_latlon[:,1].max() + lon_buf
    ]
ll_latlon = np.array(ll_latlon)
ur_latlon = np.array(ur_latlon)
mm_latlon = (ll_latlon + ur_latlon) / 2

# Generate Map for plotting geolocated data
# Make basemap instance and draw features on it
my_map = Basemap(         projection = projection,         #epsg = 4326, ellps = "WGS84", \
        lat_0 = mm_latlon[0], lon_0 = mm_latlon[1],
        resolution = resolution, area_thresh = area_thresh,
        llcrnrlon=ll_latlon[1], llcrnrlat=ll_latlon[0],
        urcrnrlon=ur_latlon[1], urcrnrlat=ur_latlon[0]
)
 
my_map.drawcoastlines()
my_map.drawcountries()
my_map.drawstates()
my_map.drawrivers()
#my_map.fillcontinents(color = 'coral')
#my_map.arcgisimage()
my_map.bluemarble()
#my_map.etopo()
#my_map.drawmapboundary()

#print(help(my_map))
if plt_type == "scatter":
    x, y = my_map(calh_latlon[:, 1], calh_latlon[:, 0])
    my_map.scatter(x, y, color='red', marker='o', zorder=10)
elif plt_type == "hexbin":
    x, y = my_map(calh_latlon[:, 1], calh_latlon[:, 0])
    my_map.hexbin(x, y, gridsize=gridsize, mincnt=mincnt, cmap=cmap)
    #my_map.colorbar(location='bottom', label='Count')
    
#my_map.drawmapscale()

plt.show()





# This is a STRM shaded relief map
"""
This example illustrates the automatic download of
STRM data, gap filling (using gdal) and adding shading
to create a so-called "Shaded Relief SRTM".

Originally contributed by Thomas Lecocq (http://geophysique.be).

"""
import cartopy.crs as ccrs
from cartopy.io import srtm
import matplotlib.pyplot as plt

from cartopy.io import PostprocessedRasterSource, LocatedImage
from cartopy.io.srtm import SRTM3Source

def fill_and_shade(located_elevations):
    """
    Given an array of elevations in a LocatedImage, fill any holes in
    the data and add a relief (shadows) to give a realistic 3d appearance.

    """
    new_elevations = srtm.fill_gaps(located_elevations.image, max_distance=15)
    new_img = srtm.add_shading(new_elevations, azimuth=135, altitude=15)
    return LocatedImage(new_img, located_elevations.extent)

ax = plt.axes(projection=ccrs.PlateCarree())

# Define a raster source which uses the SRTM3 data and applies the
# fill_and_shade function when the data is retrieved.
shaded_srtm = PostprocessedRasterSource(SRTM3Source(), fill_and_shade)

# Add the shaded SRTM source to our map with a grayscale colormap.
ax.add_raster(shaded_srtm, cmap='Greys')

# This data is high resolution, so pick a small area which has some
# interesting orography.
#ax.set_extent([ll_latlon[1], ur_latlon[1], ll_latlon[0], ur_latlon[0]])
ax.set_extent([12, 13, 47, 48])

plt.title("SRTM Shaded Relief Map")

gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_left = False

plt.show()

import matplotlib.pyplot as plt

from cartopy.io.img_tiles import StamenTerrain

tiler = StamenTerrain()
mercator = tiler.crs
ax = plt.axes(projection=mercator)
ax.set_extent([ll_latlon[1], ur_latlon[1], ll_latlon[0], ur_latlon[0]])
#ax.set_extent([-90, -73, 22, 34])

ax.add_image(tiler, 6)

ax.coastlines('10m')
plt.show()

# Using Web Map Tile Service (WMPS) to get satellite images
# Following http://www.net-analysis.com/blog/cartopyimages.html
import cartopy.crs as ccrs
#from cartopy.io.img_tiles import OSM
#import cartopy.feature as cfeature
#from cartopy.io import shapereader
#from cartopy.io.img_tiles import StamenTerrain
#from cartopy.io.img_tiles import GoogleTiles
from owslib.wmts import WebMapTileService
#import matplotlib.pyplot as plt
#from matplotlib.path import Path
#import matplotlib.patheffects as PathEffects
#import matplotlib.patches as mpatches
#import numpy as np

# Selecting WTMS tiles
# URL of NASA GIBS
URL = 'http://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi'
wmts = WebMapTileService(URL)
# Layers for MODIS true color and snow RGB
layers = ['MODIS_Terra_SurfaceReflectance_Bands143',         'MODIS_Terra_CorrectedReflectance_Bands367']
date_str = '2010-07-12'
# Plotting

fig_dpi = 250
figsize_y = 8
plot_datapoints = True
savefig_name = "NCal_modis_olric_hvals.png"

# Define two coordinate reference systems (i.e. projections)
# and set plot bounding box, etc.
plot_CRS = ccrs.Mercator()
geodetic_CRS = ccrs.Geodetic()

lat0 = ll_latlon[0]
lat1 = ur_latlon[0]
lon0 = ll_latlon[1]
lon1 = ur_latlon[1]
# Calculate x,y pairs for lower-left and upper-right map corners
x0, y0 = plot_CRS.transform_point(lon0, lat0, geodetic_CRS)
x1, y1 = plot_CRS.transform_point(lon1, lat1, geodetic_CRS)
#print("x0, y0: {}, {}".format(x0, y0))
#print("x1, y1: {}, {}".format(x1, y1))
# Calculate x,y pairs for housing data lat/lon pairs
calh_xy = np.zeros((calh_latlon.shape[0], 2))
for ri, (lat, lon) in enumerate(calh_latlon):
    calh_xy[ri,:] = plot_CRS.transform_point(lon, lat, geodetic_CRS)
print(calh_xy[1000,:])
# The following gives aspect ratio for Mercator projection (?)
figsize_x = 2 * figsize_y * (x1 - x0) / (y1 - y0)
fig = plt.figure(figsize=(figsize_x, figsize_y), dpi=fig_dpi)

# Add Satellite image(s) and coastline, then show figure
RIVERS_10m = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m',
                                 edgecolor=cartopy.feature.COLORS['water'],
                                 facecolor='none', linewidth=2)
RIVERS_NA_10m = cartopy.feature.NaturalEarthFeature('physical', 'rivers_north_america', '10m',
                                 edgecolor=cartopy.feature.COLORS['water'],
                                 facecolor='none', linewidth=1)
LAKES_10m = cartopy.feature.NaturalEarthFeature('physical', 'lakes', '10m',
                                 edgecolor='c',
                                 facecolor='c')
LAKES_NA_10m = cartopy.feature.NaturalEarthFeature('physical', 'lakes_north_america', '10m',
                                 edgecolor='c',
                                 facecolor='c', linewidth=1)
OCEAN_10m = cartopy.feature.NaturalEarthFeature('physical', 'ocean', '10m',
                                 edgecolor='b',
                                 facecolor='b')
MINOR_ISLANDS_COASTLINE_10m = cartopy.feature.NaturalEarthFeature('physical', 'minor_islands_coastline', '10m',
                                 edgecolor='k',
                                 facecolor='none')
COASTLINE_10m = cartopy.feature.NaturalEarthFeature('physical', 'coastline', '10m',
                                 edgecolor='k',
                                 facecolor='none')
BORDERS2_10m = cartopy.feature.NaturalEarthFeature('cultural', 'admin_1_states_provinces',
                                  '10m', edgecolor='grey', facecolor='none')
    #"""country boundaries.""""

ax = plt.axes([0, 0, 1, 1], projection=plot_CRS)
ax.set_xlim((x0, x1))
ax.set_ylim((y0, y1))
ax.add_wmts(wmts, layers[0], wmts_kwargs={'time': date_str})
# Add 
ax.add_feature(OCEAN_10m, zorder=1)
ax.add_feature(LAKES_NA_10m, zorder=1)
ax.add_feature(LAKES_10m, zorder=1)
ax.add_feature(RIVERS_NA_10m, zorder=1)
ax.add_feature(RIVERS_10m, zorder=1)
ax.add_feature(BORDERS2_10m, zorder=2)
ax.add_feature(MINOR_ISLANDS_COASTLINE_10m, zorder=2)
ax.add_feature(COASTLINE_10m, zorder=2)

# Add California housing data points
if plot_datapoints:
    ax.scatter(calh_latlon[:, 1], calh_latlon[:, 0], c=calh_y, cmap='jet',                marker='o', edgecolor='none', facecolor='none',                alpha=1, zorder=3, transform=ccrs.Geodetic()
    )
minlonind = calh_latlon[:,1].argmin()
maxlatind = calh_latlon[:,0].argmax()
print(calh_latlon[minlonind, :])
print(calh_latlon[maxlatind, :])
#print(help(ax.get_images))
#imax = ax.get_images()
#imdat = imax[0].make_image("png")
#print(help(imax[0].make_image))
plt.savefig(savefig_name,dpi=fig_dpi, bbox_inches='tight')
#plt.show()
plt.close()

# Load image that was just written to file and define its x/y grid
# Inspired in part by http://cgcooke.github.io/GDAL/
img  = np.asarray(Image.open(savefig_name))
print(img.shape)
#print(img[1000, 1000, :])
#print(img[img[:, :, 3]==255].shape)

# Define two lists with the x- and y-values of each pixel
yPixels, xPixels, nBand = img.shape  # number of pixels in y, x
pixelXSize =(x1-x0)/xPixels # size of the pixel in X direction     
pixelYSize = (y0-y1)/yPixels # size of the pixel in Y direction
pixelX = np.arange(x0, x1, pixelXSize)
pixelY = np.arange(y1, y0, pixelYSize)
#print(len(pixelX))
#print(len(pixelY))

# Identify image around a given cal housing data point

print(pixelXSize)







