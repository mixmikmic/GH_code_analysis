get_ipython().magic('matplotlib inline')

import contextily as ctx
import geopandas as gpd
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt

# Data
from pysal.examples import get_path

tx = gpd.read_file(get_path('us48.shp')).set_index('STATE_ABBR').loc[['TX'], 'geometry']
tx.crs = {'init': 'epsg:4326'}
tx['TX']

w, s, e, n = tx['TX'].bounds
w, s, e, n

_ = ctx.howmany(w, s, e, n, 6, ll=True)

get_ipython().magic('time img, ext = ctx.bounds2img(w, s, e, n, 6, ll=True)')

plt.imshow(img, extent=ext);

get_ipython().magic("time _ = ctx.bounds2raster(w, s, e, n, 6, 'tx.tif', ll=True)")

rtr = rio.open('tx.tif')

# NOTE the transpose of the image data
img = np.array([ band for band in rtr.read() ]).transpose(1, 2, 0)
# Plot
plt.imshow(img, extent=rtr.bounds);

# Mercator coordinates for Houston area
hou = (-10676650.69219051, 3441477.046670125, -10576977.7804825, 3523606.146650609)

# Window
wdw = ctx.bb2wdw(hou, rtr)
# Raster subset
sub = np.array([ rtr.read(band, window=wdw)       for band in range(1, rtr.count+1)]).transpose(1, 2, 0)
# Plot
plt.imshow(sub, extent=(hou[0], hou[2], hou[1], hou[3]));

# Shortify the bound box named tuple
bb = rtr.bounds
# Set up the figure
f, ax = plt.subplots(1, figsize=(9, 9))
# Load the tile raster (note the re-arrangement of the bounds)
ax.imshow(img, extent=(bb.left, bb.right, bb.bottom, bb.top))
# Overlay the polygon on top (note we reproject it to the raster's CRS)
tx.to_crs(rtr.crs).plot(edgecolor='none', ax=ax)
# Remove axis for aesthetics
ax.set_axis_off()
# Show
plt.show()

sources = [i for i in dir(ctx.tile_providers) if i[0]!='_']
sources

f, axs = plt.subplots(2, 5, figsize=(25, 10))
axs = axs.flatten()
for src, ax in zip(sources, axs):
    img, ext = ctx.bounds2img(w, s, e, n, 6, url=getattr(ctx.sources, src), ll=True)
    ax.imshow(img, extent=ext)
    ax.set_title(src)
    ax.set_axis_off()
plt.show()

