get_ipython().magic('matplotlib inline')
from __future__ import print_function, unicode_literals
import sys, os
import datetime as dt
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from pygaarst import raster
import pandas as pd
import geopandas as gp
from fiona.crs import from_string
import shapely.ops
from shapely.geometry import Point
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

sys.path.append('../firedetection/')
import landsat8fire as lfire

sns.set(rc={'image.cmap': 'gist_heat'})
sns.set(rc={'image.cmap': 'bone'})

sns.set_context("poster")

myfontsize = 20
font = {'family' : 'Calibri',
        'weight': 'bold',
        'size'   : myfontsize}
matplotlib.rc('font', **font)
matplotlib.axes.rcParams['axes.labelsize']=myfontsize-4
matplotlib.axes.rcParams['axes.titlesize']=myfontsize
cmap1 =  matplotlib.colors.ListedColormap(sns.xkcd_palette(['white', 'bright red']))
cmap2 =  matplotlib.colors.ListedColormap(sns.xkcd_palette(['white', 'neon green']))
cmap3 =  matplotlib.colors.ListedColormap(sns.xkcd_palette(['white', 'orange']))

gisdir = "/Volumes/SCIENCE_mobile_Mac/Fire/DATA_BY_PROJECT/2015VIIRSMODIS/GISout/"
newviirspolydf = gp.GeoDataFrame.from_file(os.path.join(gisdir, 'WesternAKviirsspoly_20160719.shp'))

landsatpath = '/Volumes/SCIENCE_mobile_Mac/Fire/DATA_BY_PROJECT/2015VIIRSMODIS/Landsat/L8 OLI_TIRS 20150706/'
lsscene = 'LC80730142015187LGN00'
landsat = raster.Landsatscene(os.path.join(landsatpath, lsscene))

productdir = '/Volumes/SCIENCE_mobile_Mac/Fire/DATA_BY_PROJECT/2015VIIRSMODIS/rasterout/'

landsat.infix = '_clip'
rho7 = landsat.band7.reflectance
rho6 = landsat.band6.reflectance
rho5 = landsat.band5.reflectance
rho1 = landsat.band1.reflectance

xmax = landsat.band7.ncol
ymax = landsat.band7.nrow

reload(lfire)

water = lfire.get_l8watermask(landsat)

water_masked = np.ma.masked_where(
        ~water, np.ones((ymax, xmax)))

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')

ax1.pcolormesh(np.flipud(water_masked), cmap=cmap1, vmin=0, vmax=1)

allfire, highfire, anomfire, lowfire = lfire.get_l8fire(landsat, debug=True)

import h5py

h5f = h5py.File("../data/WesternAKLandsat20150706_firepix.h5", "w")

grp = h5f.create_group('datasets')
dset_all = grp.create_dataset("all_fires", data=allfire)
dset_high = grp.create_dataset("high_fires", data=highfire)
dset_anom = grp.create_dataset("anomalous_fires", data=anomfire)
dset_low = grp.create_dataset("low_fires", data=lowfire)
grp.attrs['date'] = '2016-07-17'
grp.attrs['source'] = 'LC80730142015187LGN00'
grp.attrs['author'] = 'Chris Waigl'
grp.attrs['project'] = '2015VIIRS'

h5f.close()

sum(sum(allfire))

allfire_masked = np.ma.masked_where(
        ~allfire, np.ones((ymax, xmax)))
highfire_masked = np.ma.masked_where(
        ~highfire, np.ones((ymax, xmax)))

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((0, xmax))
ax1.set_ylim((0, ymax))
dataplot = ax1.pcolormesh(np.flipud(rho7))
cbar = fig1.colorbar(dataplot, orientation='horizontal')

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((0, xmax))
ax1.set_ylim((0, ymax))
dataplot = ax1.pcolormesh(np.flipud(rho6))
cbar = fig1.colorbar(dataplot, orientation='horizontal')

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((0, xmax))
ax1.set_ylim((0, ymax))
dataplot = ax1.pcolormesh(np.flipud(rho5))
cbar = fig1.colorbar(dataplot, orientation='horizontal')

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((0, xmax))
ax1.set_ylim((0, ymax))
dataplot = ax1.pcolormesh(np.flipud(rho1))
cbar = fig1.colorbar(dataplot, orientation='horizontal')

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((0, xmax))
ax1.set_ylim((0, ymax))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_title("Landsat 8 fire detections over band 5 (NIR) reflectance")
dataplot = ax1.pcolormesh(np.flipud(rho5))
#maskplot3 = ax1.pcolormesh(np.flipud(firecond3_masked), cmap=cmap3, vmin=0, vmax=1)
#maskplot2 = ax1.pcolormesh(np.flipud(firecond2_masked), cmap=cmap2, vmin=0, vmax=1)
maskplot1 = ax1.pcolormesh(np.flipud(allfire_masked), cmap=cmap1, vmin=0, vmax=1)

outfn = 'WesternAK_landsatB5_firemask.png'
fig1.savefig(os.path.join(productdir, outfn), dpi=400, bb_inches='tight')

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((1800, 2800))
ax1.set_ylim((800, 1400))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_title("Landsat 8 fire detections over band 5 (NIR) reflectance")
dataplot = ax1.pcolormesh(np.flipud(rho5))
maskplot1 = ax1.pcolormesh(np.flipud(allfire_masked), cmap=cmap1, vmin=0, vmax=1)

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((1800, 2800))
ax1.set_ylim((800, 1400))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_title("Landsat 8 fire detections over band 5 (NIR) reflectance")
dataplot = ax1.pcolormesh(np.flipud(rho5))
#maskplot3 = ax1.pcolormesh(np.flipud(firecond3_masked), cmap=cmap3, vmin=0, vmax=1)
#maskplot2 = ax1.pcolormesh(np.flipud(firecond2_masked), cmap=cmap2, vmin=0, vmax=1)
maskplot1 = ax1.pcolormesh(np.flipud(firecond1_masked), cmap=cmap1, vmin=0, vmax=1)

landsat.meta['PRODUCT_METADATA']['DATE_ACQUIRED'], landsat.meta['PRODUCT_METADATA']['SCENE_CENTER_TIME']

crs = from_string(landsat.band7.proj4)
samplerecords = newviirspolydf[
    (newviirspolydf['DATE'] == '2015-07-06') & 
    (newviirspolydf['GMT'] == '2136')]
firepolygons_h = samplerecords[(samplerecords['Type'] != 'L')]['geometry'].to_crs(crs)
firepolygons_l = samplerecords[(samplerecords['Type'] == 'L')]['geometry'].to_crs(crs)

samplerecords['GMT'].value_counts()

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
X, Y =  landsat.band5._XY
Y = np.flipud(Y)
ax1.set_xlim((X[0][0], X[0, -1]))
ax1.set_ylim((Y[0][0], Y[-1, 0]))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.invert_yaxis()
ax1.set_title("High (yellow) and low (grey) intensity VIIRS fire detections over Landsat (red)")
dataplot = ax1.pcolormesh(X, Y, rho5)
patches = [PolygonPatch(poly, facecolor=(0, 0.55, 0.45, 0.25), edgecolor='lightgrey', lw=3) for poly in firepolygons_l]
ax1.add_collection(PatchCollection(patches, match_original=True))
patches = [PolygonPatch(poly, facecolor=(1, 0.35, 0, 0.25), edgecolor='yellow', lw=3) for poly in firepolygons_h]
ax1.add_collection(PatchCollection(patches, match_original=True))
maskplot1 = ax1.pcolormesh(X, Y, firecond1_masked, cmap=cmap1, vmin=0, vmax=1)

outfn = 'WesternAK_landsat_firemask_footprints.png'
fig1.savefig(os.path.join(productdir, outfn), dpi=300, bb_inches='tight')

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
X, Y =  landsat.band5._XY
Y = np.flipud(Y)
ax1.set_xlim((484000, 512000))
ax1.set_ylim((Y[0][0], 7310000))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.invert_yaxis()
ax1.set_title("High (yellow) and low (grey) intensity VIIRS fire detections over Landsat")
dataplot = ax1.pcolormesh(X, Y, rho5)
patches = [PolygonPatch(poly, facecolor=(0, 0.55, 0.45, 0.25), edgecolor='lightgrey', lw=3) for poly in firepolygons_l]
ax1.add_collection(PatchCollection(patches, match_original=True))
#patches = [PolygonPatch(poly, color='pink', alpha=0.75) for poly in firepolygons_h]
patches = [PolygonPatch(poly, facecolor=(1, 0.35, 0, 0.25), edgecolor='yellow', lw=3) for poly in firepolygons_h]
ax1.add_collection(PatchCollection(patches, match_original=True))
maskplot1 = ax1.pcolormesh(X, Y, allfire_masked, cmap=cmap1, vmin=0, vmax=1)

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
X, Y =  landsat.band5._XY
Y = np.flipud(Y)
ax1.set_xlim((474000, 505000))
ax1.set_ylim((7330000, Y[-1, 0]))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.invert_yaxis()
ax1.set_title("High (yellow) and low (grey) intensity VIIRS fire detections over Landsat")
dataplot = ax1.pcolormesh(X, Y, rho5)
maskplot1 = ax1.pcolormesh(X, Y, allfire_masked, cmap=cmap1, vmin=0, vmax=1)
patches = [PolygonPatch(poly, facecolor=(0, 0.55, 0.45, 0.25), edgecolor='lightgrey', lw=1.5) for poly in firepolygons_l]
ax1.add_collection(PatchCollection(patches, match_original=True))
#patches = [PolygonPatch(poly, color='pink', alpha=0.75) for poly in firepolygons_h]
patches = [PolygonPatch(poly, facecolor=(1, 0.35, 0, 0.25), edgecolor='yellow', lw=1.5) for poly in firepolygons_h]
ax1.add_collection(PatchCollection(patches, match_original=True))

rgb753scene = raster.GeoTIFF(os.path.join(landsatpath, lsscene, 'LC80730142015187LGN00_fc753_8bit_clip.tif'))
rgb753 = np.rollaxis(rgb753scene.data, 0, 3)

fig1 = plt.figure(1, figsize=(25, 20))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
X, Y =  landsat.band5._XY
Y = np.flipud(Y)
ax1.set_xlim((X[0][0], X[0, -1]))
ax1.set_ylim((Y[0][0], Y[-1, 0]))
#ax1.set_xlim((649000, X[0, -1]))
#ax1.set_ylim((Y[0][0], 6847000))
ax1.invert_yaxis()
ax1.set_xticklabels([])
ax1.set_yticklabels([])
#ax1.set_title("High (yellow) and low (grey) intensity VIIRS fire detections over Landsat (red)")
dataplot = ax1.imshow(np.flipud(rgb753), extent=[X[0][0], X[0, -1], Y[0][0], Y[-1, 0]], interpolation='none')
#patches = [PolygonPatch(poly, facecolor=(0, 0.55, 0.45, 0.25), edgecolor='lightgrey', lw=3) for poly in firepolygons_l]
#ax1.add_collection(PatchCollection(patches, match_original=True))
#patches = [PolygonPatch(poly, facecolor=(1, 0.35, 0, 0.25), edgecolor='yellow', lw=3) for poly in firepolygons_h]
#ax1.add_collection(PatchCollection(patches, match_original=True))
maskplot1 = ax1.pcolormesh(X, Y, allfire_masked, cmap=cmap3, vmin=0, vmax=1)
maskplot2 = ax1.pcolormesh(X, Y, highfire_masked, cmap=cmap1, vmin=0, vmax=1)

fig1 = plt.figure(1, figsize=(25, 20))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
X, Y =  landsat.band5._XY
Y = np.flipud(Y)
ax1.set_xlim((X[0][0], X[0, -1]))
ax1.set_ylim((Y[0][0], Y[-1, 0]))
ax1.invert_yaxis()
ax1.set_xticklabels([])
ax1.set_yticklabels([])
#ax1.set_title("High (yellow) and low (grey) intensity VIIRS fire detections over Landsat (red)")
dataplot = ax1.imshow(np.flipud(rgb753), extent=[X[0][0], X[0, -1], Y[0][0], Y[-1, 0]], interpolation='none')
patches = [PolygonPatch(poly, facecolor=(0, 0.55, 0.45, 0.25), edgecolor='lightgrey', lw=3) for poly in firepolygons_l]
ax1.add_collection(PatchCollection(patches, match_original=True))
patches = [PolygonPatch(poly, facecolor=(1, 0.35, 0, 0.25), edgecolor='yellow', lw=3) for poly in firepolygons_h]
ax1.add_collection(PatchCollection(patches, match_original=True))

samplerecords.to_crs(crs).head()

firehotspots = "/Volumes/SCIENCE_mobile_Mac/Fire/DATA_BY_PROJECT/2015VIIRSMODIS/activefiremaps.fs.fed.us_data_fireptdata/"
viirsIdir = "viirs_iband_fire_2015_344_ak_shapefile"
viirsIshp = "viirs_iband_fire_2015_344_ak_AKAlbers.shp"
viirsIDF = gp.GeoDataFrame.from_file(os.path.join(firehotspots, viirsIdir, viirsIshp))
ullat = 66.2
ullon = -155.0
lrlat = 65.7
lrlon = -150.0

testviirsDF = viirsIDF[(viirsIDF.LAT < ullat) & (viirsIDF.LAT > lrlat)
        & (viirsIDF.LONG < lrlon) & (viirsIDF.LONG > ullon)]
testviirsDF = testviirsDF.loc[(testviirsDF['DATE'] == '2015-07-06') & (testviirsDF['GMT'] == 2137)]

testviirsDF.shape

from functools import partial
from shapely.geometry import box

side = 375.0
def makebox(point, a=100.0, b=None):
    if not b: 
        b=a
    return box((point.x)-a/2, point.y-b/2, point.x+a/2, point.y+b/2)

makeboxes = partial(makebox, a=side)

boxseries = gp.GeoSeries(map(makeboxes, testviirsDF.geometry.values))
boxseries.crs = viirsIDF.geometry.crs
boxseries.to_crs(crs)

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
X, Y =  landsat.band5._XY
Y = np.flipud(Y)
ax1.set_xlim((484000, 512000))
ax1.set_ylim((Y[0][0], 7310000))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.invert_yaxis()
ax1.set_title("Global VIIRS-I-band fire detections over Landsat")
dataplot = ax1.pcolormesh(X, Y, rho5)
maskplot1 = ax1.pcolormesh(X, Y, firecond1_masked, cmap=cmap1, vmin=0, vmax=1)

boxseries = gp.GeoSeries(map(makeboxes, testviirsDF.geometry.values))
boxseries.crs = viirsIDF.geometry.crs

patches = [PolygonPatch(poly, color='orange', alpha=0.6) for poly in boxseries.to_crs(crs)]

if patches:
    ax1.add_collection(PatchCollection(patches, match_original=True))

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
X, Y =  landsat.band5._XY
Y = np.flipud(Y)
ax1.set_xlim((474000, 505000))
ax1.set_ylim((7330000, Y[-1, 0]))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.invert_yaxis()
ax1.set_title("Global VIIRS-I-band fire detections over Landsat")
dataplot = ax1.pcolormesh(X, Y, rho5)
maskplot1 = ax1.pcolormesh(X, Y, firecond1_masked, cmap=cmap1, vmin=0, vmax=1)

boxseries = gp.GeoSeries(map(makeboxes, testviirsDF.geometry.values))
boxseries.crs = viirsIDF.geometry.crs

patches = [PolygonPatch(poly, color='orange', alpha=0.6) for poly in boxseries.to_crs(crs)]

if patches:
    ax1.add_collection(PatchCollection(patches, match_original=True))

outfn = 'WesternAK_landsat_firemask_footprints_global_zoom.png'
fig1.savefig(os.path.join(productdir, outfn), dpi=300, bb_inches='tight')

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
X, Y =  landsat.band5._XY
Y = np.flipud(Y)
ax1.set_xlim((X[0][0], X[0, -1]))
ax1.set_ylim((Y[0][0], Y[-1, 0]))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.invert_yaxis()
ax1.set_title("Global VIIRS-I-band fire detections over Landsat")
dataplot = ax1.pcolormesh(X, Y, rho5)

boxseries = gp.GeoSeries(map(makeboxes, testviirsDF.geometry.values))
boxseries.crs = viirsIDF.geometry.crs

patches = [PolygonPatch(poly, color='orange', alpha=0.6) for poly in boxseries.to_crs(crs)]

if patches:
    ax1.add_collection(PatchCollection(patches, match_original=True))

firecond1idx = (ymax - np.where(firecond1)[0], np.where(firecond1)[1])
firecond1idx

ymax

firepoints_h = [Point(lon, lat) 
    for lon, lat in zip(landsat.band7.Lon_pxcenter[firecond1idx], landsat.band7.Lat_pxcenter[firecond1idx])]

samplerecords.loc[samplerecords['Type'] == 'A', 'Type'] = 'H'

samplerecords.head()

samplerecords['pointcount'] = 0

for idx, record in samplerecords.iterrows():
    poly = record['geometry']
    for pt in firepoints_h:
        if pt.within(poly):
            samplerecords.loc[idx, 'pointcount'] += 1

samplerecords.groupby('Type').describe()

samplerecords[['pointcount', 'Type']].boxplot(column='pointcount', by='Type', sym='k.')
#ax = plt.gca()
#ax.set_ylim(-2, 57)

fig2 = plt.figure(1, figsize=(15, 10))
ax2 = fig2.add_subplot(111)
sns.boxplot(x='Type', y='pointcount', data=samplerecords, showfliers=False,
    palette="Set3", saturation=0.3, width=.3, ax=ax2)
#sns.swarmplot(
#    x='Type', y='pointcount', data=samplerecords, color=".25", size=10, ax=ax2)
ax2.set_ylim(-2, 32)

outfn = 'WesternAK_landsat_firematch_boxplot.png'
fig2.savefig(os.path.join(productdir, outfn), dpi=200, bb_inches='tight')

ax = sns.violinplot(x='Type', y='pointcount', data=samplerecords, inner=None)

NBR = landsat.NBR

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((0, xmax))
ax1.set_ylim((0, ymax))
dataplot = ax1.pcolormesh(np.flipud(NBR), cmap='BrBG', vmin=-1., vmax=1.)
cbar = fig1.colorbar(dataplot, orientation='horizontal')

