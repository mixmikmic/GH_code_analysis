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
cmap1 =  matplotlib.colors.ListedColormap(sns.xkcd_palette(['white', 'red']))
cmap2 =  matplotlib.colors.ListedColormap(sns.xkcd_palette(['white', 'neon green']))
cmap3 =  matplotlib.colors.ListedColormap(sns.xkcd_palette(['white', 'orange']))

gisdir = "/Volumes/SCIENCE_mobile_Mac/Fire/DATA_BY_PROJECT/2015VIIRSMODIS/GISout/"
newviirspolydf = gp.GeoDataFrame.from_file(os.path.join(gisdir, 'EAGLEviirsspoly.shp'))

landsatpath = '/Volumes/SCIENCE_mobile_Mac/Fire/DATA_BY_PROJECT/2015VIIRSMODIS/Landsat/L8 OLI_TIRS Eagle/'
lsscene = 'LC8065014_15_2015147_clip'
landsat = raster.Landsatscene(os.path.join(landsatpath, lsscene))

productdir = '/Volumes/SCIENCE_mobile_Mac/Fire/DATA_BY_PROJECT/2015VIIRSMODIS/rasterout/'

rho7 = landsat.band7.reflectance
rho6 = landsat.band6.reflectance
rho5 = landsat.band5.reflectance
rho1 = landsat.band1.reflectance

xmax = landsat.band7.ncol
ymax = landsat.band7.nrow

allfire, highfire, anomfire, lowfire = lfire.get_l8fire(landsat)

allfire_masked = np.ma.masked_where(
        ~allfire, np.ones((ymax, xmax)))
highfire_masked = np.ma.masked_where(
        ~highfire, np.ones((ymax, xmax)))

print(sum(sum(lowfire)))
print(sum(sum(allfire)))

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((0, xmax))
ax1.set_ylim((0, ymax))
dataplot = ax1.pcolormesh(np.flipud(rho7))
cbar = fig1.colorbar(dataplot)

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((0, xmax))
ax1.set_ylim((0, ymax))
dataplot = ax1.pcolormesh(np.flipud(rho6))
cbar = fig1.colorbar(dataplot)

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((0, xmax))
ax1.set_ylim((0, ymax))
dataplot = ax1.pcolormesh(np.flipud(rho5))
cbar = fig1.colorbar(dataplot)

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((0, xmax))
ax1.set_ylim((0, ymax))
dataplot = ax1.pcolormesh(np.flipud(rho1))
cbar = fig1.colorbar(dataplot)

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((0, xmax))
ax1.set_ylim((0, ymax))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_title("Landsat 8 fire detections over band 5 (NIR) reflectance")
dataplot = ax1.imshow(np.flipud(rho5))
maskplot1 = ax1.pcolormesh(np.flipud(allfire_masked), cmap=cmap1, vmin=0, vmax=1)

outfn = 'eagle_landsatB5_firemask2.png'
#fig1.savefig(os.path.join(productdir, outfn), dpi=200, bb_inches='tight')
fig1.savefig(os.path.join(productdir, outfn))

landsat.meta['PRODUCT_METADATA']['DATE_ACQUIRED'], landsat.meta['PRODUCT_METADATA']['SCENE_CENTER_TIME']

crs = from_string(landsat.band7.proj4)
samplerecords = newviirspolydf[
    (newviirspolydf['DATE'] == '2015-05-27')
    & (newviirspolydf['GMT'] == '2045')
    & (newviirspolydf['LAT'] < 65.2)
    & (newviirspolydf['LAT'] > 64.8)]
firepolygons_h = samplerecords[(samplerecords['Type'] != 'L')]['geometry'].to_crs(crs)
firepolygons_l = samplerecords[(samplerecords['Type'] == 'L')]['geometry'].to_crs(crs)

samplerecords

samplerecords['geometry'].plot()

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
patches = [PolygonPatch(poly, facecolor=(0, 0.55, 0.45, 0.25), edgecolor='lightgrey', lw=2) for poly in firepolygons_l]
ax1.add_collection(PatchCollection(patches, match_original=True))
patches = [PolygonPatch(poly, facecolor=(1, 0.35, 0, 0.25), edgecolor='yellow', lw=2) for poly in firepolygons_h]
ax1.add_collection(PatchCollection(patches, match_original=True))
maskplot1 = ax1.pcolormesh(X, Y, allfire_masked, cmap=cmap1, vmin=0, vmax=1)

outfn = 'eagle_landsat_firemask_footprints.png'
fig1.savefig(os.path.join(productdir, outfn), dpi=200, bb_inches='tight')

rgb753scene = raster.GeoTIFF(os.path.join(landsatpath, lsscene, 'LC8065014_15_2015147_clip_fc753_8bit.tif'))
rgb753 = np.rollaxis(rgb753scene.data, 0, 3)

fig1 = plt.figure(1, figsize=(25, 20))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
X, Y =  landsat.band5._XY
Y = np.flipud(Y)
ax1.set_xlim((X[0][0], X[0, -1]))
ax1.set_ylim((Y[0][0], Y[-1, 0]))
ax1.set_xlim((465500, 495000))
ax1.set_ylim((7225000 , 7195000))
ax1.invert_yaxis()
#ax1.set_title("High (yellow) and low (grey) intensity VIIRS fire detections over Landsat (red)")
dataplot = ax1.imshow(np.flipud(rgb753), extent=[X[0][0], X[0, -1], Y[0][0], Y[-1, 0]], interpolation='none')
patches = [PolygonPatch(poly, facecolor=(0, 0.55, 0.45, 0.25), edgecolor='lightgrey', lw=3) for poly in firepolygons_l]
ax1.add_collection(PatchCollection(patches, match_original=True))
patches = [PolygonPatch(poly, facecolor=(1, 0.35, 0, 0.25), edgecolor='yellow', lw=3) for poly in firepolygons_h]
ax1.add_collection(PatchCollection(patches, match_original=True))
maskplot1 = ax1.pcolormesh(X, Y, allfire_masked, cmap=cmap3, vmin=0, vmax=1)
maskplot2 = ax1.pcolormesh(X, Y, highfire_masked, cmap=cmap1, vmin=0, vmax=1)

sns.set_style('white')

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
X, Y =  landsat.band5._XY
Y = np.flipud(Y)
ax1.set_xlim((X[0][0], X[0, -1]))
ax1.set_ylim((Y[0][0], Y[-1, 0]))
ax1.set_xlim((465500, 495000))
ax1.set_ylim((7225000 , 7195000))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.invert_yaxis()
#ax1.set_title("High (yellow) and low (grey) intensity VIIRS fire detections over Landsat (red)")
dataplot = ax1.imshow(np.flipud(rgb753), extent=[X[0][0], X[0, -1], Y[0][0], Y[-1, 0]], interpolation='none')
patches1 = [PolygonPatch(poly, facecolor=(0, 0.55, 0.45, 0.25), edgecolor='lightgrey', lw=3) for poly in firepolygons_l]
ax1.add_collection(PatchCollection(patches1, match_original=True))
patches2 = [PolygonPatch(poly, facecolor=(1, 0.35, 0, 0.25), edgecolor='yellow', lw=3) for poly in firepolygons_h]
ax1.add_collection(PatchCollection(patches2, match_original=True))
maskplot1 = ax1.pcolormesh(X, Y, allfire_masked, cmap=cmap1, vmin=0, vmax=1)

x1, x2, y1, y2 = 480000, 483000, 7195500, 7198500
axins1 = zoomed_inset_axes(ax1, 3, loc=5)
axins1.imshow(np.flipud(rgb753), extent=[X[0][0], X[0, -1], Y[0][0], Y[-1, 0]], interpolation='none')
axins1.add_collection(PatchCollection(patches1, match_original=True))
axins1.add_collection(PatchCollection(patches2, match_original=True))
axins1.pcolormesh(X, Y, allfire_masked, cmap=cmap1, vmin=0, vmax=1)
axins1.set_xlim((x1, x2))
axins1.set_ylim((y1, y2))
axins1.set_xticklabels([])
axins1.set_yticklabels([])
mark_inset(ax1, axins1, loc1=2, loc2=4, fc='none', ec='0.1', lw=2)

x1, x2, y1, y2 = 469000, 472000, 7219000, 7222000
axins3 = zoomed_inset_axes(ax1, 3, loc=1)
axins3.imshow(np.flipud(rgb753), extent=[X[0][0], X[0, -1], Y[0][0], Y[-1, 0]], interpolation='none')
axins3.add_collection(PatchCollection(patches1, match_original=True))
axins3.add_collection(PatchCollection(patches2, match_original=True))
axins3.pcolormesh(X, Y, allfire_masked, cmap=cmap1, vmin=0, vmax=1)
axins3.set_xlim((x1, x2))
axins3.set_ylim((y1, y2))
axins3.set_xticklabels([])
axins3.set_yticklabels([])
mark_inset(ax1, axins3, loc1=2, loc2=3, fc='none', ec='0.1', lw=2)

for axis in ['top', 'bottom', 'left', 'right']:
    for ax in axins1, axins3:
        ax.spines[axis].set_color('0.1')
        ax.spines[axis].set_linewidth(4)

outfn = 'eagle_landsat_firemask_footprints01.png'
fig1.savefig(os.path.join(productdir, outfn), dpi=400, bb_inches='tight')

sns.set_style('darkgrid')

samplerecords.drop_duplicates(subset=['LAT', 'LON', 'DATE', 'GMT']).to_crs(crs)

firecond1idx = (ymax - np.where(allfire)[0], np.where(allfire)[1])
firecond1idx

ymax

firepoints_h = [Point(lon, lat) 
    for lon, lat in zip(landsat.band7.Lon_pxcenter[firecond1idx], landsat.band7.Lat_pxcenter[firecond1idx])]

samplerecords['pointcount'] = 0
for idx, record in samplerecords.iterrows():
    poly = record['geometry']
    for pt in firepoints_h:
        if pt.within(poly):
            samplerecords.loc[idx, 'pointcount'] += 1

samplerecords.loc[samplerecords['Type'] == 'A', 'Type'] = 'H'

samplerecords = samplerecords.drop_duplicates(subset=['LAT', 'LON', 'DATE', 'GMT'])

samplerecords.groupby('Type').describe()

samplerecords[['pointcount', 'Type']].boxplot(column='pointcount', by='Type', sym='k.')
ax = plt.gca()
ax.set_ylim(-2, 60)

myfig = plt.gcf()
outfn = 'eagle_landsat_firematch_boxplot.png'
myfig.savefig(os.path.join(productdir, outfn), dpi=200, bb_inches='tight')

fig2 = plt.figure(1, figsize=(15, 10))
ax2 = fig2.add_subplot(111)
sns.boxplot(x='Type', y='pointcount', data=samplerecords,
    palette="Set3", saturation=0.3, width=.3, ax=ax2)
sns.swarmplot(
    x='Type', y='pointcount', data=samplerecords, color=".25", size=10, ax=ax2)
ax2.set_ylim(-2, 67)

outfn = 'eagle_landsat_firematch_boxplot.png'
fig2.savefig(os.path.join(productdir, outfn), dpi=200, bb_inches='tight')

ax = sns.violinplot(x='Type', y='pointcount', data=samplerecords, inner=None)

NBR = landsat.NBR

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((0, xmax))
ax1.set_ylim((0, ymax))
dataplot = ax1.pcolormesh(np.flipud(NBR), cmap='BrBG', vmin=-1., vmax=1.)
cbar = fig1.colorbar(dataplot)

scene_before = 'LC80660142015138LGN00'
scene_after = 'LC80660142015170LGN00'
landsat_before = raster.Landsatscene(os.path.join(landsatpath, scene_before))
landsat_after = raster.Landsatscene(os.path.join(landsatpath, scene_after))
landsat_before.infix = '_clip'
landsat_after.infix = '_clip'

NBR_before = landsat_before.NBR
NBR_after = landsat_after.NBR

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((0, xmax))
ax1.set_ylim((0, ymax))
dataplot = ax1.pcolormesh(np.flipud(NBR_before), cmap='BrBG', vmin=-1., vmax=1.)
cbar = fig1.colorbar(dataplot)

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((0, xmax))
ax1.set_ylim((0, ymax))
dataplot = ax1.pcolormesh(np.flipud(NBR_after), cmap='BrBG', vmin=-1., vmax=1.)
cbar = fig1.colorbar(dataplot)

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlim((0, xmax))
ax1.set_ylim((0, ymax))
dataplot = ax1.pcolormesh(np.flipud(NBR_before - NBR_after)*1000, cmap='BrBG_r', vmin=-800., vmax=1300.)
cbar = fig1.colorbar(dataplot)

crs = from_string(landsat_after.band7.proj4)
firepolygons = newviirspolydf['geometry'].to_crs(crs)

fig1 = plt.figure(1, figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal')
X, Y =  landsat_after.band5._XY
Y = np.flipud(Y)
ax1.set_xlim((X[0][0], X[0, -1]))
ax1.set_ylim((Y[0][0], Y[-1, 0]))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.invert_yaxis()
ax1.set_title("VIIRS fire detections on top of dNBR")
dataplot = ax1.pcolormesh(X, Y, (NBR_before - NBR_after)*1000, cmap='BrBG_r', vmin=-800., vmax=1300.)
patches = [PolygonPatch(poly, color='salmon', alpha=0.6) for poly in firepolygons]
ax1.add_collection(PatchCollection(patches, match_original=True))



