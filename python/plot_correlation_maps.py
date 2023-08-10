import xarray as xr
import pandas as pd
import numpy as np
import fiona
import rasterio
import rasterio.mask
import matplotlib.pyplot as plt
import seaborn
import matplotlib
get_ipython().magic('matplotlib inline')
seaborn.set_style('dark')
from scipy import stats

from scipy.stats import linregress, pearsonr, spearmanr

corr_1M= xr.open_dataset('/g/data/oe9/project/team-drip/Spatial_temporal_correlation/SPI_NDVI/NDVI_SPI1M_Correlation.nc')
corr_1M

NDVI_SPI_1M = corr_1M.pearson_r.where(corr_1M.pearson_p < 0.05)
# NDVI_SPI_1M
path = '/g/data/oe9/project/team-drip/Spatial_temporal_correlation/Arc_gis_raster/NDVI_SPI_1M.nc'
NDVI_SPI_1M.to_netcdf(path)

corr_1M_EVI= xr.open_dataset('/g/data/oe9/project/team-drip/Spatial_temporal_correlation/SPI_EVI/EVI_SPI1M_Correlation.nc')
corr_1M_EVI

EVI_SPI_1M = corr_1M_EVI.pearson_r.where(corr_1M_EVI.pearson_p < 0.05)
# NDVI_SPI_1M
path = '/g/data/oe9/project/team-drip/Spatial_temporal_correlation/Arc_gis_raster/EVI_SPI_1M.nc'
EVI_SPI_1M.to_netcdf(path)

corr_3M= xr.open_dataset('/g/data/oe9/project/team-drip/Spatial_temporal_correlation/SPI_NDVI/NDVI_SPI3M_Correlation.nc')

NDVI_SPI_3M = corr_3M.pearson_r.where(corr_3M.pearson_p < 0.05)
# NDVI_SPI_1M
path = '/g/data/oe9/project/team-drip/Spatial_temporal_correlation/Arc_gis_raster/NDVI_SPI_3M.nc'
NDVI_SPI_3M.to_netcdf(path)

corr_3M_EVI= xr.open_dataset('/g/data/oe9/project/team-drip/Spatial_temporal_correlation/SPI_EVI/EVI_SPI3M_Correlation.nc')
corr_3M_EVI

EVI_SPI_3M = corr_3M_EVI.pearson_r.where(corr_3M_EVI.pearson_p < 0.05)
# NDVI_SPI_1M
path = '/g/data/oe9/project/team-drip/Spatial_temporal_correlation/Arc_gis_raster/EVI_SPI_3M.nc'
EVI_SPI_3M.to_netcdf(path)

corr_6M= xr.open_dataset('/g/data/oe9/project/team-drip/Spatial_temporal_correlation/SPI_NDVI/NDVI_SPI6M_Correlation.nc')

NDVI_SPI_6M = corr_6M.pearson_r.where(corr_6M.pearson_p < 0.05)
# NDVI_SPI_1M
path = '/g/data/oe9/project/team-drip/Spatial_temporal_correlation/Arc_gis_raster/NDVI_SPI_6M.nc'
NDVI_SPI_6M.to_netcdf(path)

corr_6M_EVI= xr.open_dataset('/g/data/oe9/project/team-drip/Spatial_temporal_correlation/SPI_EVI/EVI_SPI6M_Correlation.nc')

EVI_SPI_6M = corr_6M_EVI.pearson_r.where(corr_6M_EVI.pearson_p < 0.05)
# NDVI_SPI_1M
path = '/g/data/oe9/project/team-drip/Spatial_temporal_correlation/Arc_gis_raster/EVI_SPI_6M.nc'
EVI_SPI_6M.to_netcdf(path)

corr_12M = xr.open_dataset('/g/data/oe9/project/team-drip/Spatial_temporal_correlation/SPI_NDVI/NDVI_SPI12M_Correlation.nc')

NDVI_SPI_12M = corr_12M.pearson_r.where(corr_12M.pearson_p < 0.05)
# NDVI_SPI_1M
path = '/g/data/oe9/project/team-drip/Spatial_temporal_correlation/Arc_gis_raster/NDVI_SPI_12M.nc'
NDVI_SPI_12M.to_netcdf(path)

corr_12M_EVI= xr.open_dataset('/g/data/oe9/project/team-drip/Spatial_temporal_correlation/SPI_EVI/EVI_SPI12M_Correlation.nc')

EVI_SPI_12M = corr_12M_EVI.pearson_r.where(corr_12M_EVI.pearson_p < 0.05)
# NDVI_SPI_1M
path = '/g/data/oe9/project/team-drip/Spatial_temporal_correlation/Arc_gis_raster/EVI_SPI_12M.nc'
EVI_SPI_12M.to_netcdf(path)

corr_24M = xr.open_dataset('/g/data/oe9/project/team-drip/Spatial_temporal_correlation/SPI_NDVI/NDVI_SPI24M_Correlation.nc')

NDVI_SPI_24M = corr_24M.pearson_r.where(corr_24M.pearson_p < 0.05)
# NDVI_SPI_1M
path = '/g/data/oe9/project/team-drip/Spatial_temporal_correlation/Arc_gis_raster/NDVI_SPI_24M.nc'
NDVI_SPI_24M.to_netcdf(path)

corr_24M_EVI= xr.open_dataset('/g/data/oe9/project/team-drip/Spatial_temporal_correlation/SPI_EVI/EVI_SPI24M_Correlation.nc')

EVI_SPI_24M = corr_24M_EVI.pearson_r.where(corr_24M_EVI.pearson_p < 0.05)
# NDVI_SPI_1M
path = '/g/data/oe9/project/team-drip/Spatial_temporal_correlation/Arc_gis_raster/EVI_SPI_24M.nc'
EVI_SPI_24M.to_netcdf(path)

# create geometry coordinates of shapefile boundary 
# open the MDB shapefile 
with fiona.open("/home/563/sl1412/rainfall/mdb_boundary/mdb_boundary.shp", "r") as shapefile:
    geoms = [feature["geometry"] for feature in shapefile]

shapes = geoms[0]['coordinates'][0]


import shapefile   

shpFilePath = "/home/563/sl1412/rainfall/mdb_boundary/mdb_boundary.shp"  
listx=[]
listy=[]
test = shapefile.Reader(shpFilePath)
for sr in test.shapeRecords():
    for xNew,yNew in sr.shape.points:
        listx.append(xNew)
        listy.append(yNew)
plt.plot(listx,listy)
plt.show()

# define the colormap
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
cmaplist[0] = (.5,.5,.5,1.0)
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)


# define the bins and normalize
bounds = np.linspace(0,1,10)
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

seaborn.set_context('talk', font_scale=1.4)
title_size=25
fname = '/g/data/oe9/project/team-drip/results/correlation between SPI and VIs/spi_ndvi_evi_cor.png'

# # Create a figure with several subplots - three columns
fig, ax_s = plt.subplots(nrows = 2,ncols = 5,figsize=(45, 20))
# plt.title('Correlation between NDVI and SPI')
v_min = 0
v_max = 1

corr_1M.pearson_r.where(corr_1M.pearson_p < 0.05).plot.imshow(ax = ax_s[0,0],
                                                              robust=True, cmap=cmap, vmin=v_min, vmax=v_max)
ax_s[0,0].plot(listx,listy,color='black')
ax_s[0,0].set_title('NDVI vs SPI_1M',fontsize=title_size)

# cb = matplotlib.colorbar.Colorbar(ax_s).remove

ax_s[0,0].yaxis.set_label_text('Latitude',fontsize=30)
ax_s[0,0].xaxis.label.set_visible(False)


# cbar = plt.colorbar(im)
# cbar.remove()
# plt.draw()

# # create a second axes for the colorbar
# ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])


corr_3M.pearson_r.where(corr_3M.pearson_p < 0.05).plot.imshow(ax = ax_s[0,1],
                                                              robust=True, cmap=cmap, vmin=v_min, vmax=v_max)
ax_s[0,1].plot(listx,listy,color='black')
ax_s[0,1].set_title('NDVI vs SPI_3M',fontsize=title_size)
ax_s[0,1].xaxis.label.set_visible(False)
ax_s[0,1].yaxis.label.set_visible(False)


corr_6M.pearson_r.where(corr_6M.pearson_p < 0.05).plot.imshow(ax = ax_s[0,2],
                                                              robust=True, cmap=cmap, vmin=v_min, vmax=v_max)
ax_s[0,2].plot(listx,listy,color='black')
ax_s[0,2].set_title('NDVI vs SPI_6M',fontsize=title_size)
ax_s[0,2].xaxis.label.set_visible(False)
ax_s[0,2].yaxis.label.set_visible(False)

corr_12M.pearson_r.where(corr_12M.pearson_p < 0.05).plot.imshow(ax = ax_s[0,3],
                                                              robust=True, cmap=cmap, vmin=v_min, vmax=v_max)
ax_s[0,3].plot(listx,listy,color='black')
ax_s[0,3].set_title('NDVI vs SPI_12M',fontsize=title_size)
ax_s[0,3].xaxis.label.set_visible(False)
ax_s[0,3].yaxis.label.set_visible(False)


corr_24M.pearson_r.where(corr_24M.pearson_p < 0.05).plot.imshow(ax = ax_s[0,4],
                                                              robust=True, cmap=cmap, vmin=v_min, vmax=v_max)
ax_s[0,4].plot(listx,listy,color='black')
ax_s[0,4].set_title('NDVI vs SPI_24M',fontsize=title_size)

ax_s[0,4].xaxis.label.set_visible(False)
ax_s[0,4].yaxis.label.set_visible(False)

## EVI

corr_1M_EVI.pearson_r.where(corr_1M_EVI.pearson_p < 0.05).plot.imshow(ax = ax_s[1,0],
                                                              robust=True, cmap=cmap, vmin=v_min, vmax=v_max)
ax_s[1,0].plot(listx,listy,color='black')
ax_s[1,0].set_title('EVI vs SPI_1M',fontsize=title_size)
ax_s[1,0].yaxis.set_label_text('Latitude',fontsize=30)
# ax_s[0,1].xaxis.label.set_visible(False)
# ax_s[0,1].yaxis.label.set_visible(False)
ax_s[1,0].xaxis.set_label_text('Longitude',fontsize=30)

corr_3M_EVI.pearson_r.where(corr_3M_EVI.pearson_p < 0.05).plot.imshow(ax = ax_s[1,1],
                                                              robust=True, cmap=cmap, vmin=v_min, vmax=v_max)
ax_s[1,1].plot(listx,listy,color='black')
ax_s[1,1].set_title('EVI vs SPI_3M',fontsize=title_size)
# ax_s[1,1].xaxis.label.set_visible(False)
ax_s[1,1].yaxis.label.set_visible(False)
ax_s[1,1].xaxis.set_label_text('Longitude',fontsize=30)

corr_6M_EVI.pearson_r.where(corr_6M_EVI.pearson_p < 0.05).plot.imshow(ax = ax_s[1,2],
                                                              robust=True, cmap=cmap, vmin=v_min, vmax=v_max)
ax_s[1,2].plot(listx,listy,color='black')
ax_s[1,2].set_title('EVI vs SPI_6M',fontsize=title_size)
ax_s[1,2].xaxis.set_label_text('Longitude',fontsize=30)
ax_s[1,2].yaxis.label.set_visible(False)

corr_12M_EVI.pearson_r.where(corr_12M_EVI.pearson_p < 0.05).plot.imshow(ax = ax_s[1,3],
                                                              robust=True, cmap=cmap, vmin=v_min, vmax=v_max)
ax_s[1,3].plot(listx,listy,color='black')
ax_s[1,3].set_title('EVI vs SPI_12M',fontsize=title_size)
ax_s[1,3].xaxis.set_label_text('Longitude',fontsize=30)
ax_s[1,3].yaxis.label.set_visible(False)

corr_24M_EVI.pearson_r.where(corr_24M_EVI.pearson_p < 0.05).plot.imshow(ax = ax_s[1,4],
                                                              robust=True, cmap=cmap, vmin=v_min, vmax=v_max)
ax_s[1,4].plot(listx,listy,color='black')
ax_s[1,4].set_title('EVI vs SPI_24M',fontsize=title_size)
ax_s[1,4].xaxis.set_label_text('Longitude',fontsize=30)
ax_s[1,4].yaxis.label.set_visible(False)



fig.savefig(fname, dpi=600)



