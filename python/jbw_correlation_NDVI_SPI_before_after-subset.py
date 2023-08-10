import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# import seaborn

get_ipython().magic('matplotlib inline')
# seaborn.set_style('dark')
from scipy import stats

from scipy.stats import linregress, pearsonr, spearmanr

spi = xr.open_dataset('/g/data/oe9/project/team-drip/Rainfall/SPI_awap/SPI_12M_masked.nc')
spi_sub = spi.isel(time=range(1,204))
spi_sub_dt = spi_sub.SPI_12M.isel(time=range(0,131))
spi_sub_dt
spi_sub_rc = spi_sub.SPI_12M.isel(time=range(131,203))
spi_sub_rc
# ET

coarse_NDVI= xr.open_dataarray('/g/data/oe9/project/team-drip/resampled_NDVI/coarse_NDVI.nc')
NDVI_dt = coarse_NDVI.isel(time=range(0,131))
NDVI_rc = coarse_NDVI.isel(time=range(131,203))
NDVI_rc

climatology_dt = coarse_NDVI.groupby('time.month').mean('time')
anomalies_NDVI = coarse_NDVI.groupby('time.month') - climatology_dt
anomalies_NDVI_dt = anomalies_NDVI.isel(time=range(0,131))
anomalies_NDVI_rc = anomalies_NDVI.isel(time=range(131,203))
anomalies_NDVI_rc

# Start by setting up a new dataset, with empty arrays along latitude and longitude
dims = ('latitude', 'longitude')
coords = {d: spi_sub_dt[d] for d in dims}
correlation_data = {
    name: xr.DataArray(data=np.ndarray([len(spi_sub_dt[d]) for d in dims]),
                       name=name, dims=dims)
    for name in 'pearson_r pearson_p spearman_r spearman_p'.split()
}
corr_spi_sub_dt = xr.Dataset(data_vars=correlation_data, coords=coords)
corr_spi_sub_dt


# By looping, we make a list of lists of correlations
lat=1
lon=1
NDVI = anomalies_NDVI_dt.isel(latitude=lat, longitude=lon)
#         SPI = spi_1M_sub.sel(latitude=lat, longitude=lon)
SPI_1M = spi_sub_dt.isel(latitude=lat, longitude=lon)
mask = ~np.isinf(SPI_1M)
subset_NDVI= NDVI.where(mask, drop=True)
subset_SPI_1M= SPI_1M.where(mask, drop=True)

val = pearsonr(subset_NDVI,subset_SPI_1M)
# try:
#     # Spearman's R can fail for some values
#     val += spearmanr(NDVI,SPI_1M)
# except ValueError:
val += (np.nan, np.nan)
val

get_ipython().run_cell_magic('time', '', "# By looping, we make a list of lists of correlations\nlatout = []\nfor lat in anomalies_NDVI_dt.latitude:\n    lonout = []\n    latout.append(lonout)\n    for lon in anomalies_NDVI_dt.longitude:\n        NDVI = anomalies_NDVI_dt.sel(latitude=lat, longitude=lon)\n#         SPI = spi_1M_sub.sel(latitude=lat, longitude=lon)\n        SPI_1M = spi_sub_dt.sel(latitude=lat, longitude=lon)\n        mask = ~np.isinf(SPI_1M)\n        subset_NDVI= NDVI.where(mask, drop=True)\n        subset_SPI_1M= SPI_1M.where(mask, drop=True)\n        \n        val = pearsonr(subset_NDVI,subset_SPI_1M)\n        try:\n            # Spearman's R can fail for some values\n            val += spearmanr(NDVI,SPI_1M)\n        except ValueError:\n            val += (np.nan, np.nan)\n        lonout.append(val)\n# Then we convert the lists to an array\narr = np.array(latout)\n# And finally insert the pieces into our correlation dataset\ncorr_spi_sub_dt.pearson_r[:] = arr[..., 0]\ncorr_spi_sub_dt.pearson_p[:] = arr[..., 1]\ncorr_spi_sub_dt.spearman_r[:] = arr[..., 2]\ncorr_spi_sub_dt.spearman_p[:] = arr[..., 3]")

SIGNIFICANT = 0.05  # Choose your own!
corr_spi_sub_dt.pearson_r.where(corr_spi_sub_dt.pearson_p < 0.05).plot.imshow(robust=True,cmap = 'RdPu')

dims = ('latitude', 'longitude')
coords = {d: spi_sub_dt[d] for d in dims}
correlation_data = {
    name: xr.DataArray(data=np.ndarray([len(spi_sub_rc[d]) for d in dims]),
                       name=name, dims=dims)
    for name in 'pearson_r pearson_p spearman_r spearman_p'.split()
}
corr_spi_sub_rc = xr.Dataset(data_vars=correlation_data, coords=coords)
corr_spi_sub_rc

get_ipython().run_cell_magic('time', '', "# By looping, we make a list of lists of correlations\nlatout = []\nfor lat in anomalies_NDVI_rc.latitude:\n    lonout = []\n    latout.append(lonout)\n    for lon in anomalies_NDVI_rc.longitude:\n        NDVI = anomalies_NDVI_rc.sel(latitude=lat, longitude=lon)\n#         SPI = spi_1M_sub.sel(latitude=lat, longitude=lon)\n        SPI_1M = spi_sub_rc.sel(latitude=lat, longitude=lon)\n        mask = ~np.isinf(SPI_1M)\n        subset_NDVI= NDVI.where(mask, drop=True)\n        subset_SPI_1M= SPI_1M.where(mask, drop=True)\n        \n        val = pearsonr(subset_NDVI,subset_SPI_1M)\n        try:\n            # Spearman's R can fail for some values\n            val += spearmanr(NDVI,SPI_1M)\n        except ValueError:\n            val += (np.nan, np.nan)\n        lonout.append(val)\n# Then we convert the lists to an array\narr = np.array(latout)\n# And finally insert the pieces into our correlation dataset\ncorr_spi_sub_rc.pearson_r[:] = arr[..., 0]\ncorr_spi_sub_rc.pearson_p[:] = arr[..., 1]\ncorr_spi_sub_rc.spearman_r[:] = arr[..., 2]\ncorr_spi_sub_rc.spearman_p[:] = arr[..., 3]")

corr_spi_sub_rc

SIGNIFICANT = 0.05  # Choose your own!
corr_spi_sub_rc.pearson_r.where(corr_spi_sub_rc.pearson_p < 0.05).plot.imshow(robust=True,cmap = 'RdPu')

# outpath = '/g/data/oe9/project/team-drip/Spatial_temporal_correlation/AET_NDVI.nc'
# corr_1M_ET_NDVI.to_netcdf(outpath, mode = 'w')

import matplotlib.pyplot as plt
fname = '/g/data/oe9/project/team-drip/results/spi_ndvi_cor_dif.png'

SIGNIFICANT = 0.05  # Choose your own!
corr_diff_12M = corr_spi_sub_dt - corr_spi_sub_rc
fig = plt.figure(figsize=(12, 10))
dif = corr_diff_12M.pearson_r.where(corr_diff_12M.pearson_p < 0.05)
import shapefile   

shpFilePath = '/g/data/oe9/project/team-drip/MDB_shapefile/mdb_boundary/mdb_boundary.shp' 
listx=[]
listy=[]
test = shapefile.Reader(shpFilePath)
for sr in test.shapeRecords():
    for xNew,yNew in sr.shape.points:
        listx.append(xNew)
        listy.append(yNew)
plt.plot(listx,listy,color='black')


dif.plot(robust=True,cmap = 'bwr',vmin = -0.4, vmax = 0.4)
plt.xlabel('Longitude',fontsize=18)
plt.ylabel('Latitude',fontsize=18)
plt.title('Change in Correlation between SPI and NDVI (Before-After)', fontsize=18)
plt.grid(True)

fig.savefig(fname, dpi=600)

# import matplotlib.pyplot as plt
# fname = '/home/599/rg6346/EVI_test.png'
# fig = plt.figure()
# plt.plot(evi_ts.time,evi_ts,'g^-',ndvi_ts.time,ndvi_ts/2,'yo-',aet_ts.time,aet_ts/200,'b--')
# fig.savefig(fname, dpi=300)

# import fiona
import shapefile   

shpFilePath = '/g/data/oe9/project/team-drip/MDB_shapefile/mdb_boundary/mdb_boundary.shp' 
listx=[]
listy=[]
test = shapefile.Reader(shpFilePath)
for sr in test.shapeRecords():
    for xNew,yNew in sr.shape.points:
        listx.append(xNew)
        listy.append(yNew)
plt.plot(listx,listy)
plt.show()

# # Create a figure with several subplots - three columns
figure, ax_s = plt.subplots(nrows = 2,ncols = 2,figsize=(15, 15))
# plt.title('Correlation between NDVI and SPI')


corr_1M_ET_NDVI.pearson_r.where(corr_1M_ET_NDVI.pearson_p < 0.05).plot.imshow(ax = ax_s[0,0],
                                                              robust=True, cmap='GnBu', vmin=0.0, vmax=1.0)
ax_s[0,0].plot(listx,listy,color='black')
ax_s[0,0].set_title('NDVI vs ET')

cbr = figure.colorbar



