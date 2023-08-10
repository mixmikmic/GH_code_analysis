import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# import seaborn

get_ipython().magic('matplotlib inline')
# seaborn.set_style('dark')
from scipy import stats

from scipy.stats import linregress, pearsonr, spearmanr


spi_12M= xr.open_dataarray('/g/data/oe9/project/team-drip/Rainfall/SPI_awap/SPI_12M_masked.nc')
spi_12M_sub=spi_12M.isel(time=range(1,204))
spi_12M_sub

coarse_NDVI= xr.open_dataarray('/g/data/oe9/project/team-drip/resampled_NDVI/coarse_NDVI.nc')
coarse_NDVI

climatology = coarse_NDVI.groupby('time.month').mean('time')

anomalies_NDVI = coarse_NDVI.groupby('time.month') - climatology

anomalies_NDVI

# Start by setting up a new dataset, with empty arrays along latitude and longitude
dims = ('latitude', 'longitude')
coords = {d: spi_12M_sub[d] for d in dims}
correlation_data = {
    name: xr.DataArray(data=np.ndarray([len(spi_12M[d]) for d in dims]),
                       name=name, dims=dims)
    for name in 'pearson_r pearson_p spearman_r spearman_p'.split()
}
corr_12M = xr.Dataset(data_vars=correlation_data, coords=coords)
corr_12M


get_ipython().run_cell_magic('time', '', "# By looping, we make a list of lists of correlations\nlatout = []\nfor lat in anomalies_NDVI.latitude:\n    lonout = []\n    latout.append(lonout)\n    for lon in anomalies_NDVI.longitude:\n        NDVI = anomalies_NDVI.sel(latitude=lat, longitude=lon)\n        SPI = spi_12M_sub.sel(latitude=lat, longitude=lon)\n        mask = ~np.isinf(SPI)\n        subset_NDVI= NDVI.where(mask, drop=True)\n        subset_SPI= SPI.where(mask, drop=True)\n        \n        val = pearsonr(subset_NDVI,subset_SPI)\n        try:\n            # Spearman's R can fail for some values\n            val += spearmanr(subset_NDVI,subset_SPI)\n        except ValueError:\n            val += (np.nan, np.nan)\n        lonout.append(val)\n# Then we convert the lists to an array\narr = np.array(latout)\n# And finally insert the pieces into our correlation dataset\ncorr_12M.pearson_r[:] = arr[..., 0]\ncorr_12M.pearson_p[:] = arr[..., 1]\ncorr_12M.spearman_r[:] = arr[..., 2]\ncorr_12M.spearman_p[:] = arr[..., 3]")

SIGNIFICANT = 0.05  # Choose your own!
corr_12M.pearson_r.sel(latitude=slice(-30.1, -31),longitude=slice(148.9,150.6)).where(corr_12M.pearson_p < SIGNIFICANT).plot.imshow(robust=True)

corr_12M.pearson_r.plot.imshow(robust=True)

# SAVE corelation matrix
path = '/g/data/oe9/project/team-drip/Spatial_temporal_correlation/SPI_NDVI/NDVI_SPI12M_Correlation.nc'
corr_12M.to_netcdf(path,mode = 'w')



