import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn

get_ipython().magic('matplotlib inline')
seaborn.set_style('dark')
from scipy import stats

from scipy.stats import linregress, pearsonr, spearmanr


spi_1M= xr.open_dataarray('/g/data/oe9/project/team-drip/Rainfall/SPI_awap/SPI_1M_masked.nc')
spi_1M_sub=spi_1M.isel(time=range(1,204))
spi_1M_sub

coarse_NDVI= xr.open_dataarray('/g/data/oe9/project/team-drip/resampled_NDVI/coarse_NDVI.nc')
coarse_NDVI

climatology = coarse_NDVI.groupby('time.month').mean('time')

anomalies_NDVI = coarse_NDVI.groupby('time.month') - climatology

anomalies_NDVI

# Start by setting up a new dataset, with empty arrays along latitude and longitude
dims = ('latitude', 'longitude')
coords = {d: spi_1M_sub[d] for d in dims}
correlation_data = {
    name: xr.DataArray(data=np.ndarray([len(spi_1M[d]) for d in dims]),
                       name=name, dims=dims)
    for name in 'pearson_r pearson_p spearman_r spearman_p'.split()
}
corr_1M = xr.Dataset(data_vars=correlation_data, coords=coords)
corr_1M


get_ipython().run_cell_magic('time', '', "# By looping, we make a list of lists of correlations\nlatout = []\nfor lat in anomalies_NDVI.latitude:\n    lonout = []\n    latout.append(lonout)\n    for lon in anomalies_NDVI.longitude:\n        NDVI = anomalies_NDVI.sel(latitude=lat, longitude=lon)\n        SPI = spi_1M_sub.sel(latitude=lat, longitude=lon)\n        mask = ~np.isinf(SPI)\n        subset_NDVI= NDVI.where(mask, drop=True)\n        subset_SPI= SPI.where(mask, drop=True)\n        \n        val = pearsonr(subset_NDVI,subset_SPI)\n        try:\n            # Spearman's R can fail for some values\n            val += spearmanr(subset_NDVI,subset_SPI)\n        except ValueError:\n            val += (np.nan, np.nan)\n        lonout.append(val)\n# Then we convert the lists to an array\narr = np.array(latout)\n# And finally insert the pieces into our correlation dataset\ncorr_1M.pearson_r[:] = arr[..., 0]\ncorr_1M.pearson_p[:] = arr[..., 1]\ncorr_1M.spearman_r[:] = arr[..., 2]\ncorr_1M.spearman_p[:] = arr[..., 3]")

corr_1M

SIGNIFICANT = 0.05  # Choose your own!
corr_1M.pearson_r.where(corr_1M.pearson_p < 0.05).plot.imshow(robust=True)

corr_1M.pearson_r.plot.imshow(robust=True)

# SAVE corelation matrix
path = '/g/data/oe9/project/team-drip/Spatial_temporal_correlation/SPI_NDVI/NDVI_SPI1M_Correlation.nc'
corr_1M.to_netcdf(path,mode = 'w')



