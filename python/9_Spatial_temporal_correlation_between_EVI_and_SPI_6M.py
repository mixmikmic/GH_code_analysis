import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn

get_ipython().magic('matplotlib inline')
seaborn.set_style('dark')
from scipy import stats

from scipy.stats import linregress, pearsonr, spearmanr


spi_6M= xr.open_dataarray('/g/data/oe9/project/team-drip/Rainfall/SPI_awap/SPI_6M_masked.nc')
spi_6M_sub=spi_6M.isel(time=range(1,204))
spi_6M_sub

coarse_EVI= xr.open_dataarray('/g/data/oe9/project/team-drip/resampled_NDVI/coarse_EVI.nc')
coarse_EVI

climatology = coarse_EVI.groupby('time.month').mean('time')

anomalies_EVI = coarse_EVI.groupby('time.month') - climatology

anomalies_EVI.isel(time=range(0,2)).plot.imshow(col='time')

coarse_EVI.isel(time=range(0,2)).plot.imshow(col='time')

# Start by setting up a new dataset, with empty arrays along latitude and longitude
dims = ('latitude', 'longitude')
coords = {d: spi_6M_sub[d] for d in dims}
correlation_data = {
    name: xr.DataArray(data=np.ndarray([len(spi_6M[d]) for d in dims]),
                       name=name, dims=dims)
    for name in 'pearson_r pearson_p spearman_r spearman_p'.split()
}
corr_6M = xr.Dataset(data_vars=correlation_data, coords=coords)
corr_6M


get_ipython().run_cell_magic('time', '', "# By looping, we make a list of lists of correlations\nlatout = []\nfor lat in anomalies_EVI.latitude:\n    lonout = []\n    latout.append(lonout)\n    for lon in anomalies_EVI.longitude:\n        EVI = anomalies_EVI.sel(latitude=lat, longitude=lon)\n        SPI = spi_6M_sub.sel(latitude=lat, longitude=lon)\n        mask = ~np.isinf(SPI)\n        subset_EVI= EVI.where(mask, drop=True)\n        subset_SPI= SPI.where(mask, drop=True)\n        \n        val = pearsonr(subset_EVI,subset_SPI)\n        try:\n            # Spearman's R can fail for some values\n            val += spearmanr(subset_EVI,subset_SPI)\n        except ValueError:\n            val += (np.nan, np.nan)\n        lonout.append(val)\n# Then we convert the lists to an array\narr = np.array(latout)\n# And finally insert the pieces into our correlation dataset\ncorr_6M.pearson_r[:] = arr[..., 0]\ncorr_6M.pearson_p[:] = arr[..., 1]\ncorr_6M.spearman_r[:] = arr[..., 2]\ncorr_6M.spearman_p[:] = arr[..., 3]")

# SAVE corelation matrix
path = '/g/data/oe9/project/team-drip/Spatial_temporal_correlation/SPI_EVI/EVI_SPI6M_Correlation.nc'
corr_6M.to_netcdf(path,mode = 'w')

SIGNIFICANT = 0.05  # Choose your own!
corr_6M.pearson_r.where(corr_6M.pearson_p < 0.05).plot.imshow(robust=True)



