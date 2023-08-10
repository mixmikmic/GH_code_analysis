import numpy as np
import xarray as xr
import rasterio
get_ipython().magic('matplotlib inline')
from matplotlib.pyplot import *
from glob import glob
import os
import datetime

from rasterio_to_xarray import rasterio_to_xarray, xarray_to_rasterio, xarray_to_rasterio_by_band

def maiac_file_to_da(filename):
    da = rasterio_to_xarray(filename)
    
    da.values[da.values == -28672] = np.nan
    da.values[da.values == 0] = np.nan
    
    #da.values = da.values.astype(np.float64)
    
    time_str = os.path.basename(filename)[17:28]
    #print(time_str)
    time_obj = datetime.datetime.strptime(time_str, '%Y%j%H%M')
    da.coords['time'] = time_obj
    
    return da

files = sorted(glob('ForVFPoC/2003/Projected/*_projPM25.tif'))

list_of_das = map(maiac_file_to_da, files)

res = xr.concat(list_of_das, 'time')

newres = res.isel(time=np.argsort(res.time))

r = newres.resample('D', dim='time', how='max')

r = r.dropna(dim='time', how='all')

overall_mean = r.mean(dim='time', keep_attrs=True)

import geopandas as gpd
import pandas as pd

import rasterstats

gdf = gpd.GeoDataFrame.from_file('nuts3_OSGB.json')

aff = rasterio.Affine.from_gdal(*r.attrs['affine'])

res = rasterstats.zonal_stats(gdf, overall_mean.values, affine=aff)

gdf.join(pd.DataFrame(res))



arr = r.isel(time=[0,1,2,3,4]).values



rasterstats.zonal_stats('nuts3_OSGB.json', overall_mean.values, affine=aff, nodata=np.nan, raster_out=True)

