get_ipython().magic('matplotlib inline')

import fiona
import rasterio
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

indices = (5, 6, 7, 8, 10)

source = ('/Users/cmutel/Box Sync/LC-Impact (Chris Mutel)/12-water consumption/spatial layers EQ'
          '/shapefiles/CF_CORE_plants_noVS_and_animals_inclVS_noCpA_Option3_SW_PDF_perm3.tif')

with rasterio.open(source) as src:
    source_array = src.read(1)

def feature_count(index):
    fp = "kmeans.{}.gpkg".format(index)
    with fiona.open(fp) as src:
        l = len(src)
    return l
        
for index in indices:
    print(index, feature_count(index))

with rasterio.open("water_eq_core.10.tif") as src:
    array = src.read(1, masked=True)

mask = ~array.mask | (array.data > 0)

array[mask].shape, array.shape, source_array[mask].shape, source_array.shape

np.histogram(array[mask])

np.unique(array[mask]).shape

def plot_index(index):
    with rasterio.open("water_eq_core.{}.tif".format(index)) as src:
        array = src.read(1, masked=True)
        
    mask = ~array.mask | (array.data != 0)
    
    df = pd.DataFrame({
        'original': source_array[mask],
        'clustered': array.data[mask]
    })
    sns.jointplot('original', 'clustered', data=df, size=8, kind='scatter')

def plot_index2(index):
    with rasterio.open("water_eq_core.{}.tif".format(index)) as src:
        array = src.read(1, masked=True)
        
    mask = ~array.mask | (array.data > 0)
    
    df = pd.DataFrame({
        'original': np.log(source_array[mask].astype(np.float64)),
        'clustered': np.log(array.data[mask].astype(np.float64))
    })
    
    sns.jointplot('original', 'clustered', data=df, size=8, kind='hex')

plot_index2(10)

error = np.abs(array[mask] - source_array[mask]) / source_array[mask]

error_mask = error < 10
print((~error_mask).sum(), error.shape)

sns.distplot(error[error_mask])



