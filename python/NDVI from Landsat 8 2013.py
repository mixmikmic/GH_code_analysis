get_ipython().run_line_magic('matplotlib', 'inline')

from osgeo import gdal
import rasterio
from matplotlib import pyplot
import numpy
import pygeoprocessing


L8_RED = '../data/landsat/LC08_L1TP_042034_20130605_20170310_01_T1_B4_120x120.TIF'
L8_NIR = '../data/landsat/LC08_L1TP_042034_20130605_20170310_01_T1_B5_120x120.TIF'
L8_QA = '../data/landsat/LC08_L1TP_042034_20130605_20170310_01_T1_BQA_120x120.TIF'

def plot(array):
    """Plot a numpy array with an NDVI colormap."""
    pyplot.imshow(array, cmap='RdYlGn')
    pyplot.colorbar()

dataset = gdal.Open(L8_RED)
red = dataset.ReadAsArray()

with rasterio.open(L8_RED) as rio_dataset:
    print rio_dataset.read()

plot(red)

nir = gdal.Open(L8_NIR).ReadAsArray()

def ndvi(red, nir):
    red = red.astype(numpy.float)
    nir = nir.astype(numpy.float)
    return (nir - red)/(nir + red)

red_2013 = gdal.Open(L8_RED).ReadAsArray()
nir_2013 = gdal.Open(L8_NIR).ReadAsArray()

calculated_ndvi = ndvi(red_2013, nir_2013)
plot(calculated_ndvi)

driver = gdal.GetDriverByName('GTiff')
new_dataset = driver.Create(
    'ndvi.tif',
    dataset.RasterXSize,
    dataset.RasterYSize,
    1,
    gdal.GDT_Float32
)
new_dataset.SetProjection(dataset.GetProjection())
new_dataset.SetGeoTransform(dataset.GetGeoTransform())

band = new_dataset.GetRasterBand(1)
band.WriteArray(calculated_ndvi)
band = None
new_dataset = None

get_ipython().system('/opt/anaconda/envs/rasterenv/bin/gdalinfo ndvi.tif')



