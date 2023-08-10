import requests
import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.features import shapes
from rasterio.transform import *
from rasterio.plot import show
from rasterio.mask import geometry_mask, mask
from rasterio.warp import reproject
from shapely.geometry import mapping, shape, Polygon, Point
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from datetime import timedelta
from xml.dom import minidom
from scipy.ndimage import zoom
from skimage import img_as_ubyte
from skimage.exposure import equalize_adapthist as _hist
import seaborn as sns
import gc

import numpy as np
import os

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
plt.rcParams['figure.dpi'] = 70
plt.rcParams['figure.figsize'] = (27, 9)
plt.rcParams['font.size'] = 12

YEAR = 2017
asodb = "../data/aso-urls.csv"

asourls = pd.read_csv(asodb, parse_dates=[0]).set_index('date')
asourls = asourls[asourls.index.year == YEAR]
display(asourls)
image = asourls.iloc[1]
asoImage = rio.open(image.url)
image_crs = ccrs.epsg(asoImage.crs['init'].split(":")[1])

print(image)

plt.imshow(asoImage.read(1).astype('float32'), cmap="binary_r", vmax=1, vmin=0)
plt.title("ASO Capture: {:%d %b %Y} (EPSG:{})".format(image.name, image_crs.epsg_code))
plt.colorbar()

get_ipython().system('ls ../images/ASO/')

satimage = rio.open("../images/ASO/0_20170404_180056_0e19.tif")

plt.imshow(satimage.read(4), cmap='binary')

bl = (satimage.bounds.left, satimage.bounds.bottom)
tl = (satimage.bounds.left, satimage.bounds.top)
br = (satimage.bounds.right, satimage.bounds.bottom)
tr = (satimage.bounds.right, satimage.bounds.top)
bounds = Polygon([bl, tl, tr, br])

maskedaso, maskedtransform = mask(asoImage, [mapping(bounds)])

ax = plt.axes(projection=ccrs.PlateCarree())
plt.imshow(asoImage.read(1).astype('float32'), cmap='binary', transform=ccrs.epsg('32611'))
#plt.imshow(maskedaso.data.squeeze(), alpha=0.6)

satimage.shape

maskedaso.squeeze().shape

out = zoom(maskedaso.squeeze(), 16.61)

out.shape

show(out)



