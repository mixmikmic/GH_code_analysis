get_ipython().run_line_magic('timeit', 'data = [x * x for x in range(1000)]')

get_ipython().run_cell_magic('timeit', '', 'data = []\nfor x in range(1000):\n    data.append(x * x)')

image_path = "D:/Fastest_image/LC81390452014295LGN00_B4.TIF" ### set the path to your image choice

## method 1 - opencv
import cv2
import numpy as np
# Import GDAL method 2
from osgeo import gdal
## import method 3 - skimage
from skimage import io
## import method 4 - rasterio
##import rasterio



get_ipython().run_cell_magic('timeit', '', '### Open CV approach\n\nimg = cv2.imread(image_path, -1) ### add your image here\nprint img.shape')

get_ipython().run_cell_magic('timeit', '', '## GDAL approach\n\nraster_ds = gdal.Open(image_path, gdal.GA_ReadOnly)\nimage_gdal = raster_ds.GetRasterBand(1).ReadAsArray()\nprint image_gdal.shape')

get_ipython().run_cell_magic('timeit', '', '## skimage\n\nim = io.imread(image_path)\nprint im.shape')

get_ipython().run_cell_magic('timeit', '', 'with rasterio.open(image_path) as r:\n    arr = r.read()  # read raster\n    print(arr.shape)')



