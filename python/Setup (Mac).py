import gdal, gdalconst

dataset = gdal.Open('rasters/MUL-PanSharpen_AOI_3_Paris_img741.tif', gdalconst.GA_ReadOnly)

tif_transform = dataset.GetGeoTransform()
tif_transform_inv = gdal.InvGeoTransform(tif_transform)

ulx, xres, xskew, uly, yskew, yres = tif_transform
lrx = ulx + (dataset.RasterXSize * xres)
lry = uly + (dataset.RasterYSize * yres)

bbox = (ulx, uly, lrx, lry)
projection = dataset.GetProjectionRef()

import geopandas as gpd

buildings = gpd.read_file('vectors/nature.geojson')
buildings.unary_union

