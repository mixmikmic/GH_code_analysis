from osgeo import gdal, gdal_array
import numpy as np

import glob
import os

# aggregation code is in cython. 

# If this has not already been translated to c and built then this must be done first 
# with setuptools. Change to the directory containing the .pxd file and run 
# python setup.py build_ext --inplace

# OR to avoid the separate compilation step we can use pyximport as shown here.
# Note though that using pyximport doesn't allow for the openmp 
# parallelisation flag. That's not required for the continuous aggregation library used here.
# import pyximport
# pyximport.install()

from Cython_Raster_Funcs.RasterAggregator_float import RasterAggregator_float

from General_Raster_Funcs.GeotransformCalcs import calcAggregatedProperties
from General_Raster_Funcs.RasterTiling import getTiles
from General_Raster_Funcs.TiffManagement import SaveLZWTiff

from math import floor
def continuousAggregationRunner(dataPaths):
    '''Run the aggregation code for each file specifed in dataPaths which should be a list of tiff files.
    
    The global notebook variables outDir, method, minMaxRangeSumOnly, itemsToSave, 
    (aggregationFactor OR resolution OR requiredXSize and requiredYSize), and tileSize
    should be set first, along with the fnGetter function for producing output filenames.
    '''
    for f in dataPaths:
        print f
        ds = gdal.Open(f)
        b = ds.GetRasterBand(1)
        ndv = b.GetNoDataValue()
        if ndv is None:
            print "no ndv"
            ndv = -9999
        inputGT = ds.GetGeoTransform()
        inputProj = ds.GetProjection()
        
        # not used, but for checking if required:
        nBytesRequired = ds.RasterXSize * ds.RasterYSize * gdalBytesPerPx[b.DataType]

        outGT, outShape = calcAggregatedProperties(method, (ds.RasterYSize, ds.RasterXSize), 
                                                   inputGT, aggregationFactor, 
                                                   (requiredYSize, requiredXSize), 
                                                   resolution)
        outYSize, outXSize = outShape    
        #outYSize = int(outYSize)
        #outXSize = int(outXSize)
        print outYSize, outXSize
        #assert False
        tiles = getTiles(ds.RasterXSize, ds.RasterYSize, tileSize)
        
        aggregator = RasterAggregator_float(ds.RasterXSize, ds.RasterYSize, 
                                            outXSize, outYSize,
                                            float(ndv), 
                                            minMaxRangeSumOnly)
        
        print "Running {0!s} tiles".format(len(tiles)),
        for tile in tiles:
            print ".",
            xoff = tile[0][0]
            yoff = tile[1][0]
            xsize = tile[0][1] - xoff
            ysize = tile[1][1] - yoff
            inArr = b.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float32)
    # SPECIFIC FOR Global Urban Footprint dataset: do a cheeky reclass from 255 to 1
            # inArr[inArr == 255] = 1
            aggregator.addTile(inArr, xoff, yoff)
        r = aggregator.GetResults()
        for i in itemstosave:
            fnOut = fnGetter(os.path.basename(f), i)
            print fnOut
            # the file-saving function will save to a tiff of datatype matching the array
            # it receives.
            if i in ['min','max','range']:
                # if the input was some integer type then save as this, even though the 
                # aggregation code always outputs float32
                nptype = gdal_array.GDALTypeCodeToNumericTypeCode(b.DataType)
                SaveLZWTiff(r[i].astype(nptype), ndv, outGT, inputProj, outDir, fnOut)
            elif i in ['mean','sd', 'sum']:
                # sum might be integer but potentially of larger type than the input, don't bother
                # dealing with conversion for now
                SaveLZWTiff(r[i], ndv, outGT, inputProj, outDir, fnOut)
            elif i in ['count']:
                SaveLZWTiff(r[i].astype(np.int32), ndv, outGT, inputProj, outDir, fnOut)
            else:
                assert False
                
# For reference:        
gdalBytesPerPx = {
0:1, # UNKNOWN
1:1, # GDT_Byte
2:2, # GDT_Uint16
3:2, # GDT_Int16
4:4, # GDT_UIint32
5:4, # GDT_Int32
6:4, # GDT_Float32
7:8  # GDT_Float64
}
        

# All of the items in this cell are required by the continuousAggregationRunner function:

# 1. Specify the method by which the aggregation will be described
#method = "size" # "factor" or "size" or "resolution"
method = "resolution"
# and whichever one of these is relevant
# if method = 'factor' then
aggregationFactor = 5
# OR if method = 'size' then
requiredXSize = 4320
requiredYSize = 2160
# OR if method = 'resolution' then specify a cell size or a 
# string from '1km', '5km', '10km'
#resolution = 0.008333333333333
resolution = '5km'

# 2. Should we calculate only min, max, range sum, (count)? 
# If that's all we need, then it's quicker to set this flag as 
# average / sd are more computationally demanding
minMaxRangeSumOnly = 0

# 3. itemstosave are the stats produced by the cython code.
# choices are count, max, mean, min, range, sum, sd.
# mean and sd are produced only if the minMaxRangeSumOnly flag is not set
#itemstosave = ["mean", "max", "min", "sum", "count"]
itemstosave = ["sum","count"]
# 4. specify the folder where the outputs should be saved
outDir = r"C:\temp\testagg"
outDir = r'\\map-fs1.ndph.ox.ac.uk\map_data\Population\IHME_Matched\IHME_Matched_Frankenpop_2016\02_Processing\06_IHME_Corrected_Grids\by_gender\5k'
# 5. specify a function called fnGetter, to get the output filename 
# (excluding folder), given an input filename and statistic type 
# (as in the itemstosave above)
fnGetter = lambda filename, stat:(os.path.splitext(os.path.basename(filename))[0]
                        + "_" + stat + "_5k.tif")
#fnGetter = lambda filename, stat:(os.path.basename(filename).split("_files")[0] + "_5km_" + 
#                                 stat + ".tif")
# 6. Specify a maximum tilesize for data to read in
# for the global 1km grids use 43200 to run untiled
tileSize = 43200

#dataPaths = [r'E:\Temp\water\occurrence.vrt']
#dataPaths = [r'C:\Users\zool1301.NDPH\Documents\Other_Data\GUF\GUF28\GUF28\000777777786422\0007files_aligned.tif']
dataPaths = glob.glob(r'C:\Users\zool1301.NDPH\Documents\Other_Data\GUF\GUF28\GUF28\**\*files_aligned.tif')
#dataPaths = [r'C:\Users\zool1301.NDPH\Documents\Other_Data\GUF\GUF28\GUF28\000777777786422\0007files_aligned.tif']
#dataPaths = [r'G:\GapfillingOutputs\EVI_Africa_1\A2000065_EVI_Filled_Data.tif']
dataPaths = glob.glob(r'\\map-fs1.ndph.ox.ac.uk\map_data\Population\IHME_Matched\IHME_Matched_Frankenpop_2016\02_Processing\06_IHME_Corrected_Grids\by_gender\*.tif')
dataPaths = glob.glob(r'E:\Temp\pop\ihme_figures\04_outputs\ihme_2016_version_po_at_risk\1k\*.tif')
outDir = r'E:\Temp\pop\ihme_figures\04_outputs\ihme_2016_version_po_at_risk\5k'

dataPaths

continuousAggregationRunner(dataPaths)

