from osgeo import gdal, gdal_array
import numpy as np
import rasterio
import glob
import os

from Cython_Raster_Funcs.CoastlineMatching import reallocateToUnmasked

from General_Raster_Funcs.TiffManagement import SaveLZWTiff

# files in true (accurate) coords (resolution = 0.008333333333333 or multiple thereof)
coastlineFile = r'G:\Supporting\CoastGlobal.tiff'
#coastlineFile = r'G:\Supporting\CoastGlobal_5k.tif'


inFiles = [r'C:\Temp\testagg\mos\guf_5km_mean.tif']
inFiles = glob.glob(r'C:\Temp\testagg\mos\guf_5km_*.tif')
outDir = r'C:\Temp\testagg\mos'

inFiles = glob.glob(r'C:\Temp\dataprep\pop_2016\Americas-POP-1KM\*.tif')
inFiles = glob.glob(r'C:\Temp\dataprep\pop_2016\frankenpop_2017_creation\*.tif')
#inFiles = glob.glob(r'C:\Temp\dataprep\pop_2016\GPWv4\*.tif')
outDir = r'C:\Temp\dataprep\pop_2016\Asia-POP-1KM'

fillLandWithZero = 0

clipZerosAtSea = 1

searchPixelRadius = 100

deleteUnmoveableData = 0

for inFileName in inFiles:
    print inFileName
    outDataFile = os.path.splitext(inFileName)[0] + ".MG_Matched.tif"
    outFailFile = os.path.splitext(inFileName)[0] + ".MG_Errors.tif"
    
    inDS = gdal.Open(inFileName)
    bData = inDS.GetRasterBand(1)
    ndvIn = bData.GetNoDataValue()
    if ndvIn:
        print "Nodata incoming is " + str(ndvIn)
    gtIn = inDS.GetGeoTransform()
    projIn = inDS.GetProjection()
    
    dTypeIn = bData.DataType
    incomingNPType = gdal_array.GDALTypeCodeToNumericTypeCode(dTypeIn)
    wasRecast = 0
    if not incomingNPType == np.float32 or incomingNPType == np.float64:
        print "Raster is not a float - recasting"
        print incomingNPType
       # assert False
        popData = bData.ReadAsArray().astype(np.float32)
        wasRecast = 1
    else:
        popData = bData.ReadAsArray()
    popDS = None
    
    landDS = gdal.Open(coastlineFile)
    b = landDS.GetRasterBand(1)
    ndvMask = b.GetNoDataValue()
    gtLand = landDS.GetGeoTransform()
    
    # Check that the resolutions are the same
    # (If the rounding issue has not first been corrected then the assertion will
    # fail: in this case comment them out and satisfy yourself first that the pixel 
    # coordinates do match i.e. that the rounding doesn't lead to > 0.5 cell error)
    assert round(gtIn[1], 12) == round(gtLand[1], 12) 
    assert round(gtIn[5], 12) == round(gtLand[5], 12)
    
    # the population dataset is not global; where does it sit in the global image?
    landOffsetW = int(round((gtIn[0] - gtLand[0]) / gtLand[1]))
    landOffsetN = int(round((gtIn[3]-gtLand[3]) / gtLand[5]))
    
    #print (landOffsetN, landOffsetW)
    landData= b.ReadAsArray(landOffsetW, landOffsetN, 
                            popData.shape[1], popData.shape[0])
    
    popData[popData == ndvIn] = -9999
    ndvIn = -9999
    failedLocs = reallocateToUnmasked(popData, landData, ndvIn, 
                                    fillLandWithZero = fillLandWithZero, 
                                    clipZerosAtSea = clipZerosAtSea, 
                                    searchPixelRadius = searchPixelRadius,
                                    deleteDespiteFailure = deleteUnmoveableData)
    #popData[popData > 1]=1
    incomingNPType = np.int16
    if wasRecast:
        SaveLZWTiff(popData.astype(incomingNPType), ndvIn, gtIn, projIn,
                   outDir, outDataFile)
    else:
        SaveLZWTiff(popData, ndvIn, gtIn, projIn, outDir, outDataFile)
    
    SaveLZWTiff(np.asarray(failedLocs), None, gtIn, projIn, 
                outDir, outFailFile)
    

gtIn

gtLand

popData.shape

#popDir = r'C:\Users\zool1301\Documents\Other_Data\Population\WorldPop\WholeContinentPop2010\WorldPop-Africa'
popDir = r'G:\DataPrep\population\GRUMP\tif'
#r'C:\Users\zool1301\Documents\Other_Data\Population\WorldPop\AgeStructuresAsia'
popFiles = glob.glob(os.path.join(popDir, "*.tif"))
for inFN in popFiles:
    outFN = os.path.join(popDir, 
                         #"MG_Matched", 
                         os.path.basename(os.path.splitext(inFN)[0])
                             +"_MG_Matched.tif"
                         )
    outFailFN = outFN.replace("_MG_Matched.tif", "_MG_Failures.tif")
    if os.path.exists(outFN):
        print "Already done "+inFN
        continue
    print inFN
    popDS = gdal.Open(inFN)
    b = popDS.GetRasterBand(1)
    ndvPop = b.GetNoDataValue()
    gtPop = popDS.GetGeoTransform()
    projPop = popDS.GetProjection()
    
    popData = b.ReadAsArray()
    popDS = None
    
    landDS = gdal.Open(ls_Accurate_1kFile)
    b = landDS.GetRasterBand(1)
    ndvMask = b.GetNoDataValue()
    gtLand = landDS.GetGeoTransform()
    
    # Check that the resolutions are the same
    # (If the rounding issue has not first been corrected then the assertion will
    # fail: in this case comment them out and satisfy yourself first that the pixel 
    # coordinates do match i.e. that the rounding doesn't lead to > 0.5 cell error)
    assert round(gtPop[1], 15) == round(gtLand[1], 15) 
    assert round(gtPop[5], 15) == round(gtLand[5], 15)
    
    # the population dataset is not global; where does it sit in the global image?
    landOffsetW = int(round((gtPop[0] - gtLand[0]) / gtLand[1]))
    landOffsetN = int(round((gtPop[3]-gtLand[3]) / gtLand[5]))
    
    #print (landOffsetN, landOffsetW)
    landData= b.ReadAsArray(landOffsetW, landOffsetN, popData.shape[1], popData.shape[0])
    
    failedLocs = reallocateToUnmasked(popData, landData, ndvPop)
    
    writeTiffFile(popData, outFN, gtPop, projPop, ndvPop)
    writeTiffFile(np.asarray(failedLocs), outFailFN, gtPop, projPop, None, gdal.GDT_Byte)
    



inDirStack = r'C:\Users\zool1301\Documents\Other_Data\Population\WorldPop\AgeStructures\AgeStructuresAfrica\MG_Matched'
fnFormat = 'ap{0}v4_A*_MG_Matched.tif'
for yr in ['00','05','10','15']:
    inPattern = fnFormat.format(yr)
    inStack = glob.glob(os.path.join(inDirStack, inPattern))
    first = True
    for inFN in inStack:
        ds = gdal.Open(inFN)
        b = ds.GetRasterBand(1)
        arr = b.ReadAsArray()
        if first:
            first = False
            ndvPop = b.GetNoDataValue()
            gtPop = ds.GetGeoTransform()
            projPop = ds.GetProjection()
            sumArr = arr
        else:
            assert ndvPop == b.GetNoDataValue()
            assert gtPop == ds.GetGeoTransform()
            assert projPop == ds.GetProjection()
            assert arr.shape == sumArr.shape
            sumArr[arr != ndvPop] += arr[arr != ndvPop]
    outFN = "Africa{0}_ManualTotal.tif".format(yr)
    writeTiffFile(sumArr, os.path.join(inDirStack,outFN), gtPop, projPop, ndvPop)
    

popDS = gdal.Open(inPopFile)
b = popDS.GetRasterBand(1)
ndvPop = b.GetNoDataValue()
gtPop = popDS.GetGeoTransform()
projPop = popDS.GetProjection()

#popData = b.ReadAsArray()
popOffsetN = int((50 - gtPop[3]) / gtLand[5])
popHeight = int((50 - -60) / gtPop[1])
popData = b.ReadAsArray(0, popOffsetN, 8640, popHeight)

gtPop

landDS = gdal.Open(ls_Accurate_5kFile)
b = landDS.GetRasterBand(1)
ndvMask = b.GetNoDataValue()
gtLand = landDS.GetGeoTransform()
gtLand

# Are the resolutions the same?
assert gtPop[1] == gtLand[1]
assert gtPop[5] == gtLand[5]

# the population dataset is not global; where does it sit in the global image?
landOffsetW = int(round((gtPop[0] - gtLand[0]) / gtLand[1]))
#landOffsetN = int((gtPop[3]-gtLand[3]) / gtLand[5])
landOffsetN = int(round((50-gtLand[3]) / gtLand[5]))
landOffsetN, landOffsetW

landHeight = (50 - -60) / gtLand[1]

landHeight

# read the required portion of the land data
landData= b.ReadAsArray(landOffsetW, landOffsetN, popData.shape[1], popData.shape[0])
#landData= b.ReadAsArray(landOffsetW, landOffsetN, 8640, 2640)

# check the totals match - do this before and after
np.logical_and(np.not_equal(popData,ndvPop), np.not_equal(popData,0)).sum()

# perform the reallocation
failedLocs = reallocateToUnmasked(popData, landData, ndvPop)

# check the totals match - do this before and after
np.logical_and(np.not_equal(popData,ndvPop), np.not_equal(popData,0)).sum()

# write the outputs
writeTiffFile(popData, outPopFile, gtPop, projPop, ndvPop)
writeTiffFile(failedLocs, outFailFile, gtPop, projPop, None, gdal.GDT_Byte)

pop2005File = r'C:\Users\zool1301\Documents\Other_Data\Population\GPWv3\Futures\GPWv3_FE_2005_MGMatched.tif'
pop2010File = r'C:\Users\zool1301\Documents\Other_Data\Population\GPWv3\Futures\GPWv3_FE_2010_MGMatched.tif'
pop2015File = r'C:\Users\zool1301\Documents\Other_Data\Population\GPWv3\Futures\GPWv3_FE_2015_MGMatched.tif'
d = gdal.Open(pop2005File)
pop2005 = d.GetRasterBand(1).ReadAsArray()
d = gdal.Open(pop2010File)
pop2010 = d.GetRasterBand(1).ReadAsArray()
d = gdal.Open(pop2015File)
pop2015 = d.GetRasterBand(1).ReadAsArray()

popDir = r'\\map-fs1.ndph.ox.ac.uk\map_data\mastergrids\Other_Global_Covariates\Population\Worldpop_GPWv4_Hybrid_201601'

pop2000File = os.path.join(popDir, 'Global_Pop_1km_Adj_MGMatched_2000_Hybrid.tif')
pop2005File = os.path.join(popDir, 'Global_Pop_1km_Adj_MGMatched_2005_Hybrid.tif')
pop2010File = os.path.join(popDir, 'Global_Pop_1km_Adj_MGMatched_2010_Hybrid.tif')
pop2015File = os.path.join(popDir, 'Global_Pop_1km_Adj_MGMatched_2015_Hybrid.tif')

d = gdal.Open(pop2000File)
pop2000 = d.GetRasterBand(1).ReadAsArray()
d = gdal.Open(pop2005File)
pop2005 = d.GetRasterBand(1).ReadAsArray()
d = gdal.Open(pop2010File)
pop2010 = d.GetRasterBand(1).ReadAsArray()
d = gdal.Open(pop2015File)
pop2015 = d.GetRasterBand(1).ReadAsArray()

globalGT = d.GetGeoTransform()
globalProj = d.GetProjection()
ndv = d.GetRasterBand(1).GetNoDataValue()

pop2015.shape

stack = np.empty(shape=(16,17400,43200), dtype=np.float32)

del(stack)

stack[0] = np.copy(pop2000)
stack[5] = np.copy(pop2005)
stack[10] = np.copy(pop2010)
stack[15] = np.copy(pop2015)
gotYrs = [2000,2005,2010,2015]
baseYr = 2000
popDiff = stack[5] - stack[0]
for i in range(1,16):
    #print i
    yr = i + baseYr
    if yr in gotYrs:
        print yr
        prevYr = yr
        nextYr = gotYrs[gotYrs.index(yr)+1]
        popDiff = stack
    else:
        thisYrOffset = yr-prevYr
        

outDir = r'E:\Temp\pop'
fnTemplate = 'Global_Pop_1km_Adj_MGMatched_{0!s}-Interp_Hybrid.tif'

os.path.j

end = pop2005
start = pop2000
base = 2000

popDiffPerYr = (end - start) / 5.0
popShape = popDiffPerYr.shape
grubby = np.logical_or(start==ndv, end==ndv)

for i in range (base+1,base+5):
    offset = i - base
    print i
    thisYr = (offset*popDiffPerYr) + start
    thisYr[grubby] = ndv
    outDrv = gdal.GetDriverByName('GTiff')
    outPopFile = os.path.join(outDir, fnTemplate.format(i) )
    dataRaster = outDrv.Create(outPopFile, popShape[1], popShape[0], 1, gdal.GDT_Float32,
                                           ["COMPRESS=LZW", "TILED=YES", "SPARSE_OK=TRUE", "BIGTIFF=YES"])
   # failRaster = outDrv.Create(outFailFN, popShape[1], popShape[0], 1, gdal.GDT_Byte,
   #                                        ["COMPRESS=LZW", "TILED=YES", "SPARSE_OK=TRUE", "BIGTIFF=YES"])

    dataRaster.SetGeoTransform(globalGT)
    dataRaster.SetProjection(globalProj)
    #failRaster.SetGeoTransform(globalGT)
    #failRaster.SetProjection(globalProj)

    bnd = dataRaster.GetRasterBand(1)
    assert bnd is not None
    bnd.SetNoDataValue(ndv)
    bnd.WriteArray(thisYr)
    bnd = None
    dataRaster = None

ap2kFiles = glob.glob(r'\\map-fs1.ndph.ox.ac.uk\map_data\mastergrids\Other_Global_Covariates\Population\WorldPop\AgeStructures\Africa\2000\*.tif')
outDir = r'C:\Users\zool1301.NDPH\Documents\Dial-A-Map\pop-1990-invention'

import re
# we'll be creating three age-bin datasets for this one worldpop year
ap0005 = None
ap0515 = None
ap1599 = None
everData = None
for f in ap2kFiles:
    fnParts = os.path.basename(f).split('_')
    maybeAge = fnParts[1]
    if re.match('A\d', maybeAge):
        print maybeAge
        d = gdal.Open(f)
        b = d.GetRasterBand(1)
        arr = b.ReadAsArray()
        ndv = b.GetNoDataValue()
        startage = maybeAge[1:3]
        if everData is None:
            everData = np.zeros(arr.shape,np.bool)
            gt = d.GetGeoTransform()
            proj = d.GetProjection()
        everData = np.logical_or(everData, arr != ndv)
        
        if startage == '00':
            if ap0005 is None:
                ap0005 = np.zeros_like(arr)
            ap0005[arr != ndv] += arr[arr != ndv]
        if startage == '05':
            if ap0515 is None:
                ap0515 = np.zeros_like(arr)
            ap0515[arr != ndv] += arr[arr != ndv]
        else:
            if ap1599 is None:
                ap1599 = np.zeros_like(arr)
            ap1599[arr != ndv] += arr[arr != ndv]

# calculate the proportions            
apTot = ap0005+ap0515+ap1599
# will give runtime warning due to divide by zero if there's permanent nodata anywhere
ap0005_prop = ap0005 / apTot
ap0515_prop = ap0515 / apTot
ap1599_prop = ap1599 / apTot
# make sure those places are set to nodata 
ap0005_prop[everData == False] = ndv
ap0515_prop[everData == False] = ndv
ap1599_prop[everData == False] = ndv

writeTiffFile(ap0005_prop, os.path.join(outDir, 'ap2000_0005_prop.tif'), gt, proj, ndv)
writeTiffFile(ap0515_prop, os.path.join(outDir, 'ap2000_0515_prop.tif'), gt, proj, ndv)
writeTiffFile(ap1599_prop, os.path.join(outDir, 'ap2000_1599_prop.tif'), gt, proj, ndv)

writeTiffFile(ap0005, os.path.join(outDir, r'ap2000_0005.tif') , gt, proj, ndv)
writeTiffFile(ap0515, os.path.join(outDir, r'ap2000_0515.tif'), gt, proj, ndv)
writeTiffFile(ap1599, os.path.join(outDir, r'ap2000_1599.tif'), gt, proj, ndv)

writeTiffFile(apTot, os.path.join(outDir, r'ap2000_Tot.tif'), gt, proj, ndv)

dan0005_fn = r'\\map-fs1.ndph.ox.ac.uk\map_data\cubes\5km\AfriPop\population_surfaces_for_Pf_incidence\00-05\2000_00-05.tif'
dan0515_fn = r'\\map-fs1.ndph.ox.ac.uk\map_data\cubes\5km\AfriPop\population_surfaces_for_Pf_incidence\05-15\2000_05-15.tif'
dan15pl_fn = r'\\map-fs1.ndph.ox.ac.uk\map_data\cubes\5km\AfriPop\population_surfaces_for_Pf_incidence\15+\2000_15+.tif'

d = gdal.Open(dan0005_fn)
b = d.GetRasterBand(1)
gt = d.GetGeoTransform()
proj = d.GetProjection()
ndv = b.GetNoDataValue()

dan0005 = b.ReadAsArray()

d = gdal.Open(dan0515_fn)
b = d.GetRasterBand(1)
dan0515 = b.ReadAsArray()

d = gdal.Open(dan15pl_fn)
b = d.GetRasterBand(1)
dan15pl = b.ReadAsArray()

danTot = dan0005+dan0515+dan15pl
dan0005_prop = dan0005 / danTot
dan0515_prop = dan0515 / danTot
dan15pl_prop = dan15pl / danTot

writeTiffFile(dan0005_prop, r'C:\Users\zool1301.NDPH\Documents\Dial-A-Map\pop-1990-invention\dan2000_0005_prop.tif', gt, proj, ndv)
writeTiffFile(dan0515_prop, r'C:\Users\zool1301.NDPH\Documents\Dial-A-Map\pop-1990-invention\dan2000_0515_prop.tif', gt, proj, ndv)
writeTiffFile(dan15pl_prop, r'C:\Users\zool1301.NDPH\Documents\Dial-A-Map\pop-1990-invention\dan2000_15pl_prop.tif', gt, proj, ndv)

writeTiffFile(danTot, r'C:\Users\zool1301.NDPH\Documents\Dial-A-Map\pop-1990-invention\dan2000_total.tif', gt, proj, ndv)

