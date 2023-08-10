import glob
import numpy as np
from osgeo import gdal
import os
from collections import defaultdict

from General_Raster_Funcs.RasterTiling import getTiles
from General_Raster_Funcs.TiffManagement import *

from MODIS_Raster_Funcs.SynopticData import MonthlyStatCalculator

inFilePattern = r"F:\MOD11A2_Input_Mosaics\Day\*.tif"

tileDir = r"C:\temp\test\tiles"
outDir = r"C:\temp\test\merged"
what = "TestDA"

# Specify the height of each tile - depends on available memory.
# The algorithm needs around idealSlice * fullWidth * 80 bytes of RAM
# so with global 1k images (43200px wide), a slice of 7168 needs 
# around 25Gb RAM.
# The rasters have tilesize 256 (or a multiple thereof) so pick a size
# that is a multiple of this where possible for most efficient access
idealSlice = 7168

# alter to suit the images
fullWidth = 43200
fullHeight = 21600

# alter to whatever you want
outNdv = -9999


# build a dictionary mapping day of year to month of year, only required for the 
# day numbers that the 8-daily MODIS data occurs on
# generate this in excel with =CONCATENATE(DAYNUM,":",MONTH(DAYNUM),", ")
daymonths = {1:1, 9:1, 17:1, 25:1, 33:2, 41:2, 49:2, 57:2, 65:3, 73:3, 81:3, 89:3, 97:4, 
             105:4, 113:4, 121:4, 129:5, 137:5, 145:5, 153:6, 161:6, 169:6, 177:6, 185:7, 
             193:7, 201:7, 209:7, 217:8, 225:8, 233:8, 241:8, 249:9, 257:9, 265:9, 273:9, 
             281:10, 289:10, 297:10, 305:10, 313:11, 321:11, 329:11, 337:12, 345:12, 353:12, 
             361:12}
# swap to build list of days for each month
monthDays = defaultdict(list)
for d,m in daymonths.iteritems():
    monthDays[m].append(d)
    

# build a list of MODIS files available for each day-of-year, based on the 
# year / julian day that's encoded in the filenames such as "A2015009_LST_Day.tif"
years = defaultdict(int)
days = defaultdict(int)
dayfiles = defaultdict(list)
for fn in glob.glob(inFilePattern):
    datestr = os.path.basename(fn).split('_')[0][1:]
    yr = int(datestr[:4])
    years[yr] +=1
    day = int(datestr[4:])
    days[day] +=1
    month = daymonths[day]
    dayfiles[day].append(fn)

    

globalGT = None
globalProj = None
stats = ['Count', 'Mean', 'SD']
# work out the tiles we'll work in. We'll work with full-width slices for 
# now. 
slices = sorted(list(set([s[1] for s in getTiles(fullWidth, fullHeight, idealSlice)])))

slices

fnGetter = lambda what, when, stat, where:(
    str(what) + "_" + str(when) + "_" + str(stat)
    + "_" + str(where)+ ".tif")

def sliceRunner(top, bottom, width, outputNDV):
    assert (isinstance(bottom,int) and isinstance(top,int)
        and bottom > top)
    
    if not monthDays or not dayfiles or not fnGetter:
        print "Notebook globals monthDays, dayfiles, and fnGetter must be defined first"
        return False
    sliceHeight = bottom - top
    statsCalculator = MonthlyStatCalculator(sliceHeight, width, outputNDV)
    sliceGT = None
    sliceProj = None
    print str((top,bottom))
    for month, days in monthDays.iteritems():
        # for each calendar day of this synoptic month 
        print "\tMonth "+str(month)
        for day in days:
            # for each file on this calendar day (i.e. one per year)
            print"\t\tDay "+str(day)
            for dayfile in dayfiles[day]:
                # add slice
                data, myGT, myProj, thisNdv = ReadAOI_PixelLims(dayfile, None, (top, bottom))
                if sliceGT is None:
                    sliceGT = myGT
                    sliceProj = myProj
                else:
                    assert sliceGT == myGT
                    assert sliceProj == myProj
                # add the data to the running calculator
                statsCalculator.addFile(data, month, thisNdv)
        # get and save the results for this synoptic month
        monthResults = statsCalculator.emitMonth()
        SaveLZWTiff(monthResults['count'], outNdv, sliceGT, sliceProj, tileDir,
                   fnGetter(what, "M" + str(month).zfill(2), "Count", top))
        SaveLZWTiff(monthResults['mean'], outNdv, sliceGT, sliceProj, tileDir,
                   fnGetter(what, "M" + str(month).zfill(2), "Mean", top))
        SaveLZWTiff(monthResults['sd'], outNdv, sliceGT, sliceProj, tileDir,
                   fnGetter(what, "M" + str(month).zfill(2), "SD", top))
    statsCalculator = None
    
    # get and save the overall synoptic result
    overallResults = statsCalculator.emitTotal()
    SaveLZWTiff(overallResults['count'], outNdv, sliceGT, sliceProj, tileDir,
        fnGetter(what, "Overall", "Count", t))
    SaveLZWTiff(overallResults['mean'], outNdv, sliceGT, sliceProj, tileDir,
        fnGetter(what, "Overall", "Mean", t))
    SaveLZWTiff(overallResults['sd'], outNdv, sliceGT, sliceProj, tileDir,
        fnGetter(what, "Overall", "SD", t))
    return True
        

for t,b in slices[1]:
    sliceRunner(t, b, fullWidth, outNdv)
        

import subprocess
vrtBuilder = "gdalbuildvrt {0} {1}"
transBuilder = "gdal_translate -of GTiff -co COMPRESS=LZW "+    "-co PREDICTOR=2 -co TILED=YES -co SPARSE_OK=TRUE -co BIGTIFF=YES "+    "--config GDAL_CACHEMAX 8000 {0} {1}"
ovBuilder = "gdaladdo -ro --config COMPRESS_OVERVIEW LZW --config USE_RRD NO " +        "--config TILED YES {0} 2 4 8 16 32 64 128 256 --config GDAL_CACHEMAX 8000"
statBuilder = "gdalinfo -stats {0} >nul"    

vrts = []
tifs = []
for stat in stats:
    for month in sorted(monthDays.keys()):
        tiffWildCard = fnGetter(what, 'M'+str(month).zfill(2), stat, "*")
        sliceTiffs = os.path.join(tileDir, tiffWildCard)
        vrtName = "Month_" + str(month).zfill(2) + "_" + stat + ".vrt"
        vrtFile = os.path.join(outDir, vrtName)
        vrtCommand = vrtBuilder.format(vrtFile, 
                                      sliceTiffs)
        print vrtCommand
        vrts.append(vrtFile)
        subprocess.call(vrtCommand)
    tiffWildCard = fnGetter(what, "Overall", stat, "*")
    sliceTiffs = os.path.join(tileDir, tiffWildCard)
    vrtName = "Overall_" + stat + ".vrt"
    vrtFile = os.path.join(outDir, vrtName)
    vrtCommand = vrtBuilder.format(vrtFile, 
                                      sliceTiffs)
    print vrtCommand
    vrts.append(vrtFile)
    subprocess.call(vrtCommand)
    
for vrt in vrts:
    tif = vrt.replace('vrt', 'tif')
    transCommand = transBuilder.format(vrt, tif)
    print transCommand
    tifs.append(tif)
    subprocess.call(transCommand)
    
for tif in tifs:
    ovCommand = ovBuilder.format(tif)
    statCommand = statBuilder.format(tif)
    print ovCommand
    subprocess.call(ovCommand)
    print statCommand
    subprocess.call(statCommand)


# assuming we've run the code above and the output files from before are in the list called tifs

fileListMonths = [t for tif in tifs if t.startswith('Month') and t.endswith('Mean')]
assert len(fileListMonths) == 12
for (top, bottom) in slices:
    sliceHeight = bottom - top
    statsCalculator = MonthlyStatCalculator(sliceHeight, width, outputNDV)
    sliceGT = None
    sliceProj = None
    print str((top, bottom)) 
    for monthfile in fileListMonths:
        data, myGT, myProj, thisNdv = ReadAOI_PixelLims(monthfile, None, (top,bottom))
        if sliceGT is None:
            sliceGT = myGT
            sliceProj = myProj
        else:
            assert sliceGT == myGT
            assert sliceProj == myProj
        # calculate the mean of the months, use a fixed value for the "month" as we're not 
        # wanting monthly output from the calculator this time
        statsCalculator.addFile(data, 1, thisNdv)
    balancedRes = statsCalculator.emitTotal()
    SaveLZWTiff(balancedRes,['count'], outNdv, sliceGT, sliceProj, outDir,
               fnGetter(what, "Count_Of_Months", "", top))
    SaveLZWTiff(balancedRes,['mean'], outNdv, sliceGT, sliceProj, outDir,
               fnGetter(what, "Mean_Of_Months", "", top))
    SaveLZWTiff(balancedRes,['sd'], outNdv, sliceGT, sliceProj, outDir,
               fnGetter(what, "SD_Of_Months", "", top))
    statsCalculator = None



