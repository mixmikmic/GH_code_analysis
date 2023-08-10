import glob
import rasterio as rio
import numpy as np
from osgeo import gdal
import os
from collections import defaultdict

get_ipython().magic('load_ext cython')

inFilePattern = r"G:\MCD43B4\MCD43B4_Indices\EVI\*.tif"

# generate this in excel with =CONCATENATE(DAYNUM,":",MONTH(DAYNUM),", ")
daymonths = {1:1, 9:1, 17:1, 25:1, 33:2, 41:2, 49:2, 57:2, 65:3, 73:3, 81:3, 89:3, 97:4, 
             105:4, 113:4, 121:4, 129:5, 137:5, 145:5, 153:6, 161:6, 169:6, 177:6, 185:7, 
             193:7, 201:7, 209:7, 217:8, 225:8, 233:8, 241:8, 249:9, 257:9, 265:9, 273:9, 
             281:10, 289:10, 297:10, 305:10, 313:11, 321:11, 329:11, 337:12, 345:12, 353:12, 
             361:12}
monthDays = defaultdict(list)
for d,m in daymonths.iteritems():
    monthDays[m].append(d)
    
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
    
    
    

monthDays

# the rasters have tilesize 256 (or a multiple thereof)
# read the rasters in slices that align to this for most efficient access
edges = np.arange(0,21600,7168)
slices = zip(edges[:-1], edges[1:])
slices[-1] = (slices[-1][0],21600)

exampleFile = r'E:\MOD11A2_DiurnalDiffs_Output\LST_Diurnal_Diffs\Output_8day\A2000065_LST_DiurnalDifference.tif'
ds = gdal.Open(exampleFile)
b = ds.GetRasterBand(1)
b.GetNoDataValue()
globalGT = ds.GetGeoTransform()
globalProj = ds.GetProjection()
ds = None

outDrv = gdal.GetDriverByName('GTiff')

for b in range(1,14):
    meansRaster.GetRasterBand(b).SetNoDataValue(-9999)
    sdRaster.GetRasterBand(b).SetNoDataValue(-9999)
    countRaster.GetRasterBand(b).SetNoDataValue(-9999)

# load in 
for t,b in slices:
    sliceHeight = b - t
    #monthPixelCounts = {i+1:np.zeros((sliceHeight,43200),dtype='byte') for i in range(0,12)}
    #monthPixelOldM = {i+1:np.zeros((sliceHeight,43200), dtype='float64') for i in range(0,12)}
    #monthPixelNewM = {i+1:np.zeros((sliceHeight,43200), dtype='float64') for i in range(0,12)}
    #monthPixelOldS = {i+1:np.zeros((sliceHeight,43200), dtype='float64') for i in range(0,12)}
    #monthPixelNewS = {i+1:np.zeros((sliceHeight,43200), dtype='float64') for i in range(0,12)}
    m_nTotal = np.zeros((sliceHeight,43200),dtype='byte')
    m_oldMTotal = np.zeros((sliceHeight,43200), dtype='float64')
    m_newMTotal = np.zeros((sliceHeight,43200), dtype='float64')
    m_oldSTotal = np.zeros((sliceHeight,43200), dtype='float64')
    m_newSTotal = np.zeros((sliceHeight,43200), dtype='float64')
    
    for month, days in monthDays.iteritems():
        print "Month "+str(month)
        m_n = np.zeros((sliceHeight,43200),dtype='byte')
        m_oldM = np.zeros((sliceHeight,43200), dtype='float64')
        m_newM = np.zeros((sliceHeight,43200), dtype='float64')
        m_oldS = np.zeros((sliceHeight,43200), dtype='float64')
        m_newS = np.zeros((sliceHeight,43200), dtype='float64')
        for day in days:
            for dayfile in dayfiles[day]:
                print dayfile
                ds = gdal.Open(dayfile)
                b = ds.GetRasterBand(1)
                ndv = b.GetNoDataValue()
                
                data = b.ReadAsArray(0, t, None, sliceHeight)
                
                # do nothing with tracking data if no data
                # operate only on locations where we have a data value
                dataMask = ne.evaluate("data != ndv")
                                
                # robust (against FP errors) streaming mean and sd method from 
                # http://www.johndcook.com/blog/standard_deviation/
                # calculate a monthly and, separately, an overall mean 
                # (could do this after from the monthlies, but, hey).
                m_n[dataMask] += 1
                m_nTotal[dataMask] += 1
                
                # month pixels that have not previously had a value
                updateLocs = ne.evaluate("dataMask & (m_n==1)")
                m_oldM[updateLocs] = data[updateLocs]
                m_newM[updateLocs] = data[updateLocs]
                #m_oldS[m_n==1] = 0.0 # it is initialised to zero so no need
                
                #overall pixels that have not previously had a value
                updateLocs = ne.evaluate("dataMask & (m_nTotal==1)")
                m_oldMTotal[updateLocs] = data[updateLocs]
                m_newMTotal[updateLocs] = data[updateLocs]
                
                # month pixels for which this is the second or subsequent value
                updateLocs = ne.evaluate("dataMask & (m_n>1)")
                m_newM[updateLocs] = (m_oldM[updateLocs] + 
                            ((data[updateLocs] - m_oldM[updateLocs]) / m_n[updateLocs]))
                m_newS[updateLocs] = (m_oldS[updateLocs] + 
                            ((data[updateLocs] - m_oldM[updateLocs]) * 
                             (data[updateLocs] - m_newM[updateLocs])
                            ))
                m_oldM[updateLocs] = m_newM[updateLocs]
                m_oldS[updateLocs] = m_newS[updateLocs]
                
                #overall pixels for which this is the second or subsequent value
                updateLocs = ne.evaluate("dataMask & (m_nTotal>1)")
                m_newMTotal[updateLocs] = (m_oldMTotal[updateLocs] + 
                            ((data[updateLocs] - m_oldMTotal[updateLocs]) / m_nTotal[updateLocs]))
                m_newSTotal[updateLocs] = (m_oldSTotal[updateLocs] + 
                            ((data[updateLocs] - m_oldMTotal[updateLocs]) * 
                             (data[updateLocs] - m_newMTotal[updateLocs])
                            ))
                m_oldMTotal[updateLocs] = m_newMTotal[updateLocs]
                m_oldSTotal[updateLocs] = m_newSTotal[updateLocs]

        countRaster.GetRasterBand(month).WriteArray(m_n, 0, t)
        meansRaster.GetRasterBand(month).WriteArray(m_newM, 0, t)
        m_Var = np.zeros((sliceHeight,43200), dtype='float64')
        m_Var[m_n > 1] = m_newS[m_n > 1] / (m_n[m_n > 1] - 1)
        sdRaster.GetRasterBand(month).WriteArray(np.sqrt(m_Var), 0, t)
    countRaster.GetRasterBand(13).WriteArray(m_nTotal, 0, t)
    meansRaster.GetRasterBand(13).WriteArray(m_newMTotal, 0, t)
    m_VarTotal = np.zeros((sliceHeight, 43200), dtype='float64')
    m_VarTotal[m_nTotal > 1] = m_newSTotal[m_nTotal > 1] / (m_nTotal[m_nTotal > 1] - 1)
    sdRaster.GetRasterBand(13).WriteArray(np.sqrt(m_VarTotal), 0, t)
                

get_ipython().run_cell_magic('cython', '--compile-args=/openmp --link-args=/openmp --force', '# the above flags are needed to get ipython to use openmp, see\n# https://github.com/ipython/ipython/issues/2669/\ncimport cython\ncimport openmp\nimport numpy as np\nfrom osgeo import gdal \nfrom libc.math cimport sqrt\nimport glob\nimport os\nfrom collections import defaultdict\nfrom cython.parallel cimport parallel, prange\nimport rasterio as rio\n\n@cython.boundscheck(False)\n@cython.cdivision(True)    \ncpdef calcStats(Py_ssize_t width, Py_ssize_t height, Py_ssize_t desiredSliceHeight, char* metric, char* baseDir):\n    \'\'\'\n    Calculates mean and standard deviations for all files in a given directory, using hard-coded filename patterns\n    \'\'\'\n    cdef:\n        Py_ssize_t y, x, yShape, xShape, t, b, sliceHeight\n        float[:,::1] data\n        double[:,::1] mth_oldM, mth_newM, mth_oldS, mth_newS\n        double[:,::1] tot_oldM_Days, tot_newM_Days, tot_oldS_Days, tot_newS_Days\n        #double[:,::1] tot_oldM_Months, tot_newM_Months, tot_oldS_Months, tot_newS_Months\n        short[:,::1] mth_n, tot_n_Days, tot_n_Months\n        double value\n        double ndv, test_ndv\n        double variance, sd\n     \n    # generate this in excel with =CONCATENATE(DAYNUM,":",MONTH(DAYNUM),", ")\n    daymonths = {1:1, 9:1, 17:1, 25:1, 33:2, 41:2, 49:2, 57:2, 65:3, 73:3, 81:3, 89:3, \n                 97:4, 105:4, 113:4, 121:4, 129:5, 137:5, 145:5, 153:6, 161:6, 169:6, \n                 177:6, 185:7, 193:7, 201:7, 209:7, 217:8, 225:8, 233:8, 241:8, 249:9, \n                 257:9, 265:9, 273:9, 281:10, 289:10, 297:10, 305:10, 313:11, 321:11, \n                 329:11, 337:12, 345:12, 353:12, 361:12}\n\n    monthDays = defaultdict(list)\n    for d,m in daymonths.iteritems():\n        monthDays[m].append(d)\n\n    years = defaultdict(int)\n    days = defaultdict(int)\n    dayfiles = defaultdict(list)\n    \n    dataFilePattern = r"{0}\\*_{1}.tif"\n    lastFN = ""\n    for fn in glob.glob(dataFilePattern.format(baseDir,metric)):\n        datestr = os.path.basename(fn).split(\'_\')[0][1:]\n        yr = int(datestr[:4])\n        years[yr] +=1\n        day = int(datestr[4:])\n        days[day] +=1\n        month = daymonths[day]\n        dayfiles[day].append(fn)\n        lastFN = fn\n    \n    ds = gdal.Open(lastFN)\n    bnd = ds.GetRasterBand(1)\n    globalGT = ds.GetGeoTransform()\n    globalProj = ds.GetProjection()\n    outDrv = gdal.GetDriverByName(\'GTiff\')\n    ndv = bnd.GetNoDataValue()\n    \n    # save outputs to an uncompressed raster as the compression\n    # is fairly ineffective when not writing whole band at once, \n    # generating outputs of aroung 55Gb per file which is larger than uncompressed!\n    # Instead save to this temp file and then translate them all to compressed format with\n    #\n    # for /F "usebackq tokens=1 delims=." %f in (`dir /B *.tif`) do (\n    # gdal_translate  -of GTiff -co "COMPRESS=LZW" -co "TILED=YES" -co "PREDICTOR=2" \n    # -co "SPARSE_OK=TRUE" -co "BIGTIFF=YES" -co "INTERLAVE=BAND" %f.tif G:\\StatsBackup\\%f.tif\n    # gdal_translate -b 13 -of GTiff -co "COMPRESS=LZW" -co "TILED=YES" -co "PREDICTOR=2" \n    # -co "SPARSE_OK=TRUE" %f.tif G:\\StatsBackup\\%f_Overall.tif)\n    meansRaster = outDrv.Create(r\'C:\\Users\\zool1301\\AppData\\Local\\Temp\\{}_Monthly_Means.tif\'.format(metric),\n                            43200,21600,13,gdal.GDT_Float32,\n                            ["TILED=YES","SPARSE_OK=TRUE","BIGTIFF=YES","INTERLEAVE=BAND"])\n    sdRaster = outDrv.Create(r\'C:\\Users\\zool1301\\AppData\\Local\\Temp\\{}_Monthly_SDs.tif\'.format(metric),\n                         43200,21600,13,gdal.GDT_Float32,\n                         ["TILED=YES","SPARSE_OK=TRUE","BIGTIFF=YES","INTERLEAVE=BAND"])\n    countRaster = outDrv.Create(r\'C:\\Users\\zool1301\\AppData\\Local\\Temp\\{}_Monthly_Counts.tif\'.format(metric),\n                            43200,21600,13,gdal.GDT_Int16,\n                            ["TILED=YES","SPARSE_OK=TRUE","BIGTIFF=YES","INTERLEAVE=BAND"])\n    meansRaster.SetGeoTransform(globalGT)\n    meansRaster.SetProjection(globalProj)\n    sdRaster.SetGeoTransform(globalGT)\n    sdRaster.SetProjection(globalProj)\n    countRaster.SetGeoTransform(globalGT)\n    countRaster.SetProjection(globalProj)\n    \n    assert desiredSliceHeight <= height\n    \n    #setup arrays\n    # the rasters have tilesize 256\n    # read the rasters in slices that align to this\n    slices = [(0,height)]\n    if desiredSliceHeight < height:\n        edges = np.arange(0,21600,desiredSliceHeight)\n        slices = zip(edges[:-1], edges[1:])\n        slices[-1] = (slices[-1][0],height)\n   \n    # go through everything as many times as we need\n    for t,b in slices:\n        print "Slice "+str(t)+ " - "+str(b)\n        sliceHeight = b - t\n        \n        # initialise arrays to track the overall stats\n        tot_n_Days = np.zeros((sliceHeight,width),dtype=\'Int16\')\n        tot_oldM_Days = np.zeros((sliceHeight,width),dtype=\'float64\')\n        tot_newM_Days = np.zeros((sliceHeight,width),dtype=\'float64\')\n        tot_oldS_Days = np.zeros((sliceHeight,width),dtype=\'float64\')\n        tot_newS_Days = np.zeros((sliceHeight,width),dtype=\'float64\')\n        tot_oldM_Days[:]=ndv\n        tot_newM_Days[:]=ndv\n        tot_oldS_Days[:]=ndv\n        tot_newS_Days[:]=ndv\n        \n        # initialise arrays to track the stats of the months\n        # - this needs more than 2 slices if running globally using float64 on my 64gb machine.\n        # so doing this separately afterwards instead.\n        #tot_n_Months = np.zeros((sliceHeight,width),dtype=\'Int16\')\n        #tot_oldM_Months = np.zeros((sliceHeight,width),dtype=\'float64\')\n        #tot_newM_Months = np.zeros((sliceHeight,width),dtype=\'float64\')\n        #tot_oldS_Months = np.zeros((sliceHeight,width),dtype=\'float64\')\n        #tot_newS_Months = np.zeros((sliceHeight,width),dtype=\'float64\')\n        #tot_oldM_Months[:]=ndv\n        #tot_newM_Months[:]=ndv\n        #tot_oldS_Months[:]=ndv\n        #tot_newS_Months[:]=ndv\n        \n        for month,days in monthDays.iteritems():\n            print "Month "+str(month)\n            \n            #initialise arrays to track this month\n            mth_n = np.zeros((sliceHeight,width),dtype=\'Int16\')\n            mth_oldM = np.zeros((sliceHeight,width),dtype=\'float64\')\n            mth_newM = np.zeros((sliceHeight,width),dtype=\'float64\')\n            mth_oldS = np.zeros((sliceHeight,width),dtype=\'float64\')\n            mth_newS = np.zeros((sliceHeight,width),dtype=\'float64\')\n            # don\'t have zero but no data instead because zero is valid. Counts remain at zero.\n            mth_oldM[:]=ndv\n            mth_newM[:]=ndv\n            mth_oldS[:]=ndv\n            mth_newS[:]=ndv\n                     \n            for day in days:\n                for dayfile in dayfiles[day]:\n                    print dayfile\n                    ds = gdal.Open(dayfile)\n                    band = ds.GetRasterBand(1)\n                    test_ndv = band.GetNoDataValue()\n                    if test_ndv != ndv:\n                        print("Warning! File {0} has NDV of {1!s} which is different from set NDV of {2!s}. \\\n                              NoData may not be correctly handled.".format(dayfile,test_ndv,ndv) ) \n                    data = band.ReadAsArray(0, t, None, sliceHeight)\n                    with nogil, cython.wraparound(False), parallel(num_threads=6):\n                        # Main loop to iterate over the pixels. No python objects in loop so can release\n                        # GIL and thus multithread it.\n                        # With 6 threads and slice of 43200 * ~11000 it takes around 1.5s to calc the 464M pixels\n                        # whereas it takes around 9s to read them in from the compressed tiff!\n                        # So the overall process now is massively dominated by the actual reading of the file.\n                        for y in prange (sliceHeight, schedule=\'static\'):\n                            value = test_ndv\n                            x=-1\n                      \n                            for x in range (0, width):\n                                value = data[y,x]\n\n                                if value == test_ndv:\n                                    continue\n\n                                tot_n_Days[y,x] += 1\n                                mth_n[y,x] += 1\n\n                                if tot_n_Days[y,x] == 1:\n                                    tot_oldM_Days[y, x] = value\n                                    tot_newM_Days[y, x] = value\n                                    tot_oldS_Days[y, x] = 0\n                                    tot_newS_Days[y, x] = 0\n                                else:\n                                    tot_newM_Days[y,x] = (tot_oldM_Days[y,x] + \n                                                     ((value - tot_oldM_Days[y,x]) / tot_n_Days[y,x]))\n                                    tot_newS_Days[y,x] = (tot_oldS_Days[y,x] + \n                                                     ((value - tot_oldM_Days[y,x]) *\n                                                      (value - tot_newM_Days[y,x])\n                                                      ))\n                                    tot_oldM_Days[y,x] = tot_newM_Days[y,x]\n                                    tot_oldS_Days[y,x] = tot_newS_Days[y,x]#bollox!\n                                    \n                                if mth_n[y,x] == 1:\n                                    mth_oldM[y,x] = value\n                                    mth_newM[y,x] = value\n                                    mth_oldS[y,x] = 0\n                                    mth_newS[y,x] = 0\n                                else:\n                                    #update monthly stats\n                                    mth_newM[y,x] = (mth_oldM[y,x] + \n                                                     ((value - mth_oldM[y,x]) / mth_n[y,x]))\n                                    mth_newS[y,x] = (mth_oldS[y,x] + \n                                                     ((value - mth_oldM[y,x]) *\n                                                      (value - mth_newM[y,x])\n                                                      ))\n                                    mth_oldM[y,x] = mth_newM[y,x]\n                                    mth_oldS[y,x] = mth_newS[y,x]#bollox... this was setting S to M!!\n                      \n            #month done\n            #Write the monthly data\n            countRaster.GetRasterBand(month).WriteArray(np.asarray(mth_n), 0, t)\n            meansRaster.GetRasterBand(month).WriteArray(np.asarray(mth_newM).astype(\'float32\'), 0, t)\n            meansRaster.GetRasterBand(month).SetNoDataValue(ndv)\n            with nogil, cython.wraparound(False):\n                for y in prange (sliceHeight, schedule=\'static\', num_threads=6):\n                    for x in range (0, width):\n                        if mth_n[y,x] <= 1:\n                            continue\n                        variance = mth_newS[y,x] / (mth_n[y,x] - 1)\n                        mth_newS[y,x] = sqrt(variance)\n            sdRaster.GetRasterBand(month).WriteArray(np.asarray(mth_newS).astype(\'float32\'), 0, t)\n            sdRaster.GetRasterBand(month).SetNoDataValue(ndv)\n\n            # Calculate the mean of the months (the balanced mean)\n            # broken out to separate function for mem reasons\n            #with nogil, cython.wraparound(False), parallel(num_threads=6):\n            #    for y in prange (sliceHeight, schedule=\'static\'):\n            #        value = ndv\n            #        x=-1\n            #        for x in range (0, width):\n            #            value = mth_newM[y,x]\n            #            if value == ndv:\n            #                continue\n            #            tot_n_Months[y,x] += 1\n            #            \n            #            if tot_n_Months[y,x] == 1:\n            #                tot_oldM_Months[y, x] = value\n            #                tot_newM_Months[y, x] = value\n            #                tot_oldS_Months[y, x] = 0\n            #                tot_newS_Months[y, x] = 0\n            #                continue\n\n            #            tot_newM_Months[y,x] = (tot_oldM_Months[y,x] + \n            #                             ((value - tot_oldM_Months[y,x]) / tot_n_Months[y,x]))\n            #            tot_newS_Months[y,x] = (tot_oldS_Months[y,x] + \n            #                             ((value - tot_oldM_Months[y,x]) *\n            #                              (value - tot_newM_Months[y,x])\n            #                              ))\n            #            tot_oldM_Months[y,x] = tot_newM_Months[y,x]\n            #            tot_oldS_Months[y,x] = tot_newS_Months[y,x]#bollox!\n\n        # all months done\n        meansRaster.GetRasterBand(13).SetNoDataValue(ndv)\n        sdRaster.GetRasterBand(13).SetNoDataValue(ndv)\n        countRaster.GetRasterBand(13).WriteArray(np.asarray(tot_n_Days), 0, t)\n        meansRaster.GetRasterBand(13).WriteArray(np.asarray(tot_newM_Days).astype(\'float32\'), 0, t)\n        \n        #meansRaster.GetRasterBand(14).SetNoDataValue(ndv)\n        #sdRaster.GetRasterBand(14).SetNoDataValue(ndv)\n        #countRaster.GetRasterBand(14).WriteArray(np.asarray(tot_n_Months), 0, t)\n        #meansRaster.GetRasterBand(14).WriteArray(np.asarray(tot_newM_Months).astype(\'float32\'), 0, t)\n        \n        with nogil, cython.wraparound(False):\n            for y in prange (sliceHeight, schedule=\'static\', num_threads=6):\n                for x in range (0, width):\n                    if tot_n_Days[y,x] <= 1:\n                        continue\n                    variance = tot_newS_Days[y,x] / (tot_n_Days[y,x] - 1)\n                    tot_newS_Days[y,x] = sqrt(variance)\n            #for y in prange (sliceHeight, schedule=\'static\', num_threads=6):\n            #    for x in range (0, width):\n            #        if tot_n_Months[y,x] <= 1:\n            #            continue\n            #        variance = tot_newS_Months[y,x] / (tot_n_Months[y,x] - 1)\n            #        tot_newS_Months[y,x] = sqrt(variance)\n        sdRaster.GetRasterBand(13).WriteArray(np.asarray(tot_newS_Days).astype(\'float32\'), 0, t)\n        #sdRaster.GetRasterBand(14).WriteArray(np.asarray(tot_newS_Months).astype(\'float32\'), 0, t)\n        print ("slice done")\n    meansRaster = None\n    countRaster = None\n    sdRaster = None')

get_ipython().run_cell_magic('cython', '--compile-args=/openmp --link-args=/openmp --force', '# the above flags are needed to get ipython to use openmp, see\n# https://github.com/ipython/ipython/issues/2669/\ncimport cython\ncimport openmp\nimport numpy as np\nfrom osgeo import gdal \nfrom cython.parallel cimport parallel, prange\n\n\n@cython.boundscheck(False)\n@cython.cdivision(True)    \ncpdef calcBalancedMeans(metric):\n    \'\'\' Calculates balanced mean i.e. mean of monthly means and associated standard deviation\n    \'\'\'\n    cdef:\n        float[:,::1] inputmeans\n        short[:,::1] inputcounts\n\n        short[:,::1] tot_n_Months\n        double[:,::1] tot_oldM_Months, tot_newM_Months, tot_oldS_Months, tot_newS_Months\n        Py_ssize_t width, height\n        \n        float[:,::1] means\n        short[:,::1] counts\n        float value, floatNdv\n       # short shortNdv\n        Py_ssize_t x, y\n\n    meansRasterStacked = gdal.Open(r\'C:\\Users\\zool1301\\AppData\\Local\\Temp\\{}_Monthly_Means.tif\'.format(metric))\n    b = meansRasterStacked.GetRasterBand(1)\n    floatNdv = b.GetNoDataValue()\n    \n    width = meansRasterStacked.RasterXSize\n    height = meansRasterStacked.RasterYSize\n    \n    # work with whole bands at once, we can _just_ afford this on 64Gb machine\n    # and it saves having to recompress\n    tot_n_Months = np.zeros((height,width),dtype=\'Int16\')\n    tot_oldM_Months = np.zeros((height,width),dtype=\'float64\')\n    tot_newM_Months = np.zeros((height,width),dtype=\'float64\')\n    tot_oldS_Months = np.zeros((height,width),dtype=\'float64\')\n    tot_newS_Months = np.zeros((height,width),dtype=\'float64\')\n    tot_oldM_Months[:]=floatNdv\n    tot_newM_Months[:]=floatNdv\n    tot_oldS_Months[:]=floatNdv\n    tot_newS_Months[:]=floatNdv\n    \n    for i in range (1,13):\n        print ("Adding band {0!s} to mean".format(i))\n        meanBnd = meansRasterStacked.GetRasterBand(i)\n        means = meanBnd.ReadAsArray()\n        with nogil, cython.wraparound(False), parallel(num_threads=6):\n            for y in prange (height, schedule=\'static\'):\n                value = floatNdv\n                x=-1\n                for x in range (0, width):\n                    value = means[y,x]\n                    if value == floatNdv:\n                        continue\n                    tot_n_Months[y,x] += 1\n\n                    if tot_n_Months[y,x] == 1:\n                        tot_oldM_Months[y, x] = value\n                        tot_newM_Months[y, x] = value\n                        tot_oldS_Months[y, x] = 0\n                        tot_newS_Months[y, x] = 0\n                        continue\n\n                    tot_newM_Months[y,x] = (tot_oldM_Months[y,x] + \n                                     ((value - tot_oldM_Months[y,x]) / tot_n_Months[y,x]))\n                    tot_newS_Months[y,x] = (tot_oldS_Months[y,x] + \n                                     ((value - tot_oldM_Months[y,x]) *\n                                      (value - tot_newM_Months[y,x])\n                                      ))\n                    tot_oldM_Months[y,x] = tot_newM_Months[y,x]\n                    tot_oldS_Months[y,x] = tot_newS_Months[y,x]\n    \n    outDrv = gdal.GetDriverByName(\'GTiff\')\n    globalGT = meansRasterStacked.GetGeoTransform()\n    globalProj = meansRasterStacked.GetProjection()\n    meansRasterOut = outDrv.Create(r\'C:\\Users\\zool1301\\AppData\\Local\\Temp\\{}_Mean_From_Monthly.tif\'.format(metric),\n                            width,height,1,gdal.GDT_Float32,\n                            ["TILED=YES","SPARSE_OK=TRUE","BIGTIFF=YES"])\n    sdRasterOut = outDrv.Create(r\'C:\\Users\\zool1301\\AppData\\Local\\Temp\\{}_SD_From_Monthly.tif\'.format(metric),\n                         43200,21600,1,gdal.GDT_Float32,\n                         ["TILED=YES","SPARSE_OK=TRUE","BIGTIFF=YES"])\n    countRasterOut = outDrv.Create(r\'C:\\Users\\zool1301\\AppData\\Local\\Temp\\{}_Count_Of_Months.tif\'.format(metric),\n                            43200,21600,1,gdal.GDT_Int16,\n                            ["TILED=YES","SPARSE_OK=TRUE","BIGTIFF=YES"])\n    meansRasterOut.SetGeoTransform(globalGT)\n    meansRasterOut.SetProjection(globalProj)\n    sdRasterOut.SetGeoTransform(globalGT)\n    sdRasterOut.SetProjection(globalProj)\n    countRasterOut.SetGeoTransform(globalGT)\n    countRasterOut.SetProjection(globalProj)\n    \n    b = meansRasterOut.GetRasterBand(1)\n    b.SetNoDataValue(floatNdv)\n    b.WriteArray(np.asarray(tot_newM_Months).astype(\'float32\'))\n    b.FlushCache()\n    \n    b = sdRasterOut.GetRasterBand(1)\n    b.SetNoDataValue(floatNdv)\n    b.WriteArray(np.asarray(tot_newS_Months).astype(\'float32\'))\n    b.FlushCache()\n    \n    b = countRasterOut.GetRasterBand(1)\n    #b.SetNoDataValue(shortNdv)\n    b.WriteArray(np.asarray(tot_n_Months))\n    b.FlushCache()\n    \n              ')

# pick a slice size that aligns to 256 pixel tiff tile size for optimal access.
# 5376 gives 4 slices and seems much quicker overall than 10752 (2 slices)
calcStats(43200,21600,5376,'EVI',r'E:\MCD43B4\MCD43B4_Indices')

calcBalancedMeans('EVI_WinterCut_OcclusionMasked_Nbr2')

calcStats(43200,21600,5376,'Day',r'F:\MOD11A2\MOD11A2_Data')

calcBalancedMeans('Day')


calcStats(43200,21600,5376,'Night',r'F:\MOD11A2\MOD11A2_Data')

calcBalancedMeans('Night')

calcStats(43200,21600,5376,'TCW',r'E:\MCD43B4\MCD43B4_Indices')

calcBalancedMeans('TCW')

calcStats(43200,21600,10752,'TCB',r'E:\MCD43B4\MCD43B4_Indices')

calcBalancedMeans('TCB')

calcStats(43200,21600,5376,'LST_DiurnalDifference',r'E:\MOD11A2_DiurnalDiffs_Output\LST_Diurnal_Diffs\Output_8day')

calcBalancedMeans('LST_DiurnalDifference')

