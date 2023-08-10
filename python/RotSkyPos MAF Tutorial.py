import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.sliceMetrics as sliceMetrics
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils

slicer = slicers.HealpixSlicer(nside=64)

metriclist = []
metriclist.append(metrics.MeanAngleMetric('rotSkyPos'))
metriclist.append(metrics.RmsAngleMetric('rotSkyPos'))
metriclist.append(metrics.FullRangeAngleMetric('rotSkyPos'))

outDir = 'rotSkyPos_Test'
sm = sliceMetrics.RunSliceMetric(outDir=outDir)
sm.setMetricsSlicerStackers(metriclist, slicer)

runName = 'enigma_1189'
opsdb = db.OpsimDatabase('sqlite:///' + runName + '_sqlite.db')

dbcols = sm.findReqCols()
print dbcols

sqlconstraint = "filter = 'r'"
simdata = utils.getSimData(opsdb, sqlconstraint, dbcols)
print simdata.dtype.names

sm.runSlices(simdata, simDataName=runName, sqlconstraint=sqlconstraint, metadata='r band')

for iid in sm.metricNames:
    print "%d || %s || %s || %.2f" %(iid, sm.metricNames[iid], sm.metadatas[iid], sm.metricValues[iid].compressed().mean())

sm.writeAll()

get_ipython().system('ls $outDir')

sm.plotAll()

iid = 1
print sm.metricNames[iid]
print sm.plotDicts[iid]

sm.plotDicts[iid]['colorMin'] = np.radians(30)
sm.plotDicts[iid]['colorMax'] = np.radians(90)
sm.plotDicts[iid]['xMin'] = np.radians(30)
sm.plotDicts[iid]['xMax'] = np.radians(90)
sm.plotMetric(iid)

summaryStats = []
summaryStats.append(metrics.MeanMetric('metricdata'))
summaryStats.append(metrics.MedianMetric('metricdata'))
summaryStats.append(metrics.RmsMetric('metricdata'))

for iid in sm.metricValues:
    for stat in summaryStats:
        val = sm.computeSummaryStatistics(iid, stat)
        print sm.metricNames[iid], '\t', stat.name, '\t', val

sqlcommand = 'select metrics.metricId, metricName, slicerName, metricMetadata, summaryName, summaryValue '
sqlcommand += 'from metrics, summaryStats where metrics.metricId = summaryStats.metricId'
resultsDb = utils.connectResultsDb(dbDir=outDir)
data = resultsDb.tables['metrics'].execute_arbitrary(sqlcommand)
print data

get_ipython().system('ls $outDir')

utils.writeConfigs(opsdb, outDir)
get_ipython().system('showMaf.py -d $outDir -p 8080')



