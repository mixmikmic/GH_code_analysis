import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plotters

#runs = {'ops2_1098':'Pairs','ops2_1093':'No Pairs'}
runs = {'ops2_1098':'ops2_1098 (pairs)','ops2_1093':'ops2_1093 (no pairs)'}

for runName in runs:
    # Set up the database connection
    opsdb = db.OpsimDatabase(runName+'_sqlite.db')
    md = runs[runName]
    outDir = 'intraDay'
    resultsDb = db.ResultsDb(outDir=outDir)
    plotList = [plotters.HealpixSkyMap(), plotters.HealpixHistogram()]
    slicer=slicers.HealpixSlicer()
    metric = metrics.IntraNightGapsMetric()
    metric2 = metrics.InterNightGapsMetric()
    plotDict = {'cmap':'jet', 'colorMin':0,'colorMax':0.5, 'xMin':0,'xMax':1.2}
    plotDict2 = {'cmap':'jet', 'colorMin':0,'colorMax':40, 'xMin':0,'xMax':40}
    plotDict3 = {'cmap':'jet', 'colorMin':0,'colorMax':40, 'xMin':0,'xMax':40}
    sql = 'filter = "r"'
    bundleList = []
    #bundleList.append(metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, 
    #                                             plotFuncs=plotList, metadata=md))
    #bundleList.append(metricBundles.MetricBundle(metric2,slicer,sql, plotDict=plotDict2, plotFuncs=plotList))

    slicer=slicers.HealpixSlicer(lonCol='ditheredRA', latCol='ditheredDec')
    bundleList.append(metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, 
                                                 metadata=md, plotFuncs=plotList))
    bundleList.append(metricBundles.MetricBundle(metric2,slicer,sql, plotDict=plotDict2, 
                                                 metadata=md, plotFuncs=plotList))

    bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
    bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)

for runName in runs:
    # Set up the database connection
    opsdb = db.OpsimDatabase(runName+'_sqlite.db')
    md = runs[runName]
    outDir = 'intraDay'
    resultsDb = db.ResultsDb(outDir=outDir)
    plotList = [plotters.HealpixSkyMap(), plotters.HealpixHistogram()]
    slicer=slicers.HealpixSlicer()
    metric = metrics.IntraNightGapsMetric()
    metric2 = metrics.InterNightGapsMetric()
    plotDict = {'cmap':'jet', 'colorMin':0,'colorMax':0.5, 'xMin':0,'xMax':1.2}
    plotDict2 = {'cmap':'jet', 'colorMin':0,'colorMax':10, 'xMin':0,'xMax':40}
    plotDict3 = {'cmap':'jet', 'colorMin':0,'colorMax':10, 'xMin':0,'xMax':40}
    bundleList = []
    #bundleList.append(metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, 
    #                                             plotFuncs=plotList, metadata=md))
    #bundleList.append(metricBundles.MetricBundle(metric2,slicer,sql, plotDict=plotDict2, plotFuncs=plotList))

    slicer=slicers.HealpixSlicer(lonCol='ditheredRA', latCol='ditheredDec')
    bundleList.append(metricBundles.MetricBundle(metric,slicer,'', plotDict=plotDict, 
                                                 metadata=md, plotFuncs=plotList))
    bundleList.append(metricBundles.MetricBundle(metric2,slicer,'', plotDict=plotDict2, 
                                                 metadata=md, plotFuncs=plotList))
    
    bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
    bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)



for runName in runs:
    md = runs[runName]
    metric3 = metrics.InterNightGapsMetric(reduceFunc=np.max, metricName='Max nightly gap')
    plotDict3 = {'cmap':'jet', 'colorMin':0,'colorMax':500, 'xMin':0,'xMax':500}
    bundleList = []

    slicer=slicers.HealpixSlicer(lonCol='ditheredRA', latCol='ditheredDec')
    bundleList.append(metricBundles.MetricBundle(metric3,slicer,'filter="r"',plotDict=plotDict3, plotFuncs=plotList,
                                                 metadata=md))

    bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
    bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)

for runName in runs:
    md = runs[runName]
    metric3 = metrics.InterNightGapsMetric(reduceFunc=np.max, metricName='Max nightly gap')
    plotDict3 = {'cmap':'jet', 'colorMin':0,'colorMax':500, 'xMin':0,'xMax':500}
    bundleList = []

    slicer=slicers.HealpixSlicer(lonCol='ditheredRA', latCol='ditheredDec')
    bundleList.append(metricBundles.MetricBundle(metric3,slicer,'',plotDict=plotDict3, plotFuncs=plotList,
                                                 metadata=md))

    bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
    bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)

# OK, now let's look at the median time gap for each of the runs
plotList = [plotters.HealpixSkyMap()]
plotDict = {'cmap':'jet', 'colorMin':0.4,'colorMax':0.8, 'xMin':0,'xMax':40, 'logScale':False}
plotDict2 = {'cmap':'jet','colorMin':0, 'colorMax':50,  'logScale':False}
sql = 'filter="r"'
metric = metrics.AveGapMetric(metricName='Median Gap')
metric2 = metrics.InterNightGapsMetric()
runs = {'ops2_1094':'No Pairs', 'enigma_1257':'Pairs', 'enigma_1258':'Triples', 'enigma_1259':'Quads'}
orderedRuns = ['ops2_1094','enigma_1257','enigma_1258','enigma_1259']
outDir = 'Avegap'
resultsDb = db.ResultsDb(outDir=outDir)
bundleList = []
slicer=slicers.HealpixSlicer(lonCol='ditheredRA', latCol='ditheredDec')
for runName in orderedRuns:
    bundleList = []
    opsdb = db.OpsimDatabase('/Users/yoachim/Scratch/Opsim_sqlites/'+runName+'_sqlite.db')
    md = runs[runName]
    bundleList.append(metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, 
                                                 metadata=md+',dithered', plotFuncs=plotList))
    bundleList.append(metricBundles.MetricBundle(metric2,slicer,sql, plotDict=plotDict2, 
                                                 metadata=md+',dithered', plotFuncs=plotList))
    bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
    bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)
 





