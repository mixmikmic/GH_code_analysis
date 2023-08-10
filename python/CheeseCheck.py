import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plotters

runList = ['enigma_1189','ops2_1102', 'enigma_1260', 'enigma_1261']
runDict = {'enigma_1189':'Baseline','ops2_1102':'Cheese1', 'enigma_1260':'Cheese2', 'enigma_1261':'Cheese_no_restart'}
dbDir = '/Users/yoachim/Scratch/Opsim_sqlites/'
outDir = 'CheeseTest'
resultsDb = db.ResultsDb(outDir=outDir)
plotList = [plotters.HealpixSkyMap()]#, plotters.HealpixHistogram()]
summaryMetrics=[metrics.MedianMetric()]
nightBins = np.arange(0,3564+365.25,365.25)


BLinter=[]
BLintra = []
for i in range(nightBins.size-1):
    
    for runName in runList:
        # Set up the database connection
        opsdb = db.OpsimDatabase(dbDir+runName+'_sqlite.db')
        md = runDict[runName]+' year %i' % (i+1)
        metric = metrics.IntraNightGapsMetric()
        metric2 = metrics.InterNightGapsMetric()
        plotDict = {'cmap':'jet', 'colorMin':0,'colorMax':0.5, 'xMin':0,'xMax':1.2}
        plotDict2 = {'cmap':'jet', 'colorMin':0,'colorMax':40, 'xMin':0,'xMax':40}
        sql = 'night > % i and night < %i and filter="r"' % (nightBins[i], nightBins[i+1])
        bundleList = []
        slicer=slicers.HealpixSlicer(lonCol='ditheredRA', latCol='ditheredDec')
        bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, 
                                            metadata=md, plotFuncs=plotList,
                                            summaryMetrics=summaryMetrics)
        bundleList.append(bundle)
        BLintra.append(bundle)
        bundle = metricBundles.MetricBundle(metric2,slicer,sql, plotDict=plotDict2, 
                                            metadata=md, plotFuncs=plotList,
                                            summaryMetrics=summaryMetrics)
        bundleList.append(bundle)
        BLinter.append(bundle)
        bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
        bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
        bgroup.runAll()
        bgroup.plotAll(closefigs=False)



print 'run,  median gap (days)'
for bundle in BLinter:
    print bundle.metadata, bundle.summaryValues['Median']

print 'run,  median gap (hours)'
for bundle in BLintra:
    print bundle.metadata, bundle.summaryValues['Median']

# Try out demanding to observations in g,r,or i
#peaks = {'uPeak':35, 'gPeak':20, 'rPeak':20, 'iPeak':20, 'zPeak':35, 'yPeak':35}
#sql = 'filter="g" or filter="r" or filter="i"'
allBundles = []
summaryMetrics=[metrics.SumMetric()]
peaks = {'uPeak':25.9, 'gPeak':23.6, 'rPeak':22.6, 'iPeak':22.7, 'zPeak':22.7,'yPeak':22.8}
peakTime = 15.
transDuration = peakTime+30. # Days
for i in range(nightBins.size-1):

    for runName in runList:
        sql = 'night > % i and night < %i' % (nightBins[i], nightBins[i+1])
        # Set up the database connection
        opsdb = db.OpsimDatabase(dbDir+runName+'_sqlite.db')
        md = runDict[runName]+' year %i' % (i+1)
        #metric = metrics.TransientMetric(transDuration=20., peakTime=10., nPrePeak=2, nFilters=2, 
        #                                 nPerLC=3, **peaks)
        metric = metrics.TransientMetric(riseSlope= -2./peakTime, declineSlope=1.4/30.0,
                                         transDuration=transDuration, peakTime=peakTime,
                                         nFilters=3, nPrePeak=3, nPerLC=2,
                                         metricName='SNLots',nPhaseCheck=3,**peaks)
        bundleList = []
        plotDict = {'cmap':'jet'}
        slicer=slicers.HealpixSlicer(lonCol='ditheredRA', latCol='ditheredDec', nside=64)

        bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, 
                                            metadata=md, plotFuncs=plotList,
                                            summaryMetrics=summaryMetrics)
        bundleList.append(bundle)
        allBundles.append(bundle)
        bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
        bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
        bgroup.runAll()
        bgroup.plotAll(closefigs=False)

print 'run,  number of SN (proportional)'
for bundle in allBundles:
    print bundle.metadata, bundle.summaryValues['Sum']







