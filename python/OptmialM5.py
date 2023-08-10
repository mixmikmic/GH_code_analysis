import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import healpy as hp
import numpy as np

# Set up the database connection
opsdb = db.OpsimDatabase('ewok_1004_sqlite.db')
outDir = 'optm5'
resultsDb = db.ResultsDb(outDir=outDir)

bundleList = []
filters = ['u','g','r','i','z','y']
slicer = slicers.HealpixSlicer(nside=64)
plotter = plots.HealpixSkyMap()
metric1 = metrics.Coaddm5Metric(m5Col='fiveSigmaDepth')
metric2 = metrics.Coaddm5Metric(m5Col='m5Optimal', metricName='Optimal m5')
summaryList = [metrics.MedianMetric()]
for filterName in filters:
    sql = 'filter="%s"' % filterName
    bundleList.append(metricBundles.MetricBundle(metric1,slicer,sql, plotFuncs=[plotter], summaryMetrics=summaryList))
    bundleList.append(metricBundles.MetricBundle(metric2,slicer,sql, plotFuncs=[plotter], summaryMetrics=summaryList))

bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()

for i,filtername in enumerate(filters):
    newmap = bundleList[i*2+1].metricValues-bundleList[i*2].metricValues
    newmap[np.where(bundleList[i*2+1].metricValues.mask == True)] = hp.UNSEEN
    hp.mollview(newmap, 
                title='Optimal co-add - regular, %s' % filtername, min=0,max=.5)

print 'filter,  Optimal m5, Regular m5, diff'
for i,filtername in enumerate(filters):
    optimal = bundleList[i*2+1].summaryValues['Median']
    regular = bundleList[i*2].summaryValues['Median']
    print '%s, %.2f, %.2f, %.2f' % (filtername, optimal, regular, optimal-regular)

bundleList = []
filters = ['u','g','r','i','z','y']
slicer = slicers.HealpixSlicer(nside=64, lonCol='ditheredRA', latCol='ditheredDec')
stacker = stackers.M5OptimalStacker()
plotters = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
metric1 = metrics.OptimalM5Metric()
metric2 = metrics.OptimalM5Metric(normalize=True, metricName='PercentBehind')
metric3 = metrics.OptimalM5Metric(magDiff=True, metricName='MagDiff')
summaryList = [metrics.MedianMetric()]
plotDict={'colorMin':0., 'colorMax':40.}
for filterName in filters:
    sql = 'filter="%s"' % filterName
    bundleList.append(metricBundles.MetricBundle(metric1,slicer,sql, plotFuncs=plotters, 
                                                 summaryMetrics=summaryList,plotDict=plotDict,
                                                 stackerList=[stacker]))
    bundleList.append(metricBundles.MetricBundle(metric2,slicer,sql, plotFuncs=plotters, 
                                                 summaryMetrics=summaryList, plotDict=plotDict,
                                                 stackerList=[stacker]))
    bundleList.append(metricBundles.MetricBundle(metric3,slicer,sql, plotFuncs=plotters, 
                                                 summaryMetrics=summaryList, 
                                                 stackerList=[stacker]))

bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)

# figure out median number of visits
bl = []
sql = 'filter="%s"' % filterName
metric = metrics.CountMetric(col='expMJD')
bl.append(metricBundles.MetricBundle(metric,slicer,'', summaryMetrics=summaryList))
bundleDict = metricBundles.makeBundlesDictFromList(bl)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()

print 'filter,  N-behind, %-behind'
totBehind = 0
for i,filtername in enumerate(filters):
    perbehind = bundleList[i*2+1].summaryValues['Median']
    nbehind  = bundleList[i*2].summaryValues['Median']
    totBehind += nbehind
    print '%s, %.1f, %.2f' % (filtername, nbehind, perbehind)
print '------'
print 'all filters, %.1f visits behind, median of %.1f visits per field' % (totBehind, bl[0].summaryValues['Median'])





