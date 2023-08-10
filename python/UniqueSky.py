import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plotters

outDir = 'PairCheck'
resultsDb = db.ResultsDb(outDir=outDir)
opsdb = db.OpsimDatabase('ops2_1093_sqlite.db')

bundleList = []
summaryStats = [metrics.MedianMetric()]
slicer = slicers.OneDSlicer(sliceColName='night', binsize=1)
metric = metrics.CountMetric(col='fieldID')
sql=''
displayDict = {'group':'Fields Per Night'}
bundleList.append(metricBundles.MetricBundle(metric,slicer,sql,
                                             summaryMetrics=summaryStats,
                                             displayDict=displayDict))
metric = metrics.CountUniqueMetric(col='fieldID')
bundleList.append(metricBundles.MetricBundle(metric,slicer,sql,
                                             summaryMetrics=summaryStats,
                                             displayDict=displayDict))

mbd = metricBundles.makeBundlesDictFromList(bundleList)
group = metricBundles.MetricBundleGroup(mbd, opsdb,
                                        outDir=outDir, resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)

plt.plot(bundleList[0].metricValues.compressed(),
         bundleList[1].metricValues.compressed()/bundleList[0].metricValues.compressed(), 
          'ko', alpha = 0.1)
plt.xlabel('Number of Visits')
plt.ylabel('Unique Field Fraction')





