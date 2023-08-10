import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plotters

database = 'enigma_1189_sqlite.db'
opsdb = db.OpsimDatabase(database)
outDir = 'CoaddChips'
resultsDb = db.ResultsDb(outDir=outDir)
filters=['u','g','r','i','z','y']

metric = metrics.Coaddm5Metric()
slicer = slicers.HealpixSlicer(nside=128,lonCol='ditheredRA', latCol='ditheredDec')
bundleList = []
for filtName in filters:
    sql = 'filter="%s"' % filtName
    bundleList.append(metricBundles.MetricBundle(metric,slicer,sql))
bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
bg = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bg.runAll()
bg.plotAll(closefigs=False)

slicer = slicers.HealpixSlicer(nside=16,lonCol='ditheredRA', latCol='ditheredDec', useCamera=True)
bundleList = []
for filtName in filters:
    sql = 'filter="%s"' % filtName
    bundleList.append(metricBundles.MetricBundle(metric,slicer,sql))
bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
bg = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bg.runAll()
bg.plotAll(closefigs=False)



