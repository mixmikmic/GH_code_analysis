import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
import lsst.sims.maf.plots as plots
import healpy as hp



nside = 16

opsdb = db.OpsimDatabase('minion_1016_sqlite.db')
outDir = 'output'
sql='night < 365'
resultsDb = db.ResultsDb(outDir=outDir)
plotFuncs = [plots.TwoDMap()]

bundleList = []
mags = np.arange(15.,30,1)
metric = metrics.ParallaxMetric(rmag=mags)
slicer = slicers.HealpixSlicer(nside=nside, latCol='ditheredDec', lonCol='ditheredRA')
plotDict = {'xlabel':'mag', 'cbarTitle':'mas', 'colorMax':10, 'xextent':[mags.min(), mags.max()]}
bundleList.append(metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, plotFuncs=plotFuncs))

bd = metricBundles.makeBundlesDictFromList(bundleList)
group = metricBundles.MetricBundleGroup(bd, opsdb, outDir=outDir,
                                        resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)

# Note that each column in the above image is a full healpixel map
hp.mollview(bundleList[0].metricValues[:,3])





