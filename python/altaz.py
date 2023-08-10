import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots

outDir = 'AltAz'
resultsDb = db.ResultsDb(outDir=outDir)
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
propids, propTags = opsdb.fetchPropInfo()
filters = ['u','g','r','i','z','y']

slicer = slicers.HealpixSlicer(nside=64, latCol='zenithDistance', lonCol='azimuth', useCache=False)
metric = metrics.CountMetric('expMJD', metricName='Nvisits as function of Alt/Az')
plotDict = {'rot':(0,90,0)}
plotFuncs = [plots.HealpixSkyMap()]
bundleList = []
for filt in filters:
    for propid in propids:
        md = '%s, %s' % (filt, propids[propid])
        sql = 'filter="%s" and propID=%i' % (filt,propid)
        bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict,
                                            plotFuncs=plotFuncs, metadata=md)
        group = metricBundles.MetricBundleGroup({0:bundle}, opsdb,
                                                outDir=outDir, resultsDb=resultsDb)
        group.runAll()
        group.plotAll(closefigs=False)







