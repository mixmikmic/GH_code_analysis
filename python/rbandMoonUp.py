import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils

opsdb = db.OpsimDatabase('ewok_1004_sqlite.db')
outDir = 'Out'
resultsDb = db.ResultsDb(outDir=outDir)

sql = 'moonPhase > 90 and filter="r" and moonAlt > 0'
slicer=slicers.UniSlicer()
metric = metrics.CountMetric(col='expMJD')
bundle = metricBundles.MetricBundle(metric,slicer,sql)

bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()

bundle.metricValues

sql = 'filter="r"'
slicer=slicers.UniSlicer()
metric = metrics.CountMetric(col='expMJD')
bundle = metricBundles.MetricBundle(metric,slicer,sql)

bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
print bundle.metricValues

sql = 'moonPhase > 0.9 and moonAlt > 0'
slicer=slicers.UniSlicer()
metric = metrics.CountMetric(col='expMJD')
bundle = metricBundles.MetricBundle(metric,slicer,sql)

bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
print bundle.metricValues

