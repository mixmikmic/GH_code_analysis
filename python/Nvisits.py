import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db

# Set the database and query
database = 'enigma_1189_sqlite.db'
opsdb = db.OpsimDatabase(database)
outDir = 'Nvis'
resultsDb = db.ResultsDb(outDir=outDir)

metric = metrics.CountMetric(col='expMJD', units='Number of Visits')
#slicer = slicers.HealpixSlicer(nside=128, lonCol='ditheredRA', latCol='ditheredDec')
slicer=slicers.OpsimFieldSlicer()
sql = ''
plotDict={'colorMax':2000,'colorMin':0}
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict)

bg = metricBundles.MetricBundleGroup({0:bundle},opsdb, outDir=outDir, resultsDb=resultsDb)
bg.runAll()

bg.plotAll(closefigs=False)

# adjust the color max
bundle.plotDict['colorMax'] = 1400
bg.plotAll(closefigs=False)

# Change the color map
bundle.plotDict['cmap'] = 'jet'
bg.plotAll(closefigs=False)





