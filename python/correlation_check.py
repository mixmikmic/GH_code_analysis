get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles

outDir ='out'
dbFile = 'minion_1016_sqlite.db'
opsimdb = utils.connectOpsimDb(dbFile)
resultsDb = db.ResultsDb(outDir=outDir)

# The pass metric just passes data straight through.
metric = metrics.PassMetric(cols=['filter','fiveSigmaDepth','expMJD', 'HA', 
                                  'ra_pi_amp', 'dec_pi_amp'])
slicer = slicers.HealpixSlicer(nside=16)
sql = 'fieldID = 1600 and filter != "u" and filter != "y"'
bundle = metricBundles.MetricBundle(metric,slicer,sql)
bundle2 = metricBundles.MetricBundle(metrics.ParallaxHADegenMetric(),slicer,sql)
bg =  metricBundles.MetricBundleGroup({0:bundle, 1:bundle2}, opsimdb,
                                      outDir=outDir, resultsDb=resultsDb)

bg.runAll()
bg.plotAll(closefigs=False)

good = np.where(~bundle.metricValues.mask)[0]
print 'Correlation between parallax offset and Hour Angle = %f' % np.max(bundle2.metricValues.data[good])

good = np.where(~bundle.metricValues.mask)[0]

import pandas as pd
df = pd.DataFrame(data=bundle.metricValues.data[good][0])
pd.set_option('display.max_rows', len(df))
print df



