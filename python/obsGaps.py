import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
from lsst.sims.maf.metrics import BaseMetric

database = 'enigma_1189_sqlite.db'
opsdb = db.OpsimDatabase(database)
outDir = 'Gaps'
resultsDb = db.ResultsDb(outDir=outDir)

constraint_dict = {}
constraint_dict['anyFilter'] = ''
constraint_dict['blue'] = "filter='u' or filter='g'"
constraint_dict['red'] = "filter='r' or filter='i' or filter='z' or filter='y'"
constraint_dict['u'] = "filter='u'"
constraint_dict['g'] = "filter='g'"
bundleList = []
slicer = slicers.OpsimFieldSlicer()
metric = metrics.InterNightGapsMetric()

for key in constraint_dict.keys():
    bundle=metricBundles.MetricBundle(metric, slicer, constraint_dict[key])
    bundleList.append(bundle)

bdict = metricBundles.makeBundlesDictFromList(bundleList)
bgroup = metricBundles.MetricBundleGroup(bdict, opsdb, outDir=outDir,
                                         resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)



