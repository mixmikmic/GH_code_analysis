import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
from lsst.sims.maf.metrics import BaseMetric
import lsst.sims.maf.utils as utils

db_minion = db.OpsimDatabase('/Users/yoachim/Scratch/Opsim_sqlites/minion_1016_sqlite.db')
db_enigma = db.OpsimDatabase('/Users/yoachim/Scratch/Opsim_sqlites/enigma_1189_sqlite.db')

sql = 'fieldRA < %f and fieldDec < %f and fieldDec > %f and filter = "r"' % (np.radians(3.5), np.radians(0.1), np.radians(-3.5) )

cols = ['expMJD', 'fiveSigmaDepth', 'FWHMgeom', 'filter']
minion_data = utils.getSimData(db_minion, sql, cols)
cols = ['expMJD', 'fiveSigmaDepth', 'finSeeing', 'filter']
enigma_data = utils.getSimData(db_enigma, sql, cols)

plt.hist(minion_data['FWHMgeom'], bins=20)
plt.title('minion, median=%f' % np.median(minion_data['FWHMgeom']))

plt.hist(enigma_data['finSeeing'], bins=20)
plt.title('minion, median=%f' % np.median(enigma_data['finSeeing']))

star_mag = 24.
enigma_prec = utils.astrom_precision(enigma_data['finSeeing'], utils.m52snr(star_mag, enigma_data['fiveSigmaDepth']))
minion_prec = utils.astrom_precision(minion_data['FWHMgeom'], utils.m52snr(star_mag,minion_data['fiveSigmaDepth']))

bins = np.arange(0,.5,.01)
plt.hist(minion_prec, bins=bins)
plt.title('Minion astrometric precision')
plt.xlabel('arcsec')

plt.hist(enigma_prec, bins=bins)
plt.title('Enigma astrometric precision')
plt.xlabel('arcsec')

enigma_metric = metrics.ProperMotionMetric(seeingCol='finSeeing', rmag=24.)
minion_metric = metrics.ProperMotionMetric(seeingCol='FWHMgeom', rmag=24.)

print 'enigma proper motion precision = % f' % enigma_metric.run(enigma_data)
print 'minion proper motion precision = % f' % minion_metric.run(minion_data)

nside = 32
plotDict = {'colorMin': 0, 'colorMax': 5, 'xMin':0., 'xMax': 5.}
slicer = slicers.HealpixSlicer(nside=nside, lonCol='ditheredRA', latCol='ditheredDec')
minion_bundle = metricBundles.MetricBundle(minion_metric, slicer, '', plotDict=plotDict)
bg = metricBundles.MetricBundleGroup({0:minion_bundle}, db_minion)
bg.runAll()
bg.plotAll(closefigs=False)



slicer = slicers.HealpixSlicer(nside=nside, lonCol='ditheredRA', latCol='ditheredDec')
enigma_bundle = metricBundles.MetricBundle(enigma_metric, slicer, '', plotDict=plotDict)
bg = metricBundles.MetricBundleGroup({0:enigma_bundle}, db_enigma)
bg.runAll()
bg.plotAll(closefigs=False)

print 'median enigma proper motion precision = %f mas/yr' % np.median(enigma_bundle.metricValues)
print 'median minion proper motion precision = %f mas/yr' % np.median(minion_bundle.metricValues)



