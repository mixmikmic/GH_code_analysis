import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')

from moSlicer import MoSlicer
import moMetrics as MoMetrics
from moSummaryMetrics import ValueAtHMetric
import moPlots as moPlots
import moStackers as moStackers
import moMetricBundle as mmb
import lsst.sims.maf.plots as plots

orbitfile = 'neo_49353.des.db'
obsfile1 = 'neos_s3m_747_allObs.txt'
obsfile2 = 'neos_s3m_474_allObs.txt' #I ran this night also, by mistake. Might as well have a look.

mos = MoSlicer(orbitfile, Hrange=None)
mos.readObs(obsfile1)

mos2 = MoSlicer(orbitfile, Hrange=None)
mos2.readObs(obsfile2)
# Since we'll only be considering a single H value for each object, go ahead and:
# Add mag limit (including detection losses) to obs
# Add SNR to obs 
# Add calculated 'visibility' to each object (probabilities m5 calculation)
allStackers = moStackers.AllStackers()
mos.obs = allStackers.run(mos.obs, mos.orbits['H'], mos.orbits['H'])
mos2.obs = allStackers.run(mos2.obs, mos2.orbits['H'], mos2.orbits['H'])

mos.obs.tail()

# How many objects were observed (without mag limit)
print "IN NIGHT 747:"
print "How many objects were in orbit file?", mos.nSso
print "How many objects made it onto the fields LSST observed?", len(np.unique(mos.obs['objId']))
vis = np.where(mos.obs['SNR']>=5)[0]
print "How many objects were bright enough to see?", len(np.unique(mos.obs['objId'][vis]))

# How many objects were observed (without mag limit)
print "IN NIGHT 474"
print "How many objects were in orbit file?", mos2.nSso
print "How many objects made it onto the fields LSST observed?", len(np.unique(mos2.obs['objId']))
vis2 = np.where(mos2.obs['SNR']>=5)[0]
print "How many objects were bright enough to see?", len(np.unique(mos2.obs['objId'][vis2]))

# Look at where the fields that were observed were distributed across the sky.
from lsst.sims.maf.metrics import CountMetric, FilterColorsMetric
from lsst.sims.maf.stackers import FilterColorStacker
from lsst.sims.maf.slicers import OpsimFieldSlicer, HealpixSlicer
from lsst.sims.maf.db import OpsimDatabase
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots

plotfunc = [plots.BaseSkyMap()]
nvisits1 = metricBundles.MetricBundle(metric=CountMetric('expMJD'), 
                                      slicer=OpsimFieldSlicer(), sqlconstraint='night=747')
filtervisits1 = metricBundles.MetricBundle(metric=FilterColorsMetric(), 
                                           slicer=OpsimFieldSlicer(), sqlconstraint='night=747',
                                          plotDict={'metricIsColor':True}, plotFuncs=plotfunc)
nvisits2 = metricBundles.MetricBundle(metric=CountMetric('expMJD'), 
                                      slicer=OpsimFieldSlicer(), sqlconstraint='night=474')
filtervisits2 = metricBundles.MetricBundle(metric=FilterColorsMetric(), 
                                           slicer=OpsimFieldSlicer(), sqlconstraint='night=474',
                                           plotDict={'metricIsColor':True}, plotFuncs=plotfunc)

opsdb = OpsimDatabase('../enigma_1189_sqlite.db')

mbg = metricBundles.MetricBundleGroup({'nv1':nvisits1, 'nv2':nvisits2, 'fv1':filtervisits1, 'fv2':filtervisits2}, 
                                          opsdb)
mbg.runAll()
mbg.plotAll(closefigs=False)

# Just check that our five sigma depth went the correct direction / what were the adjustments due to trailing.
# magFilter = magnitude in filter
# does not include mag losses due to detection - apply these to fiveSigmaDepth
plt.plot(mos.obs['fiveSigmaDepth'], mos.obs['magLimit'], 'k.')
plt.xlabel('Five Sigma Image Depth - point source')
plt.ylabel('m5 including detection losses')

# Histogram the SNR distribution of the observations.
n, b, p = plt.hist(mos.obs['SNR'], bins=np.arange(0, 20, 0.3))
plt.ylim(0, 200)
plt.xlabel("SNR")
# Most objects have very low SNR, which is expected because NEO model has many small objects
vis = np.where(mos.obs['SNR'] >= 5)[0]
plt.figure()
n, b, p = plt.hist(mos.obs['SNR'][vis], bins=np.arange(0, 20, 0.3))
plt.xlabel('SNR (visible)')

# What did our apparent magnitude distribution look like?
n, b, p = plt.hist(mos.obs['appMag'], bins=100, alpha=0.2)
n, b, p = plt.hist(mos.obs['appMag'][vis], bins=50)
plt.xlabel('Apparent magnitude')
plt.ylabel('Number of NEOS')
plt.ylim(0, 300)
plt.xlim(15, 26)

# What did our orbital distribution look like?
plt.plot(mos.orbits['q'], mos.orbits['e'], 'r.', alpha=0.5)
plt.plot(mos.orbits['q'][vis], mos.orbits['e'][vis], 'k.')
plt.xlabel('q')
plt.ylabel('e')
plt.xlim(0, 1.5)

# Let's just look at where the observations of the *objects* (not the fields) were in the sky.
slicer = HealpixSlicer(nside=128)
metric = CountMetric('expMJD', metricName='Count NEO Obs', units='')
simdata = mos.obs.to_records()
sqlconstraint = 'night=747'
mb = metricBundles.MetricBundle(metric, slicer, sqlconstraint=sqlconstraint, 
                                plotFuncs=[plots.HealpixSkyMap()], 
                                plotDict={'rot':(90, 0, 0), 'colorMax':1000})
bg = metricBundles.MetricBundleGroup({'mb':mb}, opsdb)

bg._setCurrent(sqlconstraint)
bg.simData = simdata
bg._runCompatible(['mb'])
bg.plotAll(closefigs=False)


slicer = HealpixSlicer(nside=128)
metric = CountMetric('expMJD', metricName='Count NEO Obs', units='')
simdata = mos2.obs.to_records()
sqlconstraint = 'night=474'
mb = metricBundles.MetricBundle(metric, slicer, sqlconstraint=sqlconstraint, 
                                plotFuncs=[plots.HealpixSkyMap()], 
                                plotDict={'rot':(90, 0, 0), 'colorMax':1000})
bg = metricBundles.MetricBundleGroup({'mb':mb}, opsdb)

bg._setCurrent(sqlconstraint)
bg.simData = simdata
bg._runCompatible(['mb'])
bg.plotAll(closefigs=False)



