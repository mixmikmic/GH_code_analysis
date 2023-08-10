import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as metricBundles
import numpy as np

# Set up the database connection
opsdb = db.OpsimDatabase('/Users/yoachim/Scratch/Opsim_sqlites/minion_1016_sqlite.db')
#opsdb = db.OpsimDatabase('/Users/yoachim/Scratch/Opsim_sqlites/opsim3_61_sqlite.db', defaultdbTables={'Summary':['Summary', 'obsHistID']})
outDir = 'astrometry_dcr'
resultsDb = db.ResultsDb(outDir=outDir)

bundleList = []

slicer = slicers.UserPointsSlicer(113.75, -60.68)
metric = metrics.PassMetric()
sql = 'filter = "r" or filter = "g" or filter = "i" or filter = "z"'
stackerList = []
stackerList.append(stackers.ParallaxFactorStacker())
stackerList.append(stackers.DcrStacker())
stackerList.append(stackers.HourAngleStacker())
bundle = metricBundles.MetricBundle(metric, slicer, sql, stackerList=stackerList)
bundleList.append(bundle)

bundleList.append(metricBundles.MetricBundle(metrics.ParallaxDcrDegenMetric(), slicer, sql, stackerList=stackerList))
#bundleList.append(metricBundles.MetricBundle(metrics.ParallaxHADegenMetric(), slicer, sql, stackerList=stackerList))

# bundleList.append(metricBundles.MetricBundle(metrics.ParallaxDcrDegenMetric(seeingCol='seeing', m5Col='fivesigma_modified'), slicer, sql, stackerList=stackerList))

bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()

bgroup.plotAll(closefigs=False)

print 'Parallax-DCR amplitude correlation', bundleList[1].metricValues

plt.scatter(bundle.metricValues[0]['ra_pi_amp'], bundle.metricValues[0]['ra_dcr_amp'], c=bundle.metricValues[0]['HA'])
plt.xlabel('Parallax RA offset (arcsec)')
plt.ylabel('DCR RA offset (arcsec)')
cb = plt.colorbar()
cb.set_label('Hour Angle (hours)')

pi_r = (bundle.metricValues[0]['ra_pi_amp']**2 + bundle.metricValues[0]['dec_pi_amp']**2)**0.5

plt.scatter(pi_r, bundle.metricValues[0]['HA'], c=bundle.metricValues[0]['ra_dcr_amp'], alpha=.5)
plt.xlabel('Parallax Factor (arcsec)')
plt.ylabel('Hour Angle (hours)')
cb = plt.colorbar()
cb.set_label('DCR RA offset (arcsec)')

plt.scatter(bundle.metricValues[0]['dec_pi_amp'], bundle.metricValues[0]['dec_dcr_amp'], 
            c=bundle.metricValues[0]['HA'], alpha=.5)
plt.xlabel('Parallax Dec offset (arcsec)')
plt.ylabel('DCR Dec offset (arcsec)')
cb = plt.colorbar()
cb.set_label('Hour Angle (hours)')

pi_r = (bundle.metricValues[0]['ra_pi_amp']**2 + bundle.metricValues[0]['dec_pi_amp']**2)**0.5
dcr_r = (bundle.metricValues[0]['ra_dcr_amp']**2 + bundle.metricValues[0]['dec_dcr_amp']**2)**0.5
plt.plot(pi_r, dcr_r, 'ko')
plt.xlabel('Parallax amplitude (arcsec)')
plt.ylabel('DCR amplitude (arcsec)')

# And now for one that shouldn't look as bad
bundleList = []

slicer = slicers.UserPointsSlicer(289.95, -42.78)
metric = metrics.PassMetric()
sql = 'filter = "r" or filter = "g" or filter = "i" or filter = "z"'
stackerList = []
stackerList.append(stackers.ParallaxFactorStacker())
stackerList.append(stackers.DcrStacker())
bundle = metricBundles.MetricBundle(metric, slicer, sql, stackerList=stackerList)
bundleList.append(bundle)

bundleList.append(metricBundles.MetricBundle(metrics.ParallaxDcrDegenMetric(), slicer, sql, stackerList=stackerList))

bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)
print 'Parallax-DCR amplitude correlation', bundleList[1].metricValues
plt.figure()
plt.scatter(bundle.metricValues[0]['ra_pi_amp'], bundle.metricValues[0]['ra_dcr_amp'], 
            c=bundle.metricValues[0]['HA'], alpha=0.5)
cb = plt.colorbar()
cb.set_label('Hour Angle (hours)')
plt.xlabel('Parallax RA offset (arcsec)')
plt.ylabel('DCR RA offset (arcsec)')
plt.figure()
plt.scatter(bundle.metricValues[0]['dec_pi_amp'], bundle.metricValues[0]['dec_dcr_amp'], 
            c=bundle.metricValues[0]['HA'], alpha=0.5)
cb = plt.colorbar()
cb.set_label('Hour Angle (hours)')
plt.xlabel('Parallax Dec offset (arcsec)')
plt.ylabel('DCR Dec offset (arcsec)')
plt.figure()
pi_r = (bundle.metricValues[0]['ra_pi_amp']**2 + bundle.metricValues[0]['dec_pi_amp']**2)**0.5
dcr_r = (bundle.metricValues[0]['ra_dcr_amp']**2 + bundle.metricValues[0]['dec_dcr_amp']**2)**0.5
plt.plot(pi_r, dcr_r, 'ko')
plt.xlabel('Parallax amplitude (arcsec)')
plt.ylabel('DCR amplitude (arcsec)')

bundleList = []
slicer = slicers.HealpixSlicer()
sql = 'filter = "r" or filter = "g" or filter = "i" or filter = "z"'
bundleList.append(metricBundles.MetricBundle(metrics.ParallaxDcrDegenMetric(), slicer, sql, stackerList=stackerList))
bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)
print 'number of healpix in danger zone=', np.where(np.abs(bundleList[0].metricValues) > 0.7)[0].size

slicer = slicers.HealpixSlicer(nside=64, latCol='zenithDistance', lonCol='azimuth', useCache=False)
metric = metrics.CountMetric('expMJD', metricName='Nvisits as function of Alt/Az')
plotFuncs = [plots.LambertSkyMap()]
bundleList = []
bundleList.append(metricBundles.MetricBundle(metric, slicer, sql, plotFuncs=plotFuncs))
bundleList.append(metricBundles.MetricBundle(metrics.MeanMetric(col='HA'), slicer, sql, plotFuncs=plotFuncs))

bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)

bundleList = []
slicer = slicers.OpsimFieldSlicer()
sql = 'filter = "r" or filter = "g" or filter = "i" or filter = "z"'
bundleList.append(metricBundles.MetricBundle(metrics.ParallaxDcrDegenMetric(), slicer, sql, stackerList=stackerList))
bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)

print 'fieldID, metricValue'
for sid,val in zip(slicer.slicePoints['sid'], bundleList[0].metricValues):
    print sid, val



