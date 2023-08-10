import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as metricBundles

# Set up the database connection
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'AltAz'
resultsDb = db.ResultsDb(outDir=outDir)

slicer = slicers.HealpixSlicer(nside=64, latCol='zenithDistance', lonCol='azimuth', useCache=False)
metric = metrics.CountMetric('expMJD', metricName='Nvisits as function of Alt/Az')
plotDict = {}
nightLimits = np.arange(0,11,1)*365.25
for lowerLimit,upperLimit in zip(nightLimits[:-1],nightLimits[1:]):
    bundleList = []
    plotFuncs = [plots.LambertSkyMap()]
    sql = 'night between %i and %i' % (lowerLimit,upperLimit)
    bundle = metricBundles.MetricBundle(metric, slicer,sql,
                                        plotFuncs=plotFuncs)
    bundleList.append(bundle)
    bDict = metricBundles.makeBundlesDictFromList(bundleList)
    bgroup = metricBundles.MetricBundleGroup(bDict, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)

slicer = slicers.HealpixSlicer(nside=64, latCol='zenithDistance', lonCol='azimuth', useCache=False)
metric = metrics.CountMetric('expMJD', metricName='Nvisits as function of Alt/Az')
plotDict = {}
months = np.arange(0,365.25+365.25/12, 365.25/12)

for lowerLimit,upperLimit in zip(months[:-1],months[1:]):
    bundleList = []
    plotFuncs = [plots.LambertSkyMap()]
    sql = 'night %% 365.25 between %i and %i' % (lowerLimit,upperLimit)
    bundle = metricBundles.MetricBundle(metric, slicer,sql,
                                        plotFuncs=plotFuncs)
    bundleList.append(bundle)
    bDict = metricBundles.makeBundlesDictFromList(bundleList)
    bgroup = metricBundles.MetricBundleGroup(bDict, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)

slicer = slicers.HealpixSlicer(nside=64, latCol='zenithDistance', lonCol='azimuth', useCache=False)
metric = metrics.CountMetric('expMJD', metricName='Nvisits as function of Alt/Az')
plotDict = {}
ras = np.radians(np.arange(0,24+2, 2)*360./24.)
for lowerLimit,upperLimit in zip(ras[:-1],ras[1:]):
    bundleList = []
    plotFuncs = [plots.LambertSkyMap()]
    sql = 'fieldRA between %f and %f' % (lowerLimit,upperLimit)
    bundle = metricBundles.MetricBundle(metric, slicer,sql,
                                        plotFuncs=plotFuncs)
    bundleList.append(bundle)
    bDict = metricBundles.makeBundlesDictFromList(bundleList)
    bgroup = metricBundles.MetricBundleGroup(bDict, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)



# Make a time-to-midnight stacker
from lsst.sims.maf.utils.telescopeInfo import TelescopeInfo
import ephem
from lsst.sims.maf.stackers import BaseStacker

def nearestVal(A, val):
    return A[np.argmin(np.abs(np.array(A)-val))]

class TimeToMidnight(BaseStacker):
    def __init__(self, mjdCol='expMJD', fractionOfNight=False, telescope='LSST'):
        """
        Negative means early in the night, positive means after midnight
        """
        self.fractionOfNight = fractionOfNight
        self.units=['Days']
        self.colsAdded = ['timeToMidnight']
        self.colsReq = [mjdCol]
        self.mjdCol = mjdCol
        self.telescope = TelescopeInfo(telescope)
        
    def _run(self, simData):
        lsstObs = ephem.Observer()
        lsstObs.lat = self.telescope.lat
        lsstObs.lon = self.telescope.lon
        lsstObs.elevation = self.telescope.elev
        S = ephem.Sun()
        
        # Offset of MJD to DJD
        doff = ephem.Date(0)-ephem.Date('1858/11/17')
        nearestMidnight = np.zeros(simData.size,dtype=float)
        for i,mjd in enumerate(simData[self.mjdCol]):
            mjd = mjd-doff
            nearestMidnight[i] = nearestVal([lsstObs.previous_antitransit(S, start=mjd),
                                             lsstObs.next_antitransit(S, start=mjd)], mjd )+doff
        simData['timeToMidnight'] = simData[self.mjdCol] - nearestMidnight
        return simData



slicer = slicers.HealpixSlicer(nside=64, latCol='zenithDistance', lonCol='azimuth', useCache=False)
metric = metrics.CountMetric('expMJD', metricName='Nvisits as function of Alt/Az')
stackers = [TimeToMidnight()]
plotDict = {}


bundleList = []
plotFuncs = [plots.LambertSkyMap()]
sql = '' 
bundle = metricBundles.MetricBundle(metric, slicer,sql,
                                    plotFuncs=plotFuncs, stackerList=stackers)
bundleList.append(bundle)
bDict = metricBundles.makeBundlesDictFromList(bundleList)
bgroup = metricBundles.MetricBundleGroup(bDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()

allVisits = bgroup.simData.copy()
# Before Midnight
good = np.where(allVisits['timeToMidnight'] < 0)
bundle.plotDict = {'title':'Before Midnight'}
bgroup.runCurrent('', simData=allVisits[good])
bgroup.plotAll(closefigs=False)

#After Midnight
good = np.where(allVisits['timeToMidnight'] > 0)
bundle.plotDict = {'title':'After Midnight'}
bgroup.runCurrent('', simData=allVisits[good])
bgroup.plotAll(closefigs=False)

# Within 2 hours of midnight
good = np.where(np.abs(allVisits['timeToMidnight']) < 2./24.)
bundle.plotDict = {'title':'Within 2 hours of midnight'}
bgroup.runCurrent('', simData=allVisits[good])
bgroup.plotAll(closefigs=False)









