# repoDir = '/Users/clarkson/Progs/Projects/lsstScratchWIC'   # wherever you put the repository
#topDir = '/Users/clarkson/Data/LSST/MetricsEtc/'   # Wherever you put the results of the transients metric
topDir = '/Users/clarkson/Data/LSST/OpSimRuns/opsim20160411'

# For reference, here are the parameters used to simulate the transients:
#peaks = {'uPeak':11, 'gPeak':9, 'rPeak':8, 'iPeak':7, 'zPeak':6,'yPeak':6}
#colors = ['b','g','r','purple','y','magenta','k']
#filterNames = ['u','g','r','i','z','y']
## Timing parameters of the outbursts
#riseSlope = -2.4
#declineSlope = 0.05  # following Ofek et al. 2013
#transDuration = 80.
#peakTime = 20.

# relevant parameter for the TransientMetric:
# nPhaseCheck=20

# Additionally, all filters were used (passed as **peaks to the TransientMetric).

get_ipython().magic('matplotlib inline')

import numpy as np
import time

# Some colormaps we might use
import matplotlib.cm as cm

# Capability to load previously-computed metrics, examine them
import lsst.sims.maf.metricBundles as mb

# plotting (to help assess the results)
import lsst.sims.maf.plots as plots

import os

# Directory where we started everyhing. Outputs go in subdirectories of wDir:
wDir = os.getcwd() 

# Uncomment the following if you want the outputs from your run to go up to the repository
# wDir = '%s/data/fomOutputs' % (topDir) 

print wDir

# Default is to assume the opsim databases are symlinked into the current directory.
#opsimDir = os.getcwd()

# uncomment the following to set the directory on your system where the opsim runs are stored:
# opsimDir = '/Users/clarkson/Data/LSST/OpSimRuns/Runs_20151229/LocalCopies'
opsimDir = '/Users/clarkson/Data/LSST/OpSimRuns/opsim20160411'

# The example CountMetric provided by Mike Lund seems to have the column indices for coords
# hardcoded (which breaks the examples I try on my setup). This version finds the co-ordinates by 
# name instead. First the imports we need:
# import numpy as np
from lsst.sims.maf.metrics import BaseMetric
#from mafContrib import starcount 

# WIC - sims_maf_contrib is no longer recognized by my system. I had to copy the StarCounts module into
# the working directory. So:
from StarCounts import starcount

class AsCountMetric(BaseMetric):

    """
    WIC - Lightly modified copy of Mike Lund's example StarCounts metric in sims_maf_contrib. 
    Accepts the RA, DEC column names as keyword arguments. Docstring from the original:
    
    Find the number of stars in a given field between distNear and distFar in parsecs. 
    Field centers are read from columns raCol and decCol.
    """
    
    def __init__(self,**kwargs):
        
        self.distNear=kwargs.pop('distNear', 100)
        self.distFar=kwargs.pop('distFar', 1000)
        self.raCol=kwargs.pop('raCol', 'ra')
        self.decCol=kwargs.pop('decCol', 'dec')
        super(AsCountMetric, self).__init__(col=[], **kwargs)
        
    def run(self, dataSlice, slicePoint=None):
        sliceRA = np.degrees(slicePoint[self.raCol])
        sliceDEC = np.degrees(slicePoint[self.decCol])
        return starcount.starcount(sliceRA, sliceDEC, self.distNear, self.distFar)

# Go from the notebook directory to the METRIC outputs directory
# os.chdir('%s/data/metricOutputs' % (reppDir))  # I do not recommend using this. 
os.chdir(topDir)

distNear=10.
distFar = 8.0e4  # Get most of the plane but not the Magellanic clouds 

import lsst.sims.maf.slicers as slicers

import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db

slicer = slicers.HealpixSlicer(nside=64)

metricCount=AsCountMetric(distNear=distNear, distFar=distFar)
metricList = [metricCount]

# While the previous metric was running, I had time to write the conditional...
sqlconstraintCount = 'filter = "r" and night < 2000'  # Assume everywhere visited once in 5.5 years...
doPS1 = False

if doPS1:
    runNamePSlike = 'minion_1020'
    sqlconstraintCount = 'filter = "r" and night < 2000'  # Assume everywhere visited once in 5.5 years...
    bDictPSlike={}
    for i,metric in enumerate(metricList):
        bDictPSlike[i] = metricBundles.MetricBundle(metric, slicer, sqlconstraintCount, 
                                              runName=runNamePSlike)
    opsdbPSlike = db.OpsimDatabase(opsimDir + '/' + runNamePSlike + '_sqlite.db')
    outDirPSlike = '%s/TestCountOnly_PSlike' % (topDir)
    resultsDbPSlike = db.ResultsDb(outDir=outDirPSlike)

if doPS1:
    print opsimDir + '/' + runNamePSlike + '_sqlite.db'
    get_ipython().system(' ls /Users/clarkson/Data/LSST/OpSimRuns/opsim20160411/minion_1020_sqlite.db')

tStart = time.time()
if doPS1:
    bgroupPSlike = metricBundles.MetricBundleGroup(bDictPSlike, opsdbPSlike, outDir=outDirPSlike,                                                  resultsDb=resultsDbPSlike)
    bgroupPSlike.runAll()
    tPostPSlike = time.time()
    print "Time spent Counting PSlike: %.3e seconds" % (tPostPSlike - tStart)

# Ensure the output file actually got written...
get_ipython().system(' ls -l ./TestCountOnly_PSlike/*npz')

# We will need the same counts information for the WFDPlane survey if we want to normalize by 
# total counts in the survey area. So let's run the above for astro_lsst_01_1004 as well.
runNameWFDPlane = 'astro_lsst_01_1004'
bDictWFDPlane={}
for i,metric in enumerate(metricList):
    bDictWFDPlane[i] = metricBundles.MetricBundle(metric, slicer, sqlconstraintCount, 
                                          runName=runNameWFDPlane)
opsdbWFDPlane = db.OpsimDatabase(opsimDir + '/' + runNameWFDPlane + '_sqlite.db')
outDirWFDPlane = '%s/TestCountOnly_WFDPlane' % (topDir)
resultsDbWFDPlane = db.ResultsDb(outDir=outDirWFDPlane)

tStart = time.time()
bgroupWFDPlane = metricBundles.MetricBundleGroup(bDictWFDPlane, opsdbWFDPlane, outDir=outDirWFDPlane,                                              resultsDb=resultsDbWFDPlane)
bgroupWFDPlane.runAll()
tPostWFDPlane = time.time()
print "Time spent Counting WFDPlane: %.3e seconds" % (tPostWFDPlane - tStart)

get_ipython().system(' ls ./TestCountOnly_WFDPlane/astro_lsst_01_1004_AsCount_r_HEAL.npz')

# go back to figure-of-merit output directory
os.chdir(wDir)

pathCount='%s/TestCountOnly_PSlike/minion_1020_AsCount_r_and_night_lt_2000_HEAL.npz' % (topDir)
pathTransient='%s/TransientsLike2010mc_PSlike/minion_1020_Alert_sawtooth_HEAL.npz' % (topDir)

#Initialize then load
bundleCount = mb.createEmptyMetricBundle()
bundleTrans = mb.createEmptyMetricBundle()

bundleCount.read(pathCount)
bundleTrans.read(pathTransient)

# Set a mask for the BAD values of the transient metric
bTrans = (np.isnan(bundleTrans.metricValues)) | (bundleTrans.metricValues <= 0.)
bundleTrans.metricValues.mask[bTrans] = True

# Read in the stellar density for WFDPlane so that we can compare the total NStars...
pathCountWFDPlane='%s/TestCountOnly_WFDPlane/astro_lsst_01_1004_AsCount_r_and_night_lt_2000_HEAL.npz' % (topDir)
bundleCountWFDPlane = mb.createEmptyMetricBundle()
bundleCountWFDPlane.read(pathCountWFDPlane)

# Do the comparison
nTotPSlike = np.sum(bundleCount.metricValues)
nTotWFDPlane = np.sum(bundleCountWFDPlane.metricValues)
print "Total NStars - minion_1020: %.3e - astro_lsst_01_1004 %.3e" % (nTotPSlike, nTotWFDPlane)

bundleProc = mb.createEmptyMetricBundle()
bundleProc.read(pathTransient)

# Set the mask
bundleProc.metricValues.mask[bTrans] = True

# Multiply the two together, normalise by the total starcounts over the survey
bundleProc.metricValues = (bundleCount.metricValues * bundleTrans.metricValues) 
bundleProc.metricValues /= np.sum(bundleCount.metricValues)

bundleProc.metric.name = '(sawtooth alert) x (counts) / NStars_total'

FoMPSlike = np.sum(bundleProc.metricValues)
print "FoM PSlike: %.2e" % (FoMPSlike)

pathCountWFDPlane='%s/TestCountOnly_WFDPlane/astro_lsst_01_1004_AsCount_r_and_night_lt_2000_HEAL.npz' % (topDir)
pathTransWFDPlane='%s/TransientsLike2010mc_WFDPlane/astro_lsst_01_1004_Alert_sawtooth_HEAL.npz' % (topDir)
bundleCountWFDPlane = mb.createEmptyMetricBundle()
bundleTransWFDPlane = mb.createEmptyMetricBundle()

bundleCountWFDPlane.read(pathCountWFDPlane)
bundleTransWFDPlane.read(pathTransWFDPlane)
bTransWFDPlane = (np.isnan(bundleTransWFDPlane.metricValues)) | (bundleTransWFDPlane.metricValues <= 0.)
bundleTransWFDPlane.metricValues.mask[bTransWFDPlane] = True

# Load WFDPlane-like metric bundle and replace its values with processed values
bundleProcWFDPlane = mb.createEmptyMetricBundle()
bundleProcWFDPlane.read(pathTransWFDPlane)
bundleProcWFDPlane.metricValues.mask[bTransWFDPlane] = True

bundleProcWFDPlane.metricValues = (bundleCountWFDPlane.metricValues * bundleTransWFDPlane.metricValues) 
bundleProcWFDPlane.metricValues /= np.sum(bundleCountWFDPlane.metricValues)
bundleProcWFDPlane.metric.name = '(sawtooth alert) x (counts) / NStars_total'

FoMWFDPlane = np.sum(bundleProcWFDPlane.metricValues)
print FoMWFDPlane

# Print the sum total of our f.o.m. for each run
print "FOM for minion_1020: %.3f" % (FoMPSlike)
print "FOM for astro_lsst_01_1004: %.3f" % (FoMWFDPlane)

# Same plot information as before:
plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
plotDictProc={'logScale':True, 'cmap':cm.cubehelix_r}
bundleProc.setPlotDict(plotDictProc)
bundleProc.setPlotFuncs(plotFuncs)

plotDictProc={'logScale':True, 'cmap':cm.cubehelix_r}
bundleProcWFDPlane.setPlotDict(plotDictProc)
bundleProcWFDPlane.setPlotFuncs(plotFuncs)



bundleProc.plot(savefig=True)
bundleProcWFDPlane.plot(savefig=True)



# Plot just the spatial map and the histogram for the two. Use different colormaps for each.
#plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
bundleTrans.setPlotFuncs(plotFuncs)
bundleCount.setPlotFuncs(plotFuncs)

# Use a different colormap for each so we can tell them apart easily...
plotDictCount={'logScale':True, 'cmap':cm.gray_r}
plotDictTrans={'logScale':False, 'cmap':cm.RdBu_r}
bundleCount.setPlotDict(plotDictCount)
bundleTrans.setPlotDict(plotDictTrans)

plotDictCount={'logScale':True, 'cmap':cm.gray_r}
plotDictTrans={'logScale':False, 'cmap':cm.RdBu_r}
bundleCountWFDPlane.setPlotDict(plotDictCount)
bundleTransWFDPlane.setPlotDict(plotDictTrans)
bundleTransWFDPlane.setPlotFuncs(plotFuncs)
bundleCountWFDPlane.setPlotFuncs(plotFuncs)

bundleCount.plot()
bundleTrans.plot()

bundleCountWFDPlane.plot()
bundleTransWFDPlane.plot()

get_ipython().system(' pwd')



