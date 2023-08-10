topDir = '/Users/clarkson/Progs/Projects/lsstScratchWIC'   # wherever you put the repository

get_ipython().magic('matplotlib inline')

import matplotlib.pylab as plt
import time
import os

# Default is to assume the opsim databases are symlinked into the current directory.
opsimDir = os.getcwd()

# uncomment the following to set the directory on your system where the opsim runs are stored:
# opsimDir = '/Users/clarkson/Data/LSST/OpSimRuns/Runs_20151229/LocalCopies'

# uncomment to follow the convention of the repository. 
# Note that then the outputs will be submittable to the repository.

#topDir = '/Users/clarkson/Progs/Projects/lsstScratchWIC'
#os.chdir('%s/data/metricOutputs' % (topDir))

# First try importing needed things
from lsst.sims.maf.metrics import BaseMetric

import numpy as np

import mafContrib

from mafContrib import starcount as starcount

print starcount.starcount(200., -23., 1000, 2000)

from mafContrib import CountMetric

import healpy as hp

import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db

# Slicer will be required whatever we do
slicer = slicers.HealpixSlicer(nside=64)

metricCount = CountMetric(D1=1000, D2=2000)
metricList = [metricCount]

# Do our cargo-cult setup here...
runName1092 = 'ops2_1092'
sqlconstraint = 'filter = "r"'
bDict1092={}
for i,metric in enumerate(metricList):
    bDict1092[i] = metricBundles.MetricBundle(metric, slicer, sqlconstraint, 
                                          runName=runName1092)

opsdb1092 = db.OpsimDatabase(opsimDir + '/' + runName1092 + '_sqlite.db')
outDir1092 = 'TestDensity1092'
resultsDb1092 = db.ResultsDb(outDir=outDir1092)

tStart = time.time()
bgroup1092 = metricBundles.MetricBundleGroup(bDict1092, opsdb1092, outDir=outDir1092,                                              resultsDb=resultsDb1092)
bgroup1092.runAll()
bgroup1092.plotAll(closefigs=False)
tPost1092 = time.time()
print "Time spent on 1092: %.3e seconds" % (tPost1092 - tStart)

# hmm OK that last one failed on the index. This might illustrate a danger in hardcoding
# the index (if the slice passes information differently?) Let's follow the same approach,
# this time pulling the RA, DEC a little differently:

class AsCountMetric(BaseMetric):
    
    """Copy of Mike Lund's CountMetric, but pulling the RA and DEC a bit differently"""
    
    def __init__(self,**kwargs):
        
        self.D1=kwargs.pop('D1', 100)
        self.D2=kwargs.pop('D2', 1000)
        super(AsCountMetric, self).__init__(col=[], **kwargs)
        
    def run(self, dataSlice, slicePoint=None):
        sliceRA = np.degrees(slicePoint['ra'])
        sliceDEC = np.degrees(slicePoint['dec'])
        return starcount.starcount(sliceRA, sliceDEC, self.D1, self.D2)
    
        

# OK let's try that again, this time with the slightly altered CountMetric:
metricCount = AsCountMetric(D1=1000, D2=2000)
metricList = [metricCount]

runName1092 = 'ops2_1092'
sqlconstraint = 'filter = "r"'
bDict1092={}
for i,metric in enumerate(metricList):
    bDict1092[i] = metricBundles.MetricBundle(metric, slicer, sqlconstraint, 
                                          runName=runName1092)
opsdb1092 = db.OpsimDatabase(opsimDir + '/' + runName1092 + '_sqlite.db')
outDir1092 = 'TestDensity1092'
resultsDb1092 = db.ResultsDb(outDir=outDir1092)

tStart = time.time()
bgroup1092 = metricBundles.MetricBundleGroup(bDict1092, opsdb1092, outDir=outDir1092,                                              resultsDb=resultsDb1092)
bgroup1092.runAll()
bgroup1092.plotAll(closefigs=False)
tPost1092 = time.time()
print "Time spent on 1092: %.3e seconds" % (tPost1092 - tStart)

# Let's try different distance limits...
metricCount = AsCountMetric(D1=100., D2=9000.)
metricList = [metricCount]

runName1189 = 'enigma_1189'
sqlconstraint = 'filter = "r"'
bDict1189={}
for i,metric in enumerate(metricList):
    bDict1189[i] = metricBundles.MetricBundle(metric, slicer, sqlconstraint, 
                                          runName=runName1189)
opsdb1189 = db.OpsimDatabase(opsimDir + '/' + runName1189 + '_sqlite.db')
outDir1189 = 'TestDensity1189'
resultsDb1189 = db.ResultsDb(outDir=outDir1189)

tStart = time.time()
bgroup1189 = metricBundles.MetricBundleGroup(bDict1189, opsdb1189, outDir=outDir1189,                                              resultsDb=resultsDb1189)
bgroup1189.runAll()
bgroup1189.plotAll(closefigs=False)
tPost1189 = time.time()
print "Time spent on 1092: %.3e seconds" % (tPost1189 - tStart)

#hmm, ok... Now let's try out to 150 kpc, so that we include the halo as well...
dist1=10.
dist2=150000.

metricCount = AsCountMetric(D1=dist1, D2=dist2)
metricList = [metricCount]

runName1189 = 'enigma_1189'
sqlconstraint = 'filter = "r"'
bDict1189={}
plotDict={'logScale':True}
for i,metric in enumerate(metricList):
    bDict1189[i] = metricBundles.MetricBundle(metric, slicer, sqlconstraint, 
                                          runName=runName1189, plotDict=plotDict)

opsdb1189 = db.OpsimDatabase(opsimDir + '/' + runName1189 + '_sqlite.db')
outDir1189 = 'TestDensity1189'
resultsDb1189 = db.ResultsDb(outDir=outDir1189)

tStart = time.time()
bgroup1189 = metricBundles.MetricBundleGroup(bDict1189, opsdb1189, outDir=outDir1189,                                              resultsDb=resultsDb1189)

bgroup1189.runAll()
bgroup1189.plotAll(closefigs=False)
tPost1189 = time.time()
print "Time spent on 1189: %.3e seconds" % (tPost1189 - tStart)



