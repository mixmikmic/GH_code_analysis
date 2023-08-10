get_ipython().magic('matplotlib inline')

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

import copy

import shutil

from matplotlib.colors import LogNorm

# healpy 
import healpy as hp

# Loading pre-ran metric results
import lsst.sims.maf.metricBundles as mb

# import statements from the plot-handling notebook
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.plots as plots
from lsst.sims.maf.metricBundles import MetricBundle, MetricBundleGroup, makeBundlesDictFromList

# Copied file sims_utils/python/lsst/sims/utils/healpyUtils.py into the working directory
import healpyUtils

# Some stuff we'll need to manipulate coordinates easily
from astropy.coordinates import SkyCoord

# uncomment to follow the convention of the repository. 
# Note that then the outputs will be submittable to the repository.

#topDir = '/Users/clarkson/Progs/Projects/lsstScratchWIC'
#metricDir = '%s/data/metricOutputs' % (topDir)

# Otherwise we assume this is already being run from the directory 
# holding pre-computed metric directories.
metricDir=os.getcwd()

# update - for my system
metricDir='/Users/clarkson/Data/LSST/OpSimRuns/opsim20160411'

# for sanity when using, set a string for which quantity we're talking about?
sMetric = 'properMotion'
#sMetric = 'parallax'

# construct the path and filename from our choice of input
#sOpSim = 'minion_1020'
sOpSim = 'minion_1016'
#sOpSim = 'astro_lsst_01_1004'

# some strings to build up the run and directory
sSQL = 'night_lt_10000'
sSelDir = 'nside64_ugrizy_n10000_r21p0_lims'

# Let's set metricDir to a particular place...
#inputDir='%s/metricEvals/minion_1016_nside64_ugrizy_n10000_r21p0_lims' % (metricDir)
#inputFil='minion_1016_%s_night_lt_10000_HEAL.npz' % (sMetric)

inputDir='%s/metricEvals/%s_%s' % (metricDir, sOpSim, sSelDir)
inputFil='%s_%s_%s_HEAL.npz' % (sOpSim, sMetric, sSQL)

print inputDir
print inputFil

# what happens with ps1?
#inputDir='%s/metricEvals/minion_1020_nside64_ugrizy_n10000_r21p0_lims' % (metricDir)
#inputFil='minion_1020_%s_night_lt_10000_HEAL.npz' % (sMetric)

# check that we can read the path
inPath = '%s/%s' % (inputDir, inputFil)
print os.access(inPath, os.R_OK)

myBundle = mb.createEmptyMetricBundle()

myBundle.read(inPath)

print myBundle.metric.name

print myBundle.metric.colInfo

# Can we choose which plots to produce?
# help(myBundle.setPlotFuncs)   ## Yes but is not immediately obvious to me...
print myBundle.plotFuncs  # can we change this?
plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()] #, plots.HealpixSDSSSkyMap()]
myBundle.setPlotFuncs(plotFuncs)

# standardise the cmap throughout
cMap = cm.cubehelix_r

# aha - so we CAN just plot by calling the method!! Niiice... OK now see if we can change any parameters
#if os.access(sFigNam, os.R_OK):
#    os.remove(sFigNam)

thisPlotDict = {'logScale':True, 'cmap':cMap}# cm.RdBu_r}

vmin = np.min(myBundle.metricValues)
vmax = np.max(myBundle.metricValues)

thisPlotDict['xMax'] = vmax
thisPlotDict['xMin'] = vmin

myBundle.setPlotDict(thisPlotDict)
myBundle.plot(savefig=True)

# until I work out how to change the output plot name, use os.copy to produce a file copy 
sFigNam='thumb.%s_SkyMap.png' % (inputFil.split('.')[0])

if os.access(sFigNam, os.R_OK):
        
    sOut = 'MW_Astrom_FoM_%s_%s_all_skymap.png' % (sMetric, sOpSim)
    shutil.copy(sFigNam, sOut)

npix = myBundle.metricValues.size
nside = hp.npix2nside(npix)
print npix, nside

ra, dec = healpyUtils.hpid2RaDec(nside, np.arange(npix))
print np.shape(ra)
print np.shape(dec)
print np.min(ra), np.max(ra)
print np.min(dec), np.max(dec)

# feed these RA, DEC values into an astropy coords object
cc = SkyCoord(ra=np.copy(ra), dec=np.copy(dec), frame='fk5', unit='deg')

np.shape(myBundle.metricValues)

def getAvoidanceLatitudes(galL, peakDeg=10., taperDeg=80., constWidth=5.):
    
    """Returns the (positive) GP avoidance region for input galactic longitude"""
    
    # The following is adapted from spatialPlotters.py in sims_maf, method _plot_mwZone
    
    # astropy uses 0 <= glon <= 2pi, so we shift the input values accordingly.
    galL_use = np.copy(galL)
    gSec = np.where(galL_use > np.pi)
    galL_use[gSec] -= 2.0 * np.pi
    
    peakWidth=np.radians(peakDeg)
    taperLength=np.radians(taperDeg)
    val = peakWidth * np.cos(galL_use / taperLength * np.pi / 2.)
    
    # Remove the cosine peak at anticenter
    gFar = np.where(np.abs(galL_use) > taperLength)[0]
    val[gFar] = 0.

    val += np.radians(constWidth)
    
    return val

# Chart the avoidance regions for the plane:
step = 0.02
galL = np.arange(-np.pi, np.pi + step / 2., step) + np.pi
galB1 = getAvoidanceLatitudes(galL, 8., 80., 0.)
galB2 = 0. - galB1

# all in degrees
r2deg = 180./np.pi
galL *= r2deg
galB1 *= r2deg
galB2 *= r2deg

plt.figure(1, figsize=(12,4))
plt.clf()
plt.subplot(121)
plt.scatter(cc.ra, cc.dec,             c=myBundle.metricValues, edgecolor='none', s=2,            cmap=cMap, norm=LogNorm())
plt.title('Equatorial')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.colorbar()

plt.subplot(122)
plt.scatter(cc.galactic.l, cc.galactic.b,             c=myBundle.metricValues, edgecolor='none', s=2,            cmap=cMap, norm=LogNorm())

plt.plot(galL, galB1)
plt.plot(galL, galB2)

plt.title('Galactic')
plt.xlabel('l')
plt.ylabel('b')
plt.colorbar()

# now try selecting spatially for (region, ~region)
# mVals = myBundle.metricValues  # view for convenience

r2deg = 180./np.pi
gLon = np.asarray(cc.galactic.l)
gLat = np.asarray(cc.galactic.b)

# Set a conservative "plane" region 
latAvoidPos = getAvoidanceLatitudes(gLon*np.pi/180., constWidth=0.0, peakDeg=7.)
latAvoidPos *= r2deg

# Perform the selection
bPln = (gLat < latAvoidPos) & (gLat > 0.-latAvoidPos)

# For the "main" survey region, try excising the south polar cap for 
# a fairer comparison, and allow a bit of slop in the avoidance region
capDecMax = -60.

bNonCap = (np.asarray(cc.dec) > capDecMax)
latNonPlnPos = getAvoidanceLatitudes(gLon*np.pi/180., constWidth=0., peakDeg=10., taperDeg=115.)
latNonPlnPos *= r2deg

bNotPln = (gLat > latNonPlnPos) | (gLat < 0.-latNonPlnPos)

bAway = (bNotPln) & (bNonCap) #& (~bPln)

print np.size(np.where(bPln))
print np.size(np.where(bAway))

# set minmax ranges for plots
vmin = np.min(myBundle.metricValues)
vmax = np.max(myBundle.metricValues)

# Let's try the plot again, this time sticking with galactics and selecting out our regions.
plt.figure(2, figsize=(12,4))
plt.clf()
plt.subplot(121)
plt.scatter(cc.galactic.l[bPln], cc.galactic.b[bPln],             c=myBundle.metricValues[bPln], edgecolor='none', s=2,            cmap=cMap, norm=LogNorm(),            vmin=vmin, vmax=vmax)

plt.xlim(0, 400)
plt.ylim(-100,100)
plt.title('Galactic, inner-plane only')
plt.colorbar()

plt.subplot(122)
plt.scatter(cc.galactic.l[bAway], cc.galactic.b[bAway],             c=myBundle.metricValues[bAway], edgecolor='none', s=2,            cmap=cMap, norm=LogNorm(),            vmin=vmin, vmax=vmax)

plt.xlim(0, 400)
plt.ylim(-100,100)
plt.title('Galactic, outside the ROI')
plt.colorbar()

# in the region of interest...
print "Plane: median %.2e" % (np.median(myBundle.metricValues[bPln]))
print "Plane: stddev %.2e" % (np.std(myBundle.metricValues[bPln]))
#print myBundle.metricValues[bPln]

# NOT in the region of interest...
print "Outside avoidance region: median %.2e" % (np.median(myBundle.metricValues[bAway]))
print "Outside avoidance region: stddev %.3e" % (np.std(myBundle.metricValues[bAway]))
#print myBundle.metricValues[~bPln]

# write these to a human-readable text file...
sHuman = 'regSel_%s.txt' % (inputFil.split('.')[0])
wObj = open(sHuman, 'w')
wObj.write('########### \n')
wObj.write('# Result of TrySelectingMetricSpatially.ipynb \n')
wObj.write('# Input file: %s \n' % (inputFil))
wObj.write('# Input directory: %s \n' % (inputDir))
wObj.write('# Quantity %s \n' % (sMetric))
wObj.write('########### \n')

wObj.write("Inner plane: median %.2e \n" % (np.median(myBundle.metricValues[bPln])) )
wObj.write("Inner plane: stddev %.2e \n" % (np.std(myBundle.metricValues[bPln])) )
wObj.write('### \n')
wObj.write("Outside avoidance region: median %.2e \n" % (np.median(myBundle.metricValues[bAway])) )
wObj.write("Outside avoidance region: stddev %.3e \n" % (np.std(myBundle.metricValues[bAway])) )

wObj.close()

plt.figure(3, figsize=(10,4))
plt.clf()

dum = plt.hist(np.log10(myBundle.metricValues[bPln]), bins=250,                alpha=0.5, log=True, color='b',               histtype='step', normed=False)
dum = plt.hist(np.log10(myBundle.metricValues[bAway]), bins=250,                alpha=0.5, log=True, color='r',               histtype='step', normed=False)

plt.xlabel('log10(metric)')
plt.title('Inside (blue) and outside (red) ROI')

print myBundle.metricValues[0:5]
print myBundle.metricValues[-5:-1]

myBundleNoPlane = copy.deepcopy(myBundle)
myBundleAvoid = copy.deepcopy(myBundle)

myBundleAvoid.metricValues.mask[~bPln] = True

# set some plot characteristics of interest
sepPlotDict = copy.deepcopy(thisPlotDict)
sepPlotDict['cmap'] = cMap
sepPlotDict['xMax'] = vmax
sepPlotDict['xMin'] = vmin

myBundleAvoid.setPlotDict(sepPlotDict)
myBundleAvoid.plot(savefig=True)

if os.access(sFigNam, os.R_OK):
    sOut = 'MW_Astrom_FoM_%s_%s_plane_skymap.png' % (sMetric, sOpSim)
    shutil.copy(sFigNam, sOut)

myBundleNoPlane.metricValues.mask[~bAway] = True

myBundleNoPlane.setPlotDict(sepPlotDict)
myBundleNoPlane.plot(savefig=True)

if os.access(sFigNam, os.R_OK):
    sOut = 'MW_Astrom_FoM_%s_%s_nonPlane_skymap.png' % (sMetric, sOpSim)
    shutil.copy(sFigNam, sOut)

get_ipython().system(' pwd')



