import numpy as np
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt

#  LSST imports:
import lsst.afw.image as afw_image
from lsst.afw.table import (SourceTable, SourceCatalog)
from lsst.meas.base import SingleFrameMeasurementConfig
from lsst.meas.algorithms import (SourceDetectionConfig, SourceDetectionTask)

import diffimTests as dit

import os
import sys
import lsst.utils
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.ip.diffim as ipDiffim
import lsst.pex.logging as logging
import lsst.ip.diffim.diffimTools as diffimTools

verbosity = 5
logging.Trace_setVerbosity('lsst.ip.diffim', verbosity)
logging.Trace_setVerbosity('ImagePsfMatchTask', verbosity)

display = False

# D = I - (K.x.T + bg)

config    = ipDiffim.ImagePsfMatchTask.ConfigClass()
config.kernel.name = "AL"
subconfig = config.kernel.active

# Some of the tests are sensitive to the centroids returned by
# "stdev" vs "pixel_stdev"
subconfig.detectionConfig.detThresholdType = "stdev"

# Impacts some of the test values
subconfig.constantVarianceWeighting = True

try:
    defDataDir = lsst.utils.getPackageDir('afwdata')    
    defTemplatePath = os.path.join(defDataDir, "DC3a-Sim", "sci", "v5-e0",
                                   "v5-e0-c011-a00.sci.fits")
    defSciencePath = os.path.join(defDataDir, "DC3a-Sim", "sci", "v26-e0",
                                  "v26-e0-c011-a00.sci.fits")

    scienceImage   = afwImage.ExposureF(defSciencePath)
    templateImage  = afwImage.ExposureF(defTemplatePath)

    bgConfig = subconfig.afwBackgroundConfig
    bgConfig.useApprox = False
    bgConfig.binSize = 512
    diffimTools.backgroundSubtract(bgConfig,
                                   [templateImage.getMaskedImage(),
                                    scienceImage.getMaskedImage()])

    offset   = 1500
    bbox     = afwGeom.Box2I(afwGeom.Point2I(0, offset),
                                  afwGeom.Point2I(511, 2046))
    subconfig.spatialKernelOrder = 1
    subconfig.spatialBgOrder = 0

    # Take a stab at a PSF.  This is needed to get the KernelCandidateList if you don't provide one.
    ksize  = 21
    sigma = 2.0
    psf = measAlg.DoubleGaussianPsf(ksize, ksize, sigma)
    scienceImage.setPsf(psf)
except Exception:
    print >> sys.stderr, "Warning: afwdata is not set up"
    defDataDir = None

subconfig.fitForBackground = True

print bbox
templateSubImage = afwImage.ExposureF(templateImage, bbox)
scienceSubImage  = afwImage.ExposureF(scienceImage, bbox)

subconfig.spatialModelType = 'chebyshev1'
psfmatch1 = ipDiffim.ImagePsfMatchTask(config=config)
results1 = psfmatch1.subtractExposures(templateSubImage, scienceSubImage, doWarping = True)
spatialKernel1      = results1.psfMatchingKernel
backgroundModel1    = results1.backgroundModel

subconfig.spatialModelType = 'polynomial'
psfmatch2 = ipDiffim.ImagePsfMatchTask(config=config)
results2 = psfmatch2.subtractExposures(templateSubImage, scienceSubImage, doWarping = True)
spatialKernel2      = results2.psfMatchingKernel
backgroundModel2    = results2.backgroundModel

print spatialKernel1.getSpatialFunctionList()[0].toString()

# First order term has zero spatial variation and sum = kernel sum
kp1par0 = spatialKernel1.getSpatialFunctionList()[0].getParameters()
kp2par0 = spatialKernel2.getSpatialFunctionList()[0].getParameters()
#self.assertAlmostEqual(kp1par0[0], kp2par0[0], delta=1e-5)
print kp1par0[0], kp2par0[0]

# More improtant is the kernel needs to be then same when realized at a coordinate
kim1 = afwImage.ImageD(spatialKernel1.getDimensions())
kim2 = afwImage.ImageD(spatialKernel2.getDimensions())
ksum1 = spatialKernel1.computeImage(kim1, False, 0.0, 0.0)
ksum2 = spatialKernel2.computeImage(kim2, False, 0.0, 0.0)
#self.assertAlmostEqual(ksum1, ksum2, delta=1e-5)
print ksum1, ksum2
#for y in range(kim1.getHeight()):
#    for x in range(kim1.getHeight()):
#        self.assertAlmostEqual(kim1.get(x, y), kim2.get(x, y), delta=1e-1)

# Nterms (zeroth order)
print backgroundModel1.getNParameters(), backgroundModel2.getNParameters()

# Zero value in function
print backgroundModel1.getParameters()[0], backgroundModel2.getParameters()[0]

# Function evaluates to zero
print backgroundModel1(0, 0), backgroundModel2(0, 0)

# Spatially...
print backgroundModel1(10, 10), backgroundModel2(10, 10)

dit.plotImageGrid((templateSubImage.getMaskedImage().getArrays()[0], 
               scienceSubImage.getMaskedImage().getArrays()[0],
              results1.matchedImage.getArrays()[0], 
               results1.subtractedExposure.getMaskedImage().getArrays()[0]), 
              imScale=5., clim=(-20,20), 
              titles=['template', 'science', 'matched', 'diffim'])

# Note this was computed at coord (0,0) which is way off-center from the input images.
dit.plotImageGrid((kim1.getArray(), kim2.getArray()), titles=['kim1', 'kim2'])

xcen = (bbox.getBeginX() + bbox.getEndX()) / 2.
ycen = (bbox.getBeginY() + bbox.getEndY()) / 2.
print xcen, ycen
ksum1 = spatialKernel1.computeImage(kim1, True, xcen, ycen)
ksum2 = spatialKernel2.computeImage(kim2, True, xcen, ycen)
print ksum1, ksum2
dit.plotImageGrid((kim1.getArray(), kim2.getArray()), titles=['kim1', 'kim2'])

import scipy.stats
reload(dit)

sig1 = templateSubImage.getMaskedImage().getVariance().getArray()
print scipy.stats.describe(sig1, None)
sig2 = scienceSubImage.getMaskedImage().getVariance().getArray()
print scipy.stats.describe(sig2, None)
print dit.computeClippedImageStats(sig1), dit.computeClippedImageStats(templateSubImage.getMaskedImage().getImage().getArray())
print dit.computeClippedImageStats(sig2), dit.computeClippedImageStats(scienceSubImage.getMaskedImage().getImage().getArray())
sig1squared, _ = dit.computeClippedImageStats(sig1)
sig2squared, _ = dit.computeClippedImageStats(sig2)
print np.sqrt(sig1squared), np.sqrt(sig2squared)
dit.plotImageGrid((sig1, sig1), imScale=5., clim=(600,700))

corrKernel1 = dit.computeCorrectionKernelALZC(kim1.getArray(), sig1=np.sqrt(sig1squared), sig2=np.sqrt(sig2squared))
corrKernel2 = dit.computeCorrectionKernelALZC(kim2.getArray(), sig1=np.sqrt(sig1squared), sig2=np.sqrt(sig2squared))
dit.plotImageGrid((corrKernel1, corrKernel2), imScale=3., clim=(-0.01,0.01), cmap=None)

import scipy.ndimage.filters
diffim = results1.subtractedExposure.getMaskedImage().getArrays()[0]
pci = scipy.ndimage.filters.convolve(diffim, corrKernel1, mode='constant')

dit.plotImageGrid((diffim, pci), 
              imScale=5., clim=(-50,50), 
              titles=['diffim', 'corrected'])

import pandas as pd

print dit.computeClippedImageStats(pci[~np.isnan(pci)])
print dit.computeClippedImageStats(diffim[~np.isnan(diffim)])
print np.sqrt(sig1squared + sig2squared)

df = pd.DataFrame({'corr': pci.flatten(), 'orig': diffim.flatten()})
df.plot.hist(alpha=0.5, bins=2000)
#plt.xlim(-0.5, 0.5)

cov1 = dit.computePixelCovariance(np.nan_to_num(diffim))
cov2 = dit.computePixelCovariance(np.nan_to_num(pci))
dit.plotImageGrid((cov1, cov2), imScale=4., clim=(0, 0.05), cmap=None)

diffim2 = dit.performAlardLupton(templateSubImage.getMaskedImage().getImage().getArray(),
                                 scienceSubImage.getMaskedImage().getImage().getArray())

psf1 = templateSubImage.getPsf()
print psf1
psf2 = scienceSubImage.getPsf()
print psf2

## For fun let's to straight-out ZOGY as well...
if False:
    zogyD = dit.performZOGY(templateSubImage.getMaskedImage().getImage().getArray(), 
                        scienceSubImage.getMaskedImage().getImage().getArray(), 
                        templateSubImage.getPsf().computeImage().getArray(), 
                        scienceSubImage.getPsf().computeImage().getArray(), 
                        sig1=np.sqrt(sig1squared), sig2=np.sqrt(sig2squared))



