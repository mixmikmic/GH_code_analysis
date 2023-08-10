import numpy as np
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt

#  LSST imports:
import lsst.afw.image as afwImage
from lsst.afw.table import (SourceTable, SourceCatalog)
from lsst.meas.base import SingleFrameMeasurementConfig
from lsst.meas.algorithms import (SourceDetectionConfig, SourceDetectionTask)

import diffimTests as dit
import lsst.ip.diffim.imageDecorrelation as id

reload(dit)
reload(id);

im1 = afwImage.ExposureF('diffexp-11.fits')
im2 = afwImage.ExposureF('diffexp-11-ALZC.fits') 

sig1 = im1.getMaskedImage().getVariance()
sig2 = im2.getMaskedImage().getVariance()
print id.computeClippedImageStats(sig1)
print id.computeClippedImageStats(sig2)

sig1squared, _, _ = id.computeClippedImageStats(sig1)
sig2squared, _, _ = id.computeClippedImageStats(sig2)
print np.sqrt(sig1squared), np.sqrt(sig2squared)

sig1 = im1.getMaskedImage().getImage()
sig2 = im2.getMaskedImage().getImage()
print id.computeClippedImageStats(sig1)
print id.computeClippedImageStats(sig2)

_, _, sig1squared = id.computeClippedImageStats(sig1)
_, _, sig2squared = id.computeClippedImageStats(sig2)
print np.sqrt(sig1squared), np.sqrt(sig2squared)

import scipy.stats

im1a = im1.getMaskedImage().getImage()
im2a = im2.getMaskedImage().getImage()
print 'UNCORRECTED:', id.computeClippedImageStats(im1a)
print 'CORRECTED:  ', id.computeClippedImageStats(im2a)
print 'EXPECTED:   ', np.sqrt(sig1squared + sig2squared)

im1a = im1a.getArray()[~(np.isnan(im1a.getArray())|np.isinf(im1a.getArray()))]
im2a = im2a.getArray()[~(np.isnan(im2a.getArray())|np.isinf(im2a.getArray()))]
im1a = im1a[im1a != 0.]
im2a = im2a[im2a != 0.]
im1a, _, _ = scipy.stats.sigmaclip(im1a)
im2a, _, _ = scipy.stats.sigmaclip(im2a)
print len(im1a), len(im2a)
im1a = im1a[:len(im2a)]

import pandas as pd
df = pd.DataFrame({'corr': im2a, 'orig': im1a})
df.plot.hist(alpha=0.5, bins=200)

im1a = im1.getMaskedImage().getImage().getArray()
im1a[np.isnan(im1a)|np.isinf(im1a)] = 0.
_, low, upp = scipy.stats.sigmaclip(im1a)
im1a[(im1a < low)|(im1a > upp)] = 0.
cov1 = dit.computePixelCovariance(im1a)

im2a = im2.getMaskedImage().getImage().getArray()
im2a[np.isnan(im2a)|np.isinf(im2a)] = 0.
_, low, upp = scipy.stats.sigmaclip(im2a)
im2a[(im2a < low)|(im2a > upp)] = 0.
cov2 = dit.computePixelCovariance(im2a)

dit.plotImageGrid((cov1, cov2), imScale=4., clim=(0, 0.1), cmap=None)

print im2.hasPsf()
im1_psf = im1.getPsf().computeImage().getArray()
im2_psf = im2.getPsf().computeImage().getArray()
print im1.getPsf().computeShape().getDeterminantRadius(), im2.getPsf().computeShape().getDeterminantRadius()
print im1_psf.sum(), im2_psf.sum()
print np.unravel_index(im1_psf.argmax(), im1_psf.shape), np.unravel_index(im2_psf.argmax(), im2_psf.shape)
dit.plotImageGrid((im1.getPsf().computeImage().getArray(), im2_psf, 
                   im1.getPsf().computeImage().getArray() - im2_psf), imScale=2., clim=(-0.01,0.01))
scipy.stats.describe(im1.getPsf().computeImage().getArray() - im2_psf, None)

im1a = im1.getMaskedImage().getVariance()
im2a = im2.getMaskedImage().getVariance()

print 'UNCORRECTED:', id.computeClippedImageStats(im1a)
print 'CORRECTED:  ', id.computeClippedImageStats(im2a)
print 'EXPECTED:   ', np.sqrt(sig1squared + sig2squared)

im1a = im1a.getArray()[~(np.isnan(im1a.getArray())|np.isinf(im1a.getArray()))]
im2a = im2a.getArray()[~(np.isnan(im2a.getArray())|np.isinf(im2a.getArray()))]
im1a = im1a[im1a != 0.]
im2a = im2a[im2a != 0.]
im1a, _, _ = scipy.stats.sigmaclip(im1a)
im2a, _, _ = scipy.stats.sigmaclip(im2a)
print len(im1a), len(im2a)
im1a = im1a[:len(im2a)]
print scipy.stats.describe(im1a)
print scipy.stats.describe(im2a)

import pandas as pd
df = pd.DataFrame({'corr': im2a, 'orig': im1a})
df.plot.hist(alpha=0.5, bins=200)



