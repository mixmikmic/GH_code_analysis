get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

## borrowed from ../notebooks/17....ipynb...

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
import lsst.ip.diffim.diffimTools as diffimTools

display = False
print ipDiffim.__path__

# dont bother with CCD 1, the template doesn't have a PSF.
template = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/WARPEDTEMPLATE.fits')
science = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/processed_15A38/0411033/calexp/calexp-0411033_03.fits')
template_unwarped = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/processed_15A38/0410927/calexp/calexp-0410927_03.fits')

import diffimTests as dit
reload(dit);

def ga(exposure):
    return exposure.getMaskedImage().getImage().getArray()
def gv(exposure):
    return exposure.getMaskedImage().getVariance().getArray()
def gm(exposure):
    return exposure.getMaskedImage().getMask().getArray()

print dit.computeClippedImageStats(ga(template))
print dit.computeClippedImageStats(gv(template))
print dit.computeClippedImageStats(ga(science))
print dit.computeClippedImageStats(gv(science))
print dit.computeClippedImageStats(ga(template_unwarped))
print dit.computeClippedImageStats(gv(template_unwarped))

#ga(template)[:, :] -= dit.computeClippedImageStats(ga(template)).mean
#ga(science)[:, :] -= dit.computeClippedImageStats(ga(science)).mean
#print dit.computeClippedImageStats(ga(template))
#print dit.computeClippedImageStats(ga(science))

dit.plotImageGrid((ga(template), ga(science)), imScale=16, titles=['Template', 'Science'])

dit.plotImageGrid((science.getPsf().computeKernelImage().getArray(),
                   template.getPsf().computeKernelImage().getArray()))

dit.plotImageGrid((gv(template), gv(science)), titles=['Template (var)', 'Science (var)'],
                  imScale=16)

config = ipDiffim.ZogyImagePsfMatchConfig()
task = ipDiffim.ZogyImagePsfMatchTask(config=config)

res1 = task.subtractExposures(template_unwarped, science, spatiallyVarying=False)
print "HERE1"
res2 = task.subtractExposures(template_unwarped, science, spatiallyVarying=True)
print "HERE2"

meta = science.getMetadata()
meta.getOrderedNames()
print meta.get(u'BGMEAN')
print template_unwarped.getMetadata().get(u'BGMEAN')

alDiffim = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/diffim_15A38_g/deepDiff/v411033/diffexp-03.fits')

D1 = res1.subtractedExposure
D2 = res2.subtractedExposure
print dit.computeClippedImageStats(ga(D1))
print dit.computeClippedImageStats(ga(D2))
print dit.computeClippedImageStats(ga(alDiffim))
print dit.computeClippedImageStats(gv(D1))
print dit.computeClippedImageStats(gv(D2))
dit.plotImageGrid((ga(D2), ga(alDiffim)), imScale=16, titles=['ZOGY', 'AL'])  # gv(D)

dit.plotImageGrid((ga(template)[3000:3250, 1500:1750], ga(science)[3000:3250, 1500:1750], 
                   ga(D1)[3000:3250, 1500:1750], ga(alDiffim)[3000:3250, 1500:1750]), imScale=4, 
                  extent=(1500, 3000, 1750, 3250),
                 titles=['template', 'science', 'ZOGY', 'AL'])  # gv(D)

dit.plotImageGrid((ga(template)[1000:1250, 900:1150], ga(science)[1000:1250, 900:1150], 
                   ga(D1)[1000:1250, 900:1150], ga(alDiffim)[1000:1250, 900:1150]), imScale=4, 
                  extent=(900, 1000, 1150, 1250),
                 titles=['template', 'science', 'ZOGY', 'AL'])  # gv(D)

dit.plotImageGrid((
    template_unwarped.getPsf().computeKernelImage().getArray(),
    science.getPsf().computeKernelImage().getArray(),
    D1.getPsf().computeKernelImage(position=afwGeom.Point2D(500,500)).getArray(), 
    D2.getPsf().computeKernelImage(position=afwGeom.Point2D(500,500)).getArray(), 
    alDiffim.getPsf().computeKernelImage().getArray()),
    titles=['template', 'science', 'ZOGY1', 'ZOGY2', 'AL'], clim=(-0.001,0.001), imScale=2.5)

S = task.subtractExposures(template_unwarped, science, spatiallyVarying=False, doPreConvolve=True)

dit.plotImageGrid((ga(D1), ga(S.subtractedExposure)), imScale=16, titles=['ZOGY D', 'ZOGY Scorr'])  # gv(D)

res3 = task.subtractExposures(template_unwarped, science, inImageSpace=True, spatiallyVarying=True)
D3 = res3.subtractedExposure

dit.plotImageGrid((ga(D1), ga(D3)), imScale=16, titles=['ZOGY (F)', 'ZOGY (R)'])  # gv(D)







zogyDiffim = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/diffim_15A38_ZOGY_g/deepDiff/v411033/diffexp-03.fits')
zogyDiffim2 = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/diffim_15A38_ZOGYspatial_g/deepDiff/v411033/diffexp-03.fits')

print dit.computeClippedImageStats(ga(D1))
print dit.computeClippedImageStats(ga(zogyDiffim))
print dit.computeClippedImageStats(ga(zogyDiffim2))
print dit.computeClippedImageStats(ga(alDiffim))
print ''
print dit.computeClippedImageStats(gv(D1))
print dit.computeClippedImageStats(gv(zogyDiffim))
print dit.computeClippedImageStats(gv(zogyDiffim2))
print dit.computeClippedImageStats(gv(alDiffim))

dit.plotImageGrid((ga(D1), ga(zogyDiffim)), imScale=16, titles=['ZOGY (F)', 'ZOGY (imageDiff)'])  # gv(D)

dit.plotImageGrid((ga(zogyDiffim), ga(zogyDiffim2)), imScale=16, titles=['ZOGY (const)', 'ZOGY (spatiallyVarying=True)'])  # gv(D)

psf = zogyDiffim2.getPsf()
#print psf.computeKernelImage(position=afwGeom.Point2D(500,500)).getArray()
shape = psf.computeShape(position=afwGeom.Point2D(500,500))
print shape
#psf.computeImage().getArray()

import lsst.pipe.base as pipeBase
x = pipeBase.Struct(x=1, y=2)
d = x.getDict()
d.has_key('x')
print x

arr = zogyDiffim2.getMaskedImage().getImage().getArray()[70:80,70:80]
#print arr
arr2 = np.fft.fft2(arr)
#print arr2

# dont bother with CCD 1, the template doesn't have a PSF.
template = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/WARPEDTEMPLATE.fits')
template_unwarped = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/processed_15A38/0410927/calexp/calexp-0410927_03.fits')
science = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/processed_15A38/0411033/calexp/calexp-0411033_03.fits')
zogyDiffim = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/diffim_15A38_ZOGY_g/deepDiff/v411033/diffexp-03.fits')
zogyDiffim2 = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/diffim_15A38_ZOGYspatial_g/deepDiff/v411033/diffexp-03.fits')

#template = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/WARPEDTEMPLATE.fits')
#science = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/processed_15A38/0411033/calexp/calexp-0411033_01.fits')
#zogyDiffim = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/diffim_15A38_ZOGY_g/deepDiff/v411033/diffexp-01.fits')
#zogyDiffim2 = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/diffim_15A38_ZOGYspatial_g/deepDiff/v411033/diffexp-01.fits')

print dit.computeClippedImageStats(ga(template_unwarped))
print dit.computeClippedImageStats(gv(template_unwarped))
print dit.computeClippedImageStats(ga(science))
print dit.computeClippedImageStats(gv(science))

print template_unwarped.getMetadata().get(u'BGMEAN')
print science.getMetadata().get(u'BGMEAN')

print dit.computeClippedImageStats(ga(template))
print dit.computeClippedImageStats(gv(template))

slice = np.index_exp[50:250,50:250]

dit.plotImageGrid((ga(template)[slice], ga(science)[slice],
                   ga(zogyDiffim)[slice], ga(zogyDiffim2)[slice]), 
    imScale=2, titles=['Template', 'Science', 'ZOGY (const)', 'ZOGY (spatiallyVarying=True)'])

dit.plotImageGrid((gm(template)[slice], gm(science)[slice],
                   gm(zogyDiffim)[slice], gm(zogyDiffim2)[slice]), 
    imScale=2, titles=['Template', 'Science', 'ZOGY (const)', 'ZOGY (spatiallyVarying=True)'],
                 clim=(0,10))

m = zogyDiffim.getMaskedImage().getMask().clone()
print m.getMaskPlaneDict()
print science.getMaskedImage().getMask().getMaskPlaneDict()
print type(m)
badmask = m.getPlaneBitMask(['UNMASKEDNAN', 'NO_DATA', 'BAD', 'EDGE', 'SUSPECT', 'CR', 'SAT'])
print np.mean((m.getArray() & badmask) != 0)
m |= template.getMaskedImage().getMask()
m |= science.getMaskedImage().getMask()
dit.plotImageGrid((gm(zogyDiffim2)[slice], m.getArray()[50:250,50:250],))

m = zogyDiffim.getMaskedImage().getMask().clone()
print type(m), isinstance(m, afwImage.mask.MaskU)
print type(m.getArray()), isinstance(m.getArray(), np.ndarray)
def getMask(m, names):
    badmask = m.getPlaneBitMask(names)
    return (m.getArray() & badmask) != 0
#names = ['INTRP', 'UNMASKEDNAN', 'NO_DATA', 'BAD', 'EDGE', 'SUSPECT', 'CR', 'SAT']
names = zogyDiffim.getMaskedImage().getMask().getMaskPlaneDict().keys()
mplanes = [getMask(m, [name])[slice] for name in names]
mplanes.append(getMask(m, names)[slice])
names.append('ALL')
dit.plotImageGrid((mplanes), titles=names, clim=(0,1))

m = zogyDiffim.getMaskedImage().getMask().clone()
detectedMask = getMask(m, ['DETECTED'])[slice]
badMask = getMask(m, ['BAD'])[slice]
dit.plotImageGrid((ga(template)[slice], ga(science)[slice],
                   ga(zogyDiffim)[slice], ga(zogyDiffim2)[slice]), 
    imScale=2, titles=['Template', 'Science', 'ZOGY (const)', 'ZOGY (spatiallyVarying=True)'],
                 #masks={2: ({'mask': m, 'maskName': 'DETECTED', 'cmap': 'Blues'},
                 #           {'mask': m, 'maskName': 'BAD',      'cmap': 'Reds'})})
                 masks={2: ({'mask': detectedMask, 'cmap': 'Blues'},
                            {'mask': badMask, 'cmap': 'Reds'})})

# Look at detection masks. CCD 3 didn't work for detection, so use CCD 1 now.
zogyDiffim = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/diffim_15A38_ZOGY_g/deepDiff/v411033/diffexp-01.fits')
zogyDiffim2 = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/diffim_15A38_ZOGYspatial_g/deepDiff/v411033/diffexp-01.fits')

dit.plotImageGrid((zogyDiffim, zogyDiffim2), imScale=16,
                 masks={0:({'maskName': 'DETECTED', 'cmap': 'Reds'}, 
                           {'maskName': 'DETECTED_NEGATIVE', 'cmap': 'Greens'},
                           {'maskName': 'BAD', 'cmap': 'Blues'}),
                        1:({'maskName': 'DETECTED', 'cmap': 'Reds'}, 
                           {'maskName': 'DETECTED_NEGATIVE', 'cmap': 'Greens'},
                           {'maskName': 'BAD', 'cmap': 'Blues'})})

dit.plotImageGrid((template.getPsf().computeKernelImage(),
                  science.getPsf().computeKernelImage(),
                  zogyDiffim.getPsf().computeKernelImage(),
                  #zogyDiffim2.getPsf().computeKernelImage()
                  ))

m = zogyDiffim.getMaskedImage().getMask().clone()
names = ['INTRP', 'UNMASKEDNAN', 'NO_DATA', 'BAD', 'EDGE', 'SUSPECT', 'CR', 'SAT']
badMask = getMask(m, names)[slice]
badMask2 = getMask(m, names)

print dit.computeClippedImageStats(ga(zogyDiffim)[slice][badMask == 0])
print dit.computeClippedImageStats(ga(zogyDiffim2)[slice][badMask == 0])
print dit.computeClippedImageStats(ga(alDiffim)[slice][badMask == 0])
print dit.computeClippedImageStats(ga(template)[slice][badMask == 0])
print np.median(ga(template)[~np.isnan(ga(template)) & (badMask2 == 0)])
print dit.computeClippedImageStats(ga(science)[slice][badMask == 0])
print
print dit.computeClippedImageStats(gv(zogyDiffim)[slice][badMask == 0])
print dit.computeClippedImageStats(gv(zogyDiffim2)[slice][badMask == 0])
print dit.computeClippedImageStats(gv(alDiffim)[slice][badMask == 0])
print dit.computeClippedImageStats(gv(template)[slice][badMask == 0])
print dit.computeClippedImageStats(gv(science)[slice][badMask == 0])

dit.plotImageGrid((gv(zogyDiffim)[slice], gv(zogyDiffim2)[slice],
                  gv(alDiffim)[slice], badMask == 0))







diffimALdecorr = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/diffim_15A38_newDecorr_g/deepDiff/v411033/diffexp-03.fits')
diffimALdecorrSpatial = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/diffim_15A38_newDecorrSpatial_g/deepDiff/v411033/diffexp-03.fits')
gm(diffimALdecorr)[:, :] = gm(diffimALdecorrSpatial)

print dit.computeClippedImageStats(ga(diffimALdecorr))
print dit.computeClippedImageStats(ga(diffimALdecorrSpatial))
print dit.computeClippedImageStats(gv(diffimALdecorr))
print dit.computeClippedImageStats(gv(diffimALdecorrSpatial))

dit.plotImageGrid((diffimALdecorr, diffimALdecorrSpatial), 
                  imScale=16, titles=['AL (decorr, const)', 'AL (decorr, spatiallyVarying=True)'])  # gv(D)

zogyDiffim2 = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/diffim_15A38_ZOGYspatial_g/deepDiff/v411033/diffexp-01.fits')
m = zogyDiffim2.getMaskedImage().getMask()
def getMask(m, names):
    badmask = m.getPlaneBitMask(names)
    print badmask
    return (m.getArray() & badmask) != 0
msk = getMask(m, ['BAD', 'NO_DATA'])
print np.sum(msk)

dit.plotImageGrid((ga(zogyDiffim2), msk), imScale=16)



