import numpy as np
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import scipy.stats

#  LSST imports:
import lsst.afw.image as afwImage
from lsst.afw.table import (SourceTable, SourceCatalog)
from lsst.meas.base import SingleFrameMeasurementConfig
from lsst.meas.algorithms import (SourceDetectionConfig, SourceDetectionTask)

import diffimTests as dit
import lsst.ip.diffim.imageDecorrelation as id

reload(dit)
reload(id);

im1 = afwImage.ExposureF('diffexp-11.fits')   # uncorrected diffim
im2 = afwImage.ExposureF('decamDirTest/deepDiff/v289820/diffexp-11.fits')  # decorrelated diffim

# Also load the original science and template images
sciImg = afwImage.ExposureF('calexpDir_b1631/0289820/calexp/calexp-0289820_11.fits')
templImg = afwImage.ExposureF('calexpDir_b1631/0288976/calexp/calexp-0288976_11.fits')

print 'Stats of variance planes:'
print dit.computeClippedAfwStats(sciImg.getMaskedImage().getVariance(),
                                maskIm=sciImg.getMaskedImage().getMask())  # mean, std, var
print dit.computeClippedAfwStats(templImg.getMaskedImage().getVariance(),
                                maskIm=templImg.getMaskedImage().getMask())
print dit.computeClippedAfwStats(im1.getMaskedImage().getVariance(),
                                maskIm=im1.getMaskedImage().getMask())
print dit.computeClippedAfwStats(im2.getMaskedImage().getVariance(),
                                maskIm=im2.getMaskedImage().getMask())

print '\nStats of image pixels:'
print dit.computeClippedAfwStats(sciImg.getMaskedImage())
print dit.computeClippedAfwStats(templImg.getMaskedImage())
print dit.computeClippedAfwStats(im1.getMaskedImage())
print dit.computeClippedAfwStats(im2.getMaskedImage())

def getClippedPixels(pixels):
    pix = pixels[~(np.isnan(pixels)|np.isinf(pixels))]
    pix = pix[pix != 0.]
    pix, _, _ = scipy.stats.sigmaclip(pix)
    return pix

im1a = getClippedPixels(im1.getMaskedImage().getImage().getArray())
im2a = getClippedPixels(im2.getMaskedImage().getImage().getArray())
print len(im1a), len(im2a)
im1a = im1a[:len(im2a)]

import pandas as pd
df = pd.DataFrame({'corr': im2a, 'orig': im1a})
df.plot.hist(alpha=0.5, bins=200)

def computePixelCovariances(im):
    im1a = im.getMaskedImage().getImage().getArray()
    im1a[np.isnan(im1a)|np.isinf(im1a)] = 0.
    _, low, upp = scipy.stats.sigmaclip(im1a)
    im1a[(im1a < low)|(im1a > upp)] = 0.
    cov1, ratioOffDiag = dit.computePixelCovariance(im1a)
    return cov1, ratioOffDiag

cov1, ratioOffDiag = computePixelCovariances(im1)
print ratioOffDiag
cov2, ratioOffDiag = computePixelCovariances(im2)
print ratioOffDiag
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
imsa = sciImg.getMaskedImage().getVariance()
imta = templImg.getMaskedImage().getVariance()

tmp = sciImg.getMaskedImage().clone()
tmp -= templImg.getMaskedImage()
tmp = tmp.getVariance()

print 'SCIENCE:', dit.computeClippedAfwStats(imsa, maskIm=sciImg.getMaskedImage().getMask())
print 'TEMPLATE:  ', dit.computeClippedAfwStats(imta, maskIm=templImg.getMaskedImage().getMask())
print 'UNCORRECTED:', dit.computeClippedAfwStats(im1a, maskIm=im1.getMaskedImage().getMask())
print 'CORRECTED:  ', dit.computeClippedAfwStats(im2a, maskIm=im2.getMaskedImage().getMask())
tmp = imsa.clone(); tmp += imta
print 'EXPECTED:   ', dit.computeClippedAfwStats(tmp, maskIm=im1.getMaskedImage().getMask())

def getClippedPixels(pixels):
    pix = pixels[~(np.isnan(pixels)|np.isinf(pixels))]
    pix = pix[pix != 0.]
    pix, _, _ = scipy.stats.sigmaclip(pix)
    return pix

im1a = getClippedPixels(im1a.getArray())
im2a = getClippedPixels(im2a.getArray())
imsa = getClippedPixels(imsa.getArray())
imta = getClippedPixels(imta.getArray())
tmp = getClippedPixels(tmp.getArray())
print len(im1a), len(im2a), len(imsa), len(imta)
im1a = im1a[:len(im2a)]
imsa = imsa[:len(im2a)]
imta = imta[:len(im2a)]
tmp = tmp[:len(im2a)]
expected = tmp

import pandas as pd
df = pd.DataFrame({'corr': im2a, 'orig': im1a, 'science': imsa, 'template': imta, 'expected': expected})
df.plot.hist(alpha=0.5, bins=200)

import lsst.afw.geom as afwGeom

def computeVarianceBoxStats(im, doPixelsInstead=False):
    im1a = im.getMaskedImage().getVariance()
    if doPixelsInstead:
        im1a = im.getMaskedImage().getImage()
    im1m = im.getMaskedImage().getMask()
    box = im1a.getBBox()
    sz = im1a.getArray().shape
    boxSize = afwGeom.ExtentI(100, 100)
    boxes = [afwGeom.Box2I(afwGeom.PointI(x, y), boxSize) for y in np.linspace(0, sz[0]-101, 50, dtype=int)              for x in np.linspace(0, sz[1]-101, 50, dtype=int)]
    subims = [[afwImage.ImageF(im1a, box, afwImage.PARENT), afwImage.MaskU(im1m, box, afwImage.PARENT)] for box in boxes]
    return [dit.computeClippedAfwStats(sub[0], maskIm=sub[1]) for sub in subims]

# Plot means in variance planes...
tmp = computeVarianceBoxStats(im1)
im1stats = np.array([t[0] for t in tmp]); im1stats[im1stats>70] = 70; print im1stats.max(), im1stats.std()
tmp = computeVarianceBoxStats(im2)
im2stats = np.array([t[0] for t in tmp]); im2stats[im2stats>135] = 125; print im2stats.max(), im2stats.std()
tmp = computeVarianceBoxStats(sciImg)
scistats = np.array([t[0] for t in tmp]); print scistats.max(), scistats.std()
tmp = computeVarianceBoxStats(templImg)
templstats = np.array([t[0] for t in tmp]); print templstats.max(), templstats.std()

df = pd.DataFrame({'corr-55': im2stats-55., 'orig': im1stats, 'science': scistats, 'template': templstats})
#df = pd.DataFrame({'orig': im1stats, 'science': scistats, 'template': templstats})
df.plot.hist(alpha=0.5, bins=100)

import cPickle
import gzip
im1a = im1.getMaskedImage().getVariance()
bbox = im1a.getBBox()
spatialKernel = cPickle.load(gzip.GzipFile('spatialKernel.p.gz', 'rb'))

def getSpatialKernelImage(spatialKernel, xcen, ycen):
    kimg = afwImage.ImageD(spatialKernel.getDimensions())
    spatialKernel.computeImage(kimg, True, xcen, ycen)
    return kimg.getArray()

kimages = [getSpatialKernelImage(spatialKernel, xc, yc) for
           xc in np.linspace(bbox.getBeginX(), bbox.getEndX(), 10.) for
           yc in np.linspace(bbox.getBeginY(), bbox.getEndY(), 10.)]
print "HERE:", len(kimages)

titles = ['%d,%d' % (x,y) for           x in np.linspace(bbox.getBeginX(), bbox.getEndX(), 10.) for           y in np.linspace(bbox.getBeginY(), bbox.getEndY(), 10.)]
dit.plotImageGrid(kimages, imScale=2., clim=(0., 0.05), titles=titles)

from lsst.ip.diffim import DecorrelateALKernelTask
import lsst.ip.diffim as ipDiffim
print ipDiffim.__path__

_, _, var1 = dit.computeClippedAfwStats(sciImg.getMaskedImage().getVariance(),
                                maskIm=sciImg.getMaskedImage().getMask())  # mean, std, var
_, _, var2 = dit.computeClippedAfwStats(templImg.getMaskedImage().getVariance(),
                                maskIm=templImg.getMaskedImage().getMask())

kimages2 = [DecorrelateALKernelTask._fixOddKernel(kimg) for kimg in kimages]

corrKernels = [DecorrelateALKernelTask._computeDecorrelationKernel(kimg, svar=var1, tvar=var2)                for kimg in kimages]
corrKernels = [DecorrelateALKernelTask._fixEvenKernel(kimg) for kimg in corrKernels]
#print [np.max(k) for k in corrKernels]

titles = ['%d,%d' % (x,y) for           x in np.linspace(bbox.getBeginX(), bbox.getEndX(), 10.) for           y in np.linspace(bbox.getBeginY(), bbox.getEndY(), 10.)]
dit.plotImageGrid(corrKernels, imScale=2., clim=(-0.06, 0.06), titles=titles)

get_ipython().magic('timeit DecorrelateALKernelTask._computeDecorrelationKernel(kimages[0], svar=var1, tvar=var2)')

diffs = [np.sqrt(np.sum((k1 - k2)**2.)) for k1 in kimages for k2 in kimages]
coords = [[i, j] for i in range(len(kimages)) for j in range(len(kimages))]
print np.max(diffs), np.argmax(diffs), coords[np.argmax(diffs)]
coo = coords[np.argmax(diffs)]
dit.plotImageGrid([kimages[coo[0]], kimages[coo[1]]], imScale=2., clim=(0., 0.05), 
                  titles=[titles[coo[0]], titles[coo[1]]])

print corrKernels[coo[0]].max(), corrKernels[coo[1]].max()
dit.plotImageGrid([corrKernels[coo[0]], corrKernels[coo[1]]], imScale=2., clim=(-0.06, 0.06), 
                  titles=[titles[coo[0]], titles[coo[1]]])
print np.max([kc.max() for kc in corrKernels]), np.min([kc.max() for kc in corrKernels])

# Reload all of the images in case they were screwed up above
im1 = afwImage.ExposureF('diffexp-11.fits')   # uncorrected diffim
im2 = afwImage.ExposureF('decamDirTest/deepDiff/v289820/diffexp-11.fits')  # decorrelated diffim

# Also load the original science and template images
sciImg = afwImage.ExposureF('calexpDir_b1631/0289820/calexp/calexp-0289820_11.fits')
templImg = afwImage.ExposureF('calexpDir_b1631/0288976/calexp/calexp-0288976_11.fits')

spatialKernel = cPickle.load(gzip.GzipFile('spatialKernel.p.gz', 'rb'))

from lsst.ip.diffim import DecorrelateALKernelTask
task = DecorrelateALKernelTask()

def doDecorr(sciImg, templImg, im1, spatialKernel, xcen=None, ycen=None, svar=None, tvar=None):
    sci = sciImg.clone()
    templ = templImg.clone()
    diffim = im1.clone()
    task = DecorrelateALKernelTask()
    return task.run(sci, templ, diffim, spatialKernel, xcen=xcen, ycen=ycen, svar=svar, tvar=tvar)

decorrResult = doDecorr(sciImg, templImg, im1, spatialKernel)
print dit.computeClippedAfwStats(im2.getMaskedImage())
print dit.computeClippedAfwStats(decorrResult.correctedExposure.getMaskedImage())

print dit.computeClippedAfwStats(im2.getMaskedImage().getVariance(), 
                                 maskIm=im2.getMaskedImage().getMask())
print dit.computeClippedAfwStats(decorrResult.correctedExposure.getMaskedImage().getVariance(),
                                maskIm=decorrResult.correctedExposure.getMaskedImage().getMask())

decorrResults = [doDecorr(sciImg, templImg, im1, spatialKernel, xcen=xy[0], ycen=xy[1], svar=v2, tvar=v1) for     #xy in [(None, None)] for v1 in (58., None) for v2 in (64.5, None)]
    xy in [(1023,2047), (0,4094), (2046,0), (None, None)] for v1 in (60., 58., 62., None) for v2 in (62.7, 61.5, 64.5, None)]

print len(decorrResults)
print [(xy[0], xy[1], v2, v1) for     #xy in [(None, None)] for v1 in (58., None) for v2 in (64.5, None)]
    xy in [(1023,2047), (0,4094), (2046,0), (None, None)] for v1 in (60., 58., 62., None) for v2 in (62.7, 61.5, 64.5, None)]

reload(dit)
decorrStats = [dit.computeClippedAfwStats(res.correctedExposure.getMaskedImage().getVariance(),
                                maskIm=res.correctedExposure.getMaskedImage().getMask()) for res in decorrResults]

decorrVars = np.array([dc[0] for dc in decorrStats])
print len(decorrVars), decorrVars
print (decorrVars.max()-decorrVars.min())/decorrVars.mean(), decorrVars.std()/decorrVars.mean()
import pandas as pd
pd.DataFrame({'variances': decorrVars}).hist(bins=10)

decorrVarVars = np.array([dc[2] for dc in decorrStats])
print len(decorrVarVars), decorrVarVars
pd.DataFrame({'variancesOfVariances': decorrVarVars}).hist(bins=10)

tmp = {str(i): res.correctedExposure.getMaskedImage().getVariance().getArray()[res.correctedExposure.getMaskedImage().getMask().getArray() == 0]
                                for i, res in enumerate(decorrResults)}
print len(tmp), type(tmp['0'])
for k, t in tmp.items():
    t[t > 200] = np.nan

df = pd.DataFrame(tmp)
df.plot.hist(bins=200, alpha=0.3, legend=False)
plt.xlim(100, 150)

del tmp  # free up memory
del df

tmp = [computePixelCovariances(res.correctedExposure) for res in decorrResults]

offDiags = [t[1] for t in tmp]
pd.DataFrame({'offDiagonals': offDiags}).hist(bins=10)

covs = [t[0] for t in tmp]
#covs.append(cov2)  # original corrected diffim
#covs.append(cov1)  # uncorrected diffim
print len(covs)
dit.plotImageGrid(covs, imScale=2., clim=(0, 0.05), nrows_ncols=(8, 8), cmap=None)

import lsst.meas.algorithms as measAlg
import lsst.afw.table as afwTable

def doDetection(exposure, detectSigma=5.0, grow=2, verbose=False):
    #mask = exposure.getMaskedImage().getMask()
    #mask &= ~(mask.getPlaneBitMask('DETECTED') | mask.getPlaneBitMask('DETECTED_NEGATIVE'))
    #psf = exposure.getPsf()
    #ctr = afwGeom.Box2D(exposure.getBBox()).getCenter()
    #psfAttr = measAlg.PsfAttributes(psf, afwGeom.Point2I(ctr))
    #psfSigma = psfAttr.computeGaussianWidth(psfAttr.ADAPTIVE_MOMENT)
    #psfWidth = psf.computeShape().getDeterminantRadius()
    #print psfSigma, psfWidth
    
    config = measAlg.SourceDetectionConfig()
    config.thresholdPolarity = 'both'
    config.reEstimateBackground = False
    #config.nSigmaToGrow = psfWidth
    config.thresholdValue = detectSigma
    config.thresholdType = 'pixel_stdev'  # why are we using this instead of 'variance'?
    schema = afwTable.SourceTable.makeMinimalSchema()  
    task = measAlg.SourceDetectionTask(schema, config=config)
    table = afwTable.SourceTable.make(schema)
    results = task.run(table, exposure, doSmooth=True, clearMask=True) #, sigma=psfWidth)
    
    fpSet = results.fpSets.positive
    fpSet.merge(results.fpSets.negative, grow, grow, False) 
    sources = afwTable.SourceCatalog(table)
    fpSet.makeSources(sources)
    if verbose:
        print 'POSITIVE SOURCES:', results.fpSets.numPos
        print 'NEGATIVE SOURCES:', results.fpSets.numNeg
        print 'MERGED SOURCES:', len(sources)
    return sources

s = doDetection(im1, verbose=True)
s = doDetection(im2, verbose=True)
print;
s = doDetection(decorrResult.correctedExposure, verbose=True)
s = doDetection(decorrResults[-1].correctedExposure, verbose=True)  # last one should equal the 'original' decorr-ed one

print im2.getPsf().computeShape().getDeterminantRadius()
print decorrResult.correctedExposure.getPsf().computeShape().getDeterminantRadius()
print decorrResults[63].correctedExposure.getPsf().computeShape().getDeterminantRadius()

print dit.computeClippedAfwStats(im2.getMaskedImage())
print dit.computeClippedAfwStats(decorrResult.correctedExposure.getMaskedImage())
print dit.computeClippedAfwStats(decorrResults[-1].correctedExposure.getMaskedImage())

sources = [doDetection(res.correctedExposure) for res in decorrResults]
ns = np.array([len(s) for s in sources])
print ns
pd.DataFrame({'n_detected': ns}).hist(bins=6)



