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

# dont bother with CCD 1, the template doesn't have a PSF.
template = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/WARPEDTEMPLATE.fits')
science = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/processed_15A38/0411033/calexp/calexp-0411033_03.fits')

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

ga(template)[:, :] -= dit.computeClippedImageStats(ga(template)).mean
ga(science)[:, :] -= dit.computeClippedImageStats(ga(science)).mean
print dit.computeClippedImageStats(ga(template))
print dit.computeClippedImageStats(ga(science))

dit.plotImageGrid((ga(template), ga(science)), imScale=16, titles=['Template', 'Science'])

dit.plotImageGrid((science.getPsf().computeKernelImage().getArray(),
                   template.getPsf().computeKernelImage().getArray()))

dit.plotImageGrid((gv(template), gv(science)), titles=['Template (var)', 'Science (var)'],
                  imScale=16)

config = ipDiffim.ZogyConfig()
task = ipDiffim.ZogyTask(templateExposure=template, scienceExposure=science, config=config)

D = task.computeDiffim(inImageSpace=False)

alDiffim = afwImage.ExposureF('/Users/dreiss/DATA/HiTS_from_meredith/diffim_15A38_g/deepDiff/v411033/diffexp-03.fits')

print dit.computeClippedImageStats(ga(D))
print dit.computeClippedImageStats(ga(alDiffim))
print dit.computeClippedImageStats(gv(D))
dit.plotImageGrid((ga(D), ga(alDiffim)), imScale=16, titles=['ZOGY', 'AL'])  # gv(D)

dit.plotImageGrid((ga(template)[3000:3250, 1500:1750], ga(science)[3000:3250, 1500:1750], 
                   ga(D)[3000:3250, 1500:1750], ga(alDiffim)[3000:3250, 1500:1750]), imScale=4, 
                  extent=(1500, 3000, 1750, 3250),
                 titles=['template', 'science', 'ZOGY', 'AL'])  # gv(D)

dit.plotImageGrid((ga(template)[1000:1250, 900:1150], ga(science)[1000:1250, 900:1150], 
                   ga(D)[1000:1250, 900:1150], ga(alDiffim)[1000:1250, 900:1150]), imScale=4, 
                  extent=(900, 1000, 1150, 1250),
                 titles=['template', 'science', 'ZOGY', 'AL'])  # gv(D)

dit.plotImageGrid((
    template.getPsf().computeKernelImage().getArray(),
    science.getPsf().computeKernelImage().getArray(),
    D.getPsf().computeKernelImage().getArray(), 
    alDiffim.getPsf().computeKernelImage().getArray()),
    titles=['template', 'science', 'ZOGY', 'AL'], imScale=2.5)

S = task.computeScorr(inImageSpace=False)

dit.plotImageGrid((ga(D), ga(S)), imScale=16, titles=['ZOGY D', 'ZOGY Scorr'])  # gv(D)

D2 = task.computeDiffim(inImageSpace=True)

dit.plotImageGrid((ga(D), ga(D2)), imScale=16, titles=['ZOGY (F)', 'ZOGY (R)'])  # gv(D)



config = ipDiffim.ZogyMapReduceConfig()
config.gridStepX = config.gridStepY = 9
config.gridSizeX = config.gridSizeY = 20
config.borderSizeX = config.borderSizeY = 6
config.reducerSubtask.reduceOperation = 'average'
task = ipDiffim.ImageMapReduceTask(config=config)
print config

DF = task.run(science, template=template, inImageSpace=False,
             doScorr=False, forceEvenSized=True).exposure

task.plotBoxes(science.getBBox(), skip=23)

print dit.computeClippedImageStats(ga(D))
print dit.computeClippedImageStats(ga(DF))
dit.plotImageGrid((ga(D), ga(DF)), imScale=16, titles=['ZOGY (const)', 'ZOGY (mapReduced)'])

dit.plotImageGrid((ga(template)[1000:1250, 900:1150], ga(science)[1000:1250, 900:1150], 
                   ga(D)[1000:1250, 900:1150], ga(DF)[1000:1250, 900:1150]), imScale=4, 
                  extent=(900, 1000, 1150, 1250),
                titles=['template', 'science', 'ZOGY', 'ZOGY (spatial)'])

dit.plotImageGrid((gv(D), gv(alDiffim)), imScale=16, titles=['ZOGY (var)', 'AL (var)'])

print dit.computeClippedImageStats(ga(D))
print dit.computeClippedImageStats(ga(DF))
print dit.computeClippedImageStats(ga(alDiffim))
print dit.computeClippedImageStats(ga(D) - ga(DF))
print dit.computeClippedImageStats(ga(D) - ga(alDiffim))
print dit.computeClippedImageStats(ga(DF) - ga(alDiffim))
dit.plotImageGrid((ga(D) - ga(DF), ga(DF) - ga(alDiffim),), imScale=16, 
                   titles=['ZOGY (const) - ZOGY (mapReduced)',
                          'ZOGY (mapReduced) - AL (const decorr)'])

d1 = ga(D)
d2 = ga(alDiffim)
d3 = ga(DF)
# The ZOGY diffim has artefacts on the edges. Let's set them to zero so they dont mess up the stats.
# Actually, let's just set the edge pixels of both diffims to zero.
d1[0,:] = d1[:,0] = d1[-1,:] = d1[:,-1] = 0.
d2[0,:] = d2[:,0] = d2[-1,:] = d2[:,-1] = 0.
d3[0,:] = d3[:,0] = d3[-1,:] = d3[:,-1] = 0.
d1[np.isnan(d1)|np.isnan(d2)|np.isnan(d3)] = 0.
d2[np.isnan(d2)|np.isnan(d1)|np.isnan(d3)] = 0.
d3[np.isnan(d2)|np.isnan(d1)|np.isnan(d3)] = 0.

import scipy.stats
import pandas as pd
_, low, upp = scipy.stats.sigmaclip([d1, d2, d3])
print low, upp
low *= 1.1
upp *= 1.1
d1a = d1[(d1>low) & (d1<upp) & (d2>low) & (d2<upp) & (d3>low) & (d3<upp) & 
         (d1!=0.) & (d2!=0.) & (d3!=0.)]
d2a = d2[(d1>low) & (d1<upp) & (d2>low) & (d2<upp) & (d3>low) & (d3<upp) &
         (d1!=0.) & (d2!=0.) & (d3!=0.)]
d3a = d3[(d1>low) & (d1<upp) & (d2>low) & (d2<upp) & (d3>low) & (d3<upp) &
         (d1!=0.) & (d2!=0.) & (d3!=0.)]
df = pd.DataFrame({'ZOGY': d1a.flatten()/d1a.std(), 'AL': d2a.flatten()/d2a.std(),
                   'ZOGY (MR)': d3a.flatten()/d3a.std()})
df.plot.hist(alpha=0.1, bins=200)

config = ipDiffim.ZogyMapReduceConfig()
config.gridStepX = config.gridStepY = 100
config.gridSizeX = config.gridSizeY = 20
config.borderSizeX = config.borderSizeY = 3
config.reducerSubtask.reduceOperation = 'none'
config.returnSubImages = True
task = ipDiffim.ImageMapReduceTask(config=config)
print config

subims = task.run(science, template=template, inImageSpace=False,
             doScorr=False, forceEvenSized=True)
print len(subims.result)

res = subims.result[0]
res.getDict()

def getPsfimgAtBoxCentroid(exposure):
    box = exposure.getBBox()
    psf = exposure.getPsf()
    centroid = ((box.getEndX() + box.getBeginX())/2., (box.getEndY() + box.getBeginY())/2.)
    coord = '%.1f,%.1f' % centroid
    img = psf.computeKernelImage(afwGeom.Point2D(centroid[0], centroid[1])).getArray()
    return img, coord

psfIms = [getPsfimgAtBoxCentroid(res.inputSubExposure)[0] for res in subims.result]
coords = [getPsfimgAtBoxCentroid(res.inputSubExposure)[1] for res in subims.result]
for im in psfIms:
    im[np.isnan(im)] = 0.

dit.plotImageGrid((psfIms), titles=coords)

dit.plotImageGrid((psfIms[0] - psfIms[-1],))

psfIms = [getPsfimgAtBoxCentroid(res.subExposure)[0] for res in subims.result]
coords = [getPsfimgAtBoxCentroid(res.subExposure)[1] for res in subims.result]
for im in psfIms:
    im[np.isnan(im)] = 0.

dit.plotImageGrid((psfIms), titles=coords)

dit.plotImageGrid((psfIms[1] - psfIms[-1],))



