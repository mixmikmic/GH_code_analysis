import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd

import lsst.ip.diffim as ipDiffim

import diffimTests as dit
#reload(dit)

testObj = dit.DiffimTest(varFlux2=np.repeat(5000, 10),
                         n_sources=1000, verbose=True, sourceFluxRange=(2000., 60000.), # psf_yvary_factor=0.5, 
                         psfSize=13)
res = testObj.runTest(spatialKernelOrder=2)
print res

#dit.plotImageGrid((testObj.im1.im, testObj.im2.im), imScale=8)
testObj.doPlot(imScale=6);

exposure = testObj.im2.asAfwExposure()
template = testObj.im1.asAfwExposure()

import lsst.pex.config as pexConfig
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom

class ZogyMapperSubtask(ipDiffim.ImageMapperSubtask):
    ConfigClass = ipDiffim.ImageMapperSubtaskConfig
    _DefaultName = 'ip_diffim_ZogyMapperSubtask'
    
    def __init__(self, *args, **kwargs):
        ipDiffim.ImageMapperSubtask.__init__(self, *args, **kwargs)
        
    def run(self, subExp, expandedSubExp, fullBBox, **kwargs):
        bbox = subExp.getBBox()
        center = ((bbox.getBeginX() + bbox.getEndX()) // 2., (bbox.getBeginY() + bbox.getEndY()) // 2.)
        center = afwGeom.Point2D(center[0], center[1])
        
        #print center, subExp.getBBox(), expandedSubExp.getBBox()
        
        # Psf and image for science img (index 2)
        subExp2 = subExp
        psf2 = subExp.getPsf().computeImage(center).getArray()
        psf2_orig = psf2
        subim2 = expandedSubExp.getMaskedImage()
        subarr2 = subim2.getImage().getArray()
        subvar2 = subim2.getVariance().getArray()
        sig2 = np.sqrt(dit.computeClippedImageStats(subvar2).mean)
        
        # Psf and image for template img (index 1)
        template = kwargs.get('template')
        subExp1 = afwImage.ExposureF(template, expandedSubExp.getBBox())
        psf1 = template.getPsf().computeImage(center).getArray()
        psf1_orig = psf1
        subim1 = subExp1.getMaskedImage()
        subarr1 = subim1.getImage().getArray()
        subvar1 = subim1.getVariance().getArray()
        sig1 = np.sqrt(dit.computeClippedImageStats(subvar1).mean)
        
        #shape2 = subExp2.getPsf().computeShape(afwGeom.Point2D(center[0], center[1]))
        #shape1 = subExp1.getPsf().computeShape(afwGeom.Point2D(center[0], center[1]))
        #print shape2, shape1
        
        psf1b = psf1; psf2b = psf2
        if psf1.shape[0] == 41:   # it's a measured psf (hack!) Note this really helps for measured psfs.
            psf1b = psf1.copy()
            psf1b[psf1b < 0] = 0
            #psf1b[0:10,0:10] = psf1b[31:41,31:41] = 0
            psf1b[0:10,:] = psf1b[:,0:10] = psf1b[31:41,:] = psf1b[:,31:41] = 0
            psf1b /= psf1b.sum()

            psf2b = psf2.copy()
            psf2b[psf2b < 0] = 0
            psf2b[0:10,:] = psf2b[:,0:10] = psf2b[31:41,:] = psf2b[:,31:41] = 0
            psf2b /= psf2b.sum()

        # from diffimTests.diffimTests ...
        padSize0 = subarr1.shape[0]//2 - psf1.shape[0]//2
        padSize1 = subarr1.shape[1]//2 - psf1.shape[1]//2
        # Hastily assume the image is even-sized and the psf is odd... and that the two images
        #   and psfs have the same dimensions!
        psf1 = np.pad(psf1b, ((padSize0, padSize0-1), (padSize1, padSize1-1)), mode='constant',
                      constant_values=0)
        psf2 = np.pad(psf2b, ((padSize0, padSize0-1), (padSize1, padSize1-1)), mode='constant',
                      constant_values=0)
        if psf1.shape[0] > subarr1.shape[0]:
            psf1 = psf1[:-1, :]
            psf2 = psf2[:-1, :]
        elif psf1.shape[0] < subarr1.shape[0]:
            psf1 = np.pad(psf1, ((0, 1), (0, 0)), mode='constant', constant_values=0)
            psf2 = np.pad(psf2, ((0, 1), (0, 0)), mode='constant', constant_values=0)
        if psf1.shape[1] > subarr1.shape[1]:
            psf1 = psf1[:, :-1]
            psf2 = psf2[:, :-1]
        elif psf1.shape[1] < subarr1.shape[1]:
            psf1 = np.pad(psf1, ((0, 0), (0, 1)), mode='constant', constant_values=0)
            psf2 = np.pad(psf2, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        #psf1 = dit.psf.recenterPsf(psf1)
        #psf2 = dit.psf.recenterPsf(psf2)
        psf1 /= psf1.sum()
        psf2 /= psf2.sum()
        
        #shape2 = dit.arrayToAfwPsf(psf2).computeShape(afwGeom.Point2D(center[0], center[1]))
        #shape1 = dit.arrayToAfwPsf(psf1).computeShape(afwGeom.Point2D(center[0], center[1]))
        #print center, shape2, shape1

        D_zogy, var_zogy = dit.zogy.performZOGY(subarr1, subarr2,
                                                subvar1, subvar2,
                                                psf1, psf2, 
                                                sig1=sig1, sig2=sig2)
        
#         D_zogy, var_zogy = dit.zogy.performZOGYImageSpace(subarr1, subarr2,
#                                                           subvar1, subvar2,
#                                                           psf1, psf2, 
#                                                           sig1=sig1, sig2=sig2, padSize=15)
        tmpExp = expandedSubExp.clone()
        tmpIM = tmpExp.getMaskedImage()
        tmpIM.getImage().getArray()[:, :] = D_zogy
        tmpIM.getVariance().getArray()[:, :] = var_zogy
        # need to eventually compute diffim PSF and set it here.
        out = afwImage.ExposureF(tmpExp, subExp.getBBox())
                
        #print template
        #img += 10.
        return out #, psf1_orig, psf2_orig
    
class ZogyMapReduceConfig(ipDiffim.ImageMapReduceConfig):
    mapperSubtask = pexConfig.ConfigurableField(
        doc='Zogy subtask to run on each sub-image',
        target=ZogyMapperSubtask
    )

config = ZogyMapReduceConfig()
#config.gridStepX = config.gridStepY = 5
#config.gridSizeX = config.gridSizeY = 7
#config.borderSizeX = config.borderSizeY = 10
config.reducerSubtask.reduceOperation = 'average'
task = ipDiffim.ImageMapReduceTask(config=config)
print config
boxes0, boxes1 = task._generateGrid(exposure)

newExp = task.run(exposure, template=template)

print dit.computeClippedImageStats(ga(exposure))
print dit.computeClippedImageStats(ga(newExp))
print dit.computeClippedImageStats(ga(exposure)-ga(newExp))
print dit.computeClippedImageStats(ga(newExp) - testObj.D_ZOGY.im)
dit.plotImageGrid((ga(newExp), newExp.getMaskedImage().getImage(), ga(newExp) - testObj.D_ZOGY.im), imScale=12)

def ga(exposure):
    return exposure.getMaskedImage().getImage().getArray()

import lsst.afw.geom as afwGeom
centroid = afwGeom.Point2I(210, 146) #216, 392)
ind = 0
for box in task.boxes0:
    if box.contains(centroid):
        break
    ind += 1
print ind, task.boxes0[ind], task.boxes1[ind]

subExp = afwImage.ExposureF(exposure, task.boxes0[ind])
expandedSubExp = afwImage.ExposureF(exposure, task.boxes1[ind])
result = task.mapperSubtask.run(subExp, expandedSubExp, exposure.getBBox(), template=template)
ZOGYsub = afwImage.ExposureF(testObj.D_ZOGY.asAfwExposure(), task.boxes0[ind])
dit.plotImageGrid((ga(subExp), ga(result), ga(ZOGYsub), ga(result)-ga(ZOGYsub)))

bbox = subExp.getBBox()
center = ((bbox.getBeginX() + bbox.getEndX()) // 2., (bbox.getBeginY() + bbox.getEndY()) // 2.)
center = afwGeom.Point2D(center[0], center[1])
print center
psf1 = template.getPsf().computeImage(center).getArray()
psf2 = exposure.getPsf().computeImage(center).getArray()
print dit.psf.computeMoments(psf1), dit.psf.computeMoments(psf2)
print dit.psf.computeMoments(testObj.im1.psf), dit.psf.computeMoments(testObj.im2.psf)
dit.plotImageGrid((psf1, psf2, psf1-testObj.im1.psf, psf2-testObj.im2.psf), clim=(-0.000001, 0.000001))

psf1 = template.getPsf().computeImage(center).getArray()
psf2 = exposure.getPsf().computeImage(center).getArray()
subarr1 = expandedSubExp.getMaskedImage().getImage().getArray()

padSize0 = subarr1.shape[0]//2 - psf1.shape[0]//2
padSize1 = subarr1.shape[1]//2 - psf1.shape[1]//2

psf1 = np.pad(psf1, ((padSize0, padSize0-1), (padSize1, padSize1-1)), mode='constant',
                      constant_values=0)
psf2 = np.pad(psf2, ((padSize0, padSize0-1), (padSize1, padSize1-1)), mode='constant',
                      constant_values=0)
print psf1.shape, subarr1.shape
print dit.psf.computeMoments(psf1), dit.psf.computeMoments(psf2)
dit.plotImageGrid((psf1, psf2), clim=(-0.0000001, 0.0000001))

if psf1.shape[0] > subarr1.shape[0]:
    psf1 = psf1[:-1, :]
    psf2 = psf2[:-1, :]
elif psf1.shape[0] < subarr1.shape[0]:
    psf1 = np.pad(psf1, ((0, 1), (0, 0)), mode='constant', constant_values=0)
    psf2 = np.pad(psf2, ((0, 1), (0, 0)), mode='constant', constant_values=0)
if psf1.shape[1] > subarr1.shape[1]:
    psf1 = psf1[:, :-1]
    psf2 = psf2[:, :-1]
elif psf1.shape[1] < subarr1.shape[1]:
    psf1 = np.pad(psf1, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    psf2 = np.pad(psf2, ((0, 0), (0, 1)), mode='constant', constant_values=0)

print psf1.shape, subarr1.shape
print dit.psf.computeMoments(psf1), dit.psf.computeMoments(psf2)
dit.plotImageGrid((psf1, psf2), clim=(-0.0000001, 0.0000001))



exposure2 = testObj.im2.asAfwExposure()
res = dit.tasks.doMeasurePsf(exposure2, spatialOrder=1)
psf = res.psf
exposure2.setPsf(psf)

template2 = testObj.im1.asAfwExposure()
res = dit.tasks.doMeasurePsf(template2, spatialOrder=1)
psf = res.psf
template2.setPsf(psf)

print testObj.im1.psf.shape, testObj.im2.psf.shape
print testObj.im1.asAfwExposure().getPsf().computeImage().getArray().shape
print testObj.im2.asAfwExposure().getPsf().computeImage().getArray().shape
print exposure2.getPsf().computeImage().getArray().shape
print template2.getPsf().computeImage().getArray().shape

print testObj.im1.sig, testObj.im2.sig

import lsst.afw.geom as afwGeom
psf = exposure2.getPsf()

psf1 = psf.computeImage(afwGeom.Point2D(20., 20.)).getArray()
psf1b = psf1.copy()
psf1b[psf1b < 0] = 0
print psf1b[0:10,:].mean(), psf1b[:,0:10].mean(), psf1b[31:41,:].mean(), psf1b[:,31:41].mean()
psf1b[0:10,:] = psf1b[:,0:10] = psf1b[31:41,:] = psf1b[:,31:41] = 0
psf1b /= psf1b.sum()

dit.plotImageGrid((psf.computeImage(afwGeom.Point2D(20., 20.)), psf.computeImage(afwGeom.Point2D(500.,500.)), psf1b),
                 clim=(0.00001, 0.001))

psf = template2.getPsf()
dit.plotImageGrid((psf.computeImage(afwGeom.Point2D(20., 20.)), psf.computeImage(afwGeom.Point2D(500.,500.))))

config = ZogyMapReduceConfig()
#config.gridStepX = config.gridStepY = 5
#config.gridSizeX = config.gridSizeY = 7
#config.borderSizeX = config.borderSizeY = 10
config.reducerSubtask.reduceOperation = 'average'
task = ipDiffim.ImageMapReduceTask(config=config)
print config

newExp2 = task.run(exposure2, template=template2)

print dit.computeClippedImageStats(ga(exposure2))
print dit.computeClippedImageStats(ga(newExp))
print dit.computeClippedImageStats(ga(newExp2))
print dit.computeClippedImageStats(ga(exposure2)-ga(newExp2))
print dit.computeClippedImageStats(ga(newExp) - testObj.D_ZOGY.im)
print dit.computeClippedImageStats(ga(newExp2) - testObj.D_ZOGY.im)
print dit.computeClippedImageStats(ga(newExp) - ga(newExp2))
dit.plotImageGrid((ga(newExp2), ga(newExp), ga(newExp2) - testObj.D_ZOGY.im,
                  ga(newExp) - testObj.D_ZOGY.im), imScale=12)





