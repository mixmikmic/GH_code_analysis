import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd

#import lsst.ip.diffim as ipDiffim
# I linked from ip_diffim/python/lsst/ip/diffim/imageMapReduce.py into diffimTests/imageMapReduce.py
#    (or copy it)

import diffimTests as dit
#reload(dit)

testObj = dit.DiffimTest(varFlux2=np.repeat(5000, 10),
                         n_sources=2500, verbose=True, sourceFluxRange=(2000., 120000.), 
                         psf_yvary_factor=0.5, psfSize=13)
res = testObj.runTest(spatialKernelOrder=2)
print res

exposure = testObj.im2.asAfwExposure()
template = testObj.im1.asAfwExposure()

#dit.plotImageGrid((testObj.im1.im, testObj.im2.im), imScale=8)
testObj.doPlot(imScale=6);

dit.plotImageGrid((testObj.variablePsf.getImage(20., 20.), 
                   testObj.variablePsf.getImage(250., 250.),
                   testObj.variablePsf.getImage(500., 500.)))

import lsst.pex.config as pexConfig
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.pipe.base as pipeBase

class ZogyMapperSubtask(dit.ImageMapperSubtask):
    ConfigClass = dit.ImageMapperSubtaskConfig
    _DefaultName = 'diffimTests_ZogyMapperSubtask'
    
    def __init__(self, *args, **kwargs):
        dit.ImageMapperSubtask.__init__(self, *args, **kwargs)
        
    def run(self, subExp, expandedSubExp, fullBBox, **kwargs):
        bbox = subExp.getBBox()
        center = ((bbox.getBeginX() + bbox.getEndX()) // 2., (bbox.getBeginY() + bbox.getEndY()) // 2.)
        center = afwGeom.Point2D(center[0], center[1])
        
        variablePsf2 = kwargs.get('variablePsf', None)
        sigmas = kwargs.get('sigmas', None)
        inImageSpace = kwargs.get('inImageSpace', False)
                
        # Psf and image for science img (index 2)
        subExp2 = subExp
        if variablePsf2 is None:
            psf2 = subExp.getPsf().computeImage(center).getArray()
        else:
            psf2 = variablePsf2.getImage(center.getX(), center.getY())
        psf2_orig = psf2
        subim2 = expandedSubExp.getMaskedImage()
        subarr2 = subim2.getImage().getArray()
        subvar2 = subim2.getVariance().getArray()
        if sigmas is None:
            sig2 = np.sqrt(dit.computeClippedImageStats(subvar2).mean)
        else:
            sig2 = sigmas[1]  # for testing, can use the input sigma (global value for entire exposure)
        
        # Psf and image for template img (index 1)
        template = kwargs.get('template')
        subExp1 = afwImage.ExposureF(template, expandedSubExp.getBBox())
        psf1 = template.getPsf().computeImage(center).getArray()
        psf1_orig = psf1
        subim1 = subExp1.getMaskedImage()
        subarr1 = subim1.getImage().getArray()
        subvar1 = subim1.getVariance().getArray()
        if sigmas is None:
            sig1 = np.sqrt(dit.computeClippedImageStats(subvar1).mean)
        else:
            sig1 = sigmas[0]
        
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
        if subarr1.shape[0] < psf1.shape[0] or subarr1.shape[1] < psf1.shape[1]:
            return pipeBase.Struct(subExposure=subExp)

        D_zogy, var_zogy = None, None
        if not inImageSpace:
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

            D_zogy, var_zogy = dit.zogy.performZogy(subarr1, subarr2,
                                                subvar1, subvar2,
                                                psf1, psf2, 
                                                sig1=sig1, sig2=sig2)
        else:
            D_zogy, var_zogy = dit.zogy.performZogyImageSpace(subarr1, subarr2,
                                                           subvar1, subvar2,
                                                           psf1, psf2, 
                                                           sig1=sig1, sig2=sig2, padSize=7)
            #print 'HERE:', subExp.getBBox(), np.sum(~np.isnan(D_zogy)), np.sum(~np.isnan(var_zogy))

        tmpExp = expandedSubExp.clone()
        tmpIM = tmpExp.getMaskedImage()
        tmpIM.getImage().getArray()[:, :] = D_zogy
        tmpIM.getVariance().getArray()[:, :] = var_zogy
        # need to eventually compute diffim PSF and set it here.
        out = afwImage.ExposureF(tmpExp, subExp.getBBox())
                
        #print template
        #img += 10.
        #return out #, psf1_orig, psf2_orig
        return pipeBase.Struct(subExposure=out)

class ZogyMapReduceConfig(dit.ImageMapReduceConfig):
    mapperSubtask = pexConfig.ConfigurableField(
        doc='Zogy subtask to run on each sub-image',
        target=ZogyMapperSubtask
    )

config = ZogyMapReduceConfig()
#config.gridStepX = config.gridStepY = 5
#config.gridSizeX = config.gridSizeY = 7
#config.borderSizeX = config.borderSizeY = 3
config.reducerSubtask.reduceOperation = 'average'
task = dit.ImageMapReduceTask(config=config)
print config
#boxes0, boxes1 = task._generateGrid(exposure)

newExp = task.run(exposure, template=template).exposure #, variablePsf=testObj.variablePsf)

# Run with constant variance number to compare with single frame (non-gridded) ZOGY from above:
sig1 = np.sqrt(dit.computeClippedImageStats(template.getMaskedImage().getVariance().getArray()).mean)
sig2 = np.sqrt(dit.computeClippedImageStats(exposure.getMaskedImage().getVariance().getArray()).mean)
print sig1, sig2
newExpA = task.run(exposure, template=template, sigmas=[sig1, sig2]).exposure

# Run with ZOGY in image space... ugh. Need bigger borders, for sure!

config = ZogyMapReduceConfig()
#config.gridStepX = config.gridStepY = 5
#config.gridSizeX = config.gridSizeY = 7
config.borderSizeX = config.borderSizeY = 3
config.reducerSubtask.reduceOperation = 'average'
task = dit.ImageMapReduceTask(config=config)
print config
#boxes0, boxes1 = task._generateGrid(exposure)

newExpB = task.run(exposure, template=template, sigmas=[sig1, sig2], inImageSpace=True).exposure

def ga(exposure):
    return exposure.getMaskedImage().getImage().getArray()

print dit.computeClippedImageStats(ga(exposure))
print dit.computeClippedImageStats(ga(newExp))
print dit.computeClippedImageStats(ga(newExpA))
print dit.computeClippedImageStats(ga(newExpB))
print dit.computeClippedImageStats(ga(exposure)-ga(newExp))
print dit.computeClippedImageStats(ga(newExp) - testObj.D_Zogy.im)
print dit.computeClippedImageStats(ga(newExpA) - testObj.D_Zogy.im)
print dit.computeClippedImageStats(ga(newExpB) - testObj.D_Zogy.im)
dit.plotImageGrid((ga(newExp), ga(newExpA), ga(newExpB), testObj.D_Zogy.im, ga(newExp)-testObj.D_Zogy.im,
                  ga(newExpA)-testObj.D_Zogy.im, ga(newExpB)-testObj.D_Zogy.im), imScale=12)

subexps = task._runMapper(exposure, template=template, sigmas=[sig1, sig2])
dit.plotImageGrid((subexps[0].subExposure, subexps[1].subExposure))

subexps = task._runMapper(exposure, template=template, sigmas=[sig1, sig2], inImageSpace=True)
dit.plotImageGrid((subexps[0].subExposure, subexps[1].subExposure))

print task.config
task._plotBoxes(exposure.getBBox(), skip=5)







newExp = task.run(exposure, template=template, variablePsf=testObj.variablePsf).exposure

print dit.computeClippedImageStats(ga(exposure))
print dit.computeClippedImageStats(ga(newExp))
print dit.computeClippedImageStats(ga(exposure)-ga(newExp))
print dit.computeClippedImageStats(ga(newExp) - testObj.D_Zogy.im)
dit.plotImageGrid((ga(newExp), testObj.D_Zogy.im, ga(newExp)-testObj.D_Zogy.im), imScale=12)

newExpB = task.run(exposure, template=template, inImageSpace=True, variablePsf=testObj.variablePsf).exposure

print dit.computeClippedImageStats(ga(exposure))
print dit.computeClippedImageStats(ga(newExpB))
print dit.computeClippedImageStats(ga(exposure)-ga(newExpB))
print dit.computeClippedImageStats(ga(newExpB) - testObj.D_Zogy.im)
print dit.computeClippedImageStats(ga(newExpB) - ga(newExp))
dit.plotImageGrid((ga(newExpB), testObj.D_Zogy.im, ga(newExpB)-ga(newExp)), imScale=12)

import lsst.meas.extensions.psfex as psfex

# copied ~/lsstsw/build/meas_extensions_psfex/tests/testPsfexPsf.py to ./testPsfexPsfEDITED.py
# then made some edits...
import testPsfexPsfEDITED as tpp
reload(tpp)
obj1 = tpp.SpatialModelPsfTestCase() # removed superclass unittest.TestCase
obj1.makeExposure()
obj1.setUp()
obj1.testPsfexDeterminer()
dit.plotImageGrid((obj1.exposure.getPsf(),))

reload(tpp)
obj2 = tpp.SpatialModelPsfTestCase() # removed superclass unittest.TestCase
obj2.setExposure(exposure)
obj2.setUp()
obj2.testPsfexDeterminer()

#dit.plotImageGrid((obj.exposure.getPsf(),))
psf = obj2.exposure.getPsf()
dit.plotImageGrid((psf.computeImage(afwGeom.Point2D(20., 20.)), 
                   psf.computeImage(afwGeom.Point2D(250., 250.)),
                   psf.computeImage(afwGeom.Point2D(500.,500.))))

mi1 = obj1.exposure.getMaskedImage()
mi2 = obj2.exposure.getMaskedImage()

print dit.computeClippedImageStats(mi1.getImage().getArray())
print dit.computeClippedImageStats(mi2.getImage().getArray())
print()
print dit.computeClippedImageStats(mi1.getVariance().getArray())
print dit.computeClippedImageStats(mi2.getVariance().getArray())

dit.plotImageGrid((obj1.exposure, obj1.subtracted), imScale=6)

psf = obj1.exposure.getPsf()
dit.plotImageGrid((psf,))

dit.plotImageGrid((obj2.exposure, obj2.subtracted), imScale=6)

exposure2 = testObj.im2.asAfwExposure()
#res = dit.tasks.doMeasurePsf(exposure2) #, psfMeasureConfig=config)
#psf = res.psf
obj = tpp.SpatialModelPsfTestCase() # removed superclass unittest.TestCase
obj.setExposure(exposure2)
obj.setUp()
obj.testPsfexDeterminer()
psf = obj.exposure.getPsf()
exposure2.setPsf(psf)

template2 = testObj.im1.asAfwExposure()
#res = dit.tasks.doMeasurePsf(template2) #, psfMeasureConfig=config)
#psf = res.psf
obj = tpp.SpatialModelPsfTestCase() # removed superclass unittest.TestCase
obj.setExposure(template2)
obj.setUp()
obj.testPsfexDeterminer()
psf = obj.exposure.getPsf()
template2.setPsf(psf)

dit.plotImageGrid((testObj.variablePsf.getImage(40., 40.), 
                   testObj.variablePsf.getImage(250., 250.),
                   testObj.variablePsf.getImage(500., 500.)))

psf = exposure2.getPsf()
print psf.computeImage().getDimensions()
dit.plotImageGrid((psf.computeImage(afwGeom.Point2D(20., 20.)), 
                   psf.computeImage(afwGeom.Point2D(250., 250.)),
                   psf.computeImage(afwGeom.Point2D(500.,500.))))

psf = template2.getPsf()
dit.plotImageGrid((psf.computeImage(afwGeom.Point2D(20., 20.)),
                   psf.computeImage(afwGeom.Point2D(250., 250.)),
                   psf.computeImage(afwGeom.Point2D(500.,500.))))

config = ZogyMapReduceConfig()
config.gridStepX = config.gridStepY = 9
#config.gridSizeX = config.gridSizeY = 10
#config.borderSizeX = config.borderSizeY = 7
config.reducerSubtask.reduceOperation = 'average'
task = dit.ImageMapReduceTask(config=config)
print config

newExp2 = task.run(exposure2, template=template2).exposure
task._plotBoxes(exposure2.getBBox())

print dit.computeClippedImageStats(ga(exposure2))
print dit.computeClippedImageStats(ga(newExp))
print dit.computeClippedImageStats(ga(newExp2))
print dit.computeClippedImageStats(ga(exposure2)-ga(newExp2))
print dit.computeClippedImageStats(ga(newExp) - testObj.D_Zogy.im)
print dit.computeClippedImageStats(ga(newExp2) - testObj.D_Zogy.im)
print dit.computeClippedImageStats(ga(newExp) - ga(newExp2))
dit.plotImageGrid((ga(newExp2), ga(newExp), ga(newExp2) - ga(newExp)), #ga(newExp) - testObj.D_Zogy.im), 
                  imScale=12)

testObj2 = testObj.clone()
testObj2.D_Zogy = dit.Exposure(ga(newExp2), testObj.D_Zogy.psf, newExp2.getMaskedImage().getVariance().getArray())
testObj2.runTest()

# for comparison, without using a spatially-varying PSF:
testObj.runTest()

# let's add the one where I used the *input* but variable PSF
testObj3 = testObj.clone()
testObj3.D_Zogy = dit.Exposure(ga(newExp), testObj.D_Zogy.psf, newExp.getMaskedImage().getVariance().getArray())
testObj3.runTest()

# And the image-space version of that:
testObj2 = testObj.clone()
testObj2.D_Zogy = dit.Exposure(ga(newExpB), testObj.D_Zogy.psf, newExpB.getMaskedImage().getVariance().getArray())
testObj2.runTest()















exposure2 = testObj.im1.asAfwExposure()
#res = dit.tasks.doMeasurePsf(exposure2) #, psfMeasureConfig=config)
#psf = res.psf
obj = tpp.SpatialModelPsfTestCase() # removed superclass unittest.TestCase
obj.setExposure(exposure2)
obj.setUp()
obj.testPsfexDeterminer()
psf = obj.exposure.getPsf()
exposure2.setPsf(psf)

import lsst.meas.deblender as measDeblend
import lsst.afw.table as afwTable
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDetection
from lsst.meas.base import SingleFrameMeasurementTask

config = SingleFrameMeasurementTask.ConfigClass()
config.slots.apFlux = 'base_CircularApertureFlux_12_0'
schema = afwTable.SourceTable.makeMinimalSchema()

measureSources = SingleFrameMeasurementTask(schema, config=config)

width, height = exposure2.getDimensions()
bbox = afwGeom.BoxI(afwGeom.PointI(0, 0), afwGeom.ExtentI(width, height))
cellSet = afwMath.SpatialCellSet(bbox, 100)

mi = exposure2.getMaskedImage()
footprintSet = afwDetection.FootprintSet(mi, afwDetection.Threshold(100), "DETECTED")

catalog = afwTable.SourceCatalog(schema)

footprintSet.makeSources(catalog)
print(len(catalog))
#catalog = catalog.copy(deep=True)

measureSources.run(catalog, exposure2)

import lsst.meas.algorithms as measAlg

for source in catalog:
    try:
        cand = measAlg.makePsfCandidate(source, exposure2)
        cellSet.insertCandidate(cand)

    except Exception as e:
        print(e)
        continue

dbConfig = measDeblend.SourceDeblendTask.ConfigClass()
deblendSources = measDeblend.SourceDeblendTask(schema, config=dbConfig)
deblendSources.run(exposure2, catalog)

# filter the catalog and use the deblended heavy footprint of the 
# children only
# then use MakeCandidateTask but tell it to use the heavy footprint images of those 
# deblended children

starSelectorClass = measAlg.starSelectorRegistry["objectSize"]
starSelectorConfig = starSelectorClass.ConfigClass()
starSelectorConfig.sourceFluxField = "base_GaussianFlux_flux"
starSelectorConfig.badFlags = ["base_PixelFlags_flag_edge",
                               "base_PixelFlags_flag_interpolatedCenter",
                               "base_PixelFlags_flag_saturatedCenter",
                               "base_PixelFlags_flag_crCenter",
                               ]
starSelectorConfig.widthStdAllowed = 0.5  # Set to match when the tolerance of the test was set

starSelector = starSelectorClass(schema=schema, config=starSelectorConfig)

psfDeterminerClass = measAlg.psfDeterminerRegistry["psfex"]
psfDeterminerConfig = psfDeterminerClass.ConfigClass()
width, height = exposure.getMaskedImage().getDimensions()
psfDeterminerConfig.sizeCellX = width//8
psfDeterminerConfig.sizeCellY = height//8
psfDeterminerConfig.spatialOrder = 1

psfDeterminer = psfDeterminerClass(psfDeterminerConfig)

import lsst.daf.base as dafBase

metadata = dafBase.PropertyList()
catalog = catalog.copy(deep=True)
psfCandidateList = starSelector.run(exposure2, catalog).psfCandidates
psf, cellSet = psfDeterminer.determinePsf(exposure2, psfCandidateList, metadata)

print psf.computeShape()
dit.plotImageGrid((psf.computeImage(afwGeom.Point2D(20., 20.)),
                   psf.computeImage(afwGeom.Point2D(250., 250.)),
                   psf.computeImage(afwGeom.Point2D(500.,500.))), clim=(-0.001,0.001))

psf = exposure2.getPsf()
print psf.computeImage().getDimensions()
print psf.computeShape()
dit.plotImageGrid((psf.computeImage(afwGeom.Point2D(20., 20.)), 
                   psf.computeImage(afwGeom.Point2D(250., 250.)),
                   psf.computeImage(afwGeom.Point2D(500.,500.))), clim=(-0.001,0.001))







mi = exposure.getMaskedImage()
img = mi.getImage()

print type(exposure)
print type(mi)
print type(img)

def _isExposureOrMaskedImageOrImage(item):
    if (isinstance(item, afwImage.ExposureU) or isinstance(item, afwImage.ExposureI) or
          isinstance(item, afwImage.ExposureF) or isinstance(item, afwImage.ExposureD)):
        return 'exposure'
    elif (isinstance(item, afwImage.MaskedImageU) or isinstance(item, afwImage.MaskedImageI) or
          isinstance(item, afwImage.MaskedImageF) or isinstance(item, afwImage.MaskedImageD)):
        return 'maskedImage'
    elif (isinstance(item, afwImage.ImageU) or isinstance(item, afwImage.ImageI) or
          isinstance(item, afwImage.ImageF) or isinstance(item, afwImage.ImageD)):
        return 'image'
    return 'none'

print _isExposureOrMaskedImageOrImage(exposure)
print _isExposureOrMaskedImageOrImage(mi)
print _isExposureOrMaskedImageOrImage(img)

