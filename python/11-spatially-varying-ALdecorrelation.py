import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd

import lsst.afw.geom as afwGeom

#import lsst.ip.diffim as ipDiffim
# I linked from ip_diffim/python/lsst/ip/diffim/imageMapReduce.py into diffimTests/imageMapReduce.py
#    (or copy it)

import diffimTests as dit
#reload(dit)

testObj = dit.DiffimTest(varFlux2=np.repeat(2000, 20), #620*np.sqrt(2), 10),
                         n_sources=600, verbose=True, sourceFluxRange=(2000., 120000.), 
                         psf_yvary_factor=0.5, psfSize=13)
res = testObj.runTest(spatialKernelOrder=2)
print res

exposure = testObj.im2.asAfwExposure()
template = testObj.im1.asAfwExposure()

exposure.setPsf(testObj.variablePsf.getCoaddPsf(exposure))
exposure.getPsf().computeKernelImage(afwGeom.Point2D(28., 66.)).getDimensions()

#dit.plotImageGrid((testObj.im1.im, testObj.im2.im), imScale=8)
testObj.doPlot(imScale=6, include_Szogy=True);

dit.plotImageGrid((testObj.variablePsf.getImage(20., 20.), 
                   testObj.variablePsf.getImage(250., 250.),
                   testObj.variablePsf.getImage(500., 500.)), clim=(-0.001, 0.001))

dit.plotImageGrid((exposure.getPsf().computeImage(afwGeom.Point2D(20., 20.)),
                   exposure.getPsf().computeImage(afwGeom.Point2D(250., 250.)),
                   exposure.getPsf().computeImage(afwGeom.Point2D(500., 500.)),), clim=(-0.001, 0.001))

import lsst.pex.config as pexConfig
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.pipe.base as pipeBase
import lsst.ip.diffim as ipDiffim

config = dit.ALdecMapReduceConfig()
#config.gridStepX = config.gridStepY = 5
#config.gridSizeX = config.gridSizeY = 7
config.borderSizeX = config.borderSizeY = 3
config.reducerSubtask.reduceOperation = 'average'
task = dit.ImageMapReduceTask(config=config)
print config
#boxes0, boxes1 = task._generateGrid(exposure)

psf = exposure.getPsf().computeImage(afwGeom.Point2D(24, 66)).getArray()
print psf.shape
if psf.shape[0] < psf.shape[1]:  # sometimes CoaddPsf does this.
    psf = np.pad(psf, ((1, 1), (0, 0)), mode='constant')
elif psf.shape[0] > psf.shape[1]:
    psf = np.pad(psf, ((0, 0), (1, 1)), mode='constant')
print psf.shape
dit.plotImageGrid((psf,))

diffimOrig = testObj.ALres.subtractedExposure
newExp = task.run(diffimOrig, template=template, science=exposure,
                 alTaskResult=testObj.ALres, forceEvenSized=True).exposure

# Run with constant variance number to compare with single frame (non-gridded) ZOGY from above:
sig1 = dit.computeClippedImageStats(template.getMaskedImage().getVariance().getArray()).mean
sig2 = dit.computeClippedImageStats(exposure.getMaskedImage().getVariance().getArray()).mean
print sig1, sig2
newExpA = task.run(diffimOrig, template=template, science=exposure,
                 alTaskResult=testObj.ALres, sigmaSquared=[sig1, sig2], forceEvenSized=True).exposure

diffimOrig = testObj.ALres.subtractedExposure
newExpB = task.run(diffimOrig, template=template, science=exposure,
                  alTaskResult=testObj.ALres, forceEvenSized=True,
                  variablePsf=testObj.variablePsf).exposure

def ga(exposure):
    return exposure.getMaskedImage().getImage().getArray()
def gv(exposure):
    return exposure.getMaskedImage().getVariance().getArray()

print dit.computeClippedImageStats(ga(exposure))
print dit.computeClippedImageStats(ga(newExp))
print dit.computeClippedImageStats(ga(newExpA))
print dit.computeClippedImageStats(ga(newExpB))
print dit.computeClippedImageStats(ga(newExp)-ga(testObj.ALres.subtractedExposure))
print dit.computeClippedImageStats(ga(newExp)-ga(testObj.ALres.decorrelatedDiffim))
print dit.computeClippedImageStats(ga(newExpA)-ga(testObj.ALres.subtractedExposure))
print dit.computeClippedImageStats(ga(newExpB)-ga(testObj.ALres.decorrelatedDiffim))
dit.plotImageGrid((ga(newExp), ga(newExpA), ga(newExpB), ga(testObj.ALres.decorrelatedDiffim), 
                   ga(newExp)-ga(testObj.ALres.decorrelatedDiffim),
                   ga(newExpA)-ga(testObj.ALres.decorrelatedDiffim), 
                   ga(newExpB)-ga(testObj.ALres.decorrelatedDiffim), gv(newExp)), imScale=12)

print task.config
task._plotBoxes(exposure.getBBox(), skip=5)

testObj.runTest(spatialKernelOrder=2)

testObj2 = testObj.clone()
testObj2.ALres.decorrelatedDiffim = newExp
testObj2.runTest(spatialKernelOrder=2)

testObj2 = testObj.clone()
testObj2.ALres.decorrelatedDiffim = newExpA
testObj2.runTest(spatialKernelOrder=2)

testObj2 = testObj.clone()
testObj2.ALres.decorrelatedDiffim = newExpB
testObj2.runTest(spatialKernelOrder=2)



config = dit.ALdecMapReduceConfig()
#config.gridStepX = config.gridStepY = 5
#config.gridSizeX = config.gridSizeY = 7
config.borderSizeX = config.borderSizeY = 3
config.reducerSubtask.reduceOperation = 'average'
task = dit.ImageMapReduceTask(config=config)
print config
#boxes0, boxes1 = task._generateGrid(exposure)

results = task._runMapper(diffimOrig, template=template, science=exposure,
                 alTaskResult=testObj.ALres, forceEvenSized=True) #, variablePsf=testObj.variablePsf)

def boxCenter(bbox):
    return ((bbox.getBeginX() + bbox.getEndX()) // 2, (bbox.getBeginY() + bbox.getEndY()) // 2)

centers = [boxCenter(res.subExposure.getBBox()) for res in results]
kimgs = [dit.afw.alPsfMatchingKernelToArray(testObj2.ALres.psfMatchingKernel, coord=center) for center in centers]
titles = ['%.0f,%.0f' % (c[0], c[1]) for c in centers]
dit.plotImageGrid(kimgs, titles=titles)

dcks = [res.decorrelationKernel for res in results]
dit.plotImageGrid(dcks, titles=titles)

inpPsfs = [testObj.variablePsf.getImage(c[0], c[1]) for c in centers]
dit.plotImageGrid(inpPsfs, titles=titles)

centers = [boxCenter(res.subExposure.getBBox()) for res in results]
psfs2 = [newExpB.getPsf().computeImage(afwGeom.Point2D(c[0], c[1])) for c in centers]
titles = ['%.0f,%.0f' % (c[0], c[1]) for c in centers]
dit.plotImageGrid(psfs2, titles=titles)











import lsst.afw.table as afwTable
import lsst.afw.coord as afwCoord

schema = afwTable.ExposureTable.makeMinimalSchema()
schema.addField("customweightname", type="D", doc="Coadd weight")
mycatalog = afwTable.ExposureCatalog(schema)

cd11 = 5.55555555e-05
cd12 = cd21 = 0.0
cd22 = 5.55555555e-05
crval1 = crval2 = 0.0
crpix = afwGeom.PointD(1000, 1000)
crval = afwCoord.Coord(afwGeom.Point2D(crval1, crval2))
wcsref = afwImage.makeWcs(crval, crpix, cd11, cd12, cd21, cd22)

# Each of the 9 has its peculiar Psf, Wcs, weight, and bounding box.
for i in range(1, 10, 1):
    record = mycatalog.getTable().makeRecord()
    psf = measAlg.DoubleGaussianPsf(100, 100, i, 1.00, 0.0)
    record.setPsf(psf)
    crpix = afwGeom.PointD(i*1000.0, i*1000.0)
    wcs = afwImage.makeWcs(crval, crpix, cd11, cd12, cd21, cd22)

    record.setWcs(wcs)
    record['customweightname'] = 1.0 * (i+1)
    record['id'] = i
    bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(i*1000, i*1000))
    record.setBBox(bbox)
    mycatalog.append(record)

# create the coaddpsf
mypsf = measAlg.CoaddPsf(mycatalog, wcsref, 'customweightname')

def makePsf(mapperResults, exposure):
    import lsst.afw.table as afwTable

    schema = afwTable.ExposureTable.makeMinimalSchema()
    schema.addField("weight", type="D", doc="Coadd weight")
    mycatalog = afwTable.ExposureCatalog(schema)

    for i, res in enumerate(mapperResults):
        subExp = res.subExposure
        record = mycatalog.getTable().makeRecord()
        record.setPsf(subExp.getPsf())
        record.setWcs(subExp.getWcs())
        record.setBBox(subExp.getBBox())
        record['weight'] = 1.0
        record['id'] = i
        mycatalog.append(record)

    # create the coaddpsf
    psf = measAlg.CoaddPsf(mycatalog, exposure.getWcs(), 'weight')
    return psf

mypsf = makePsf(results, exposure)
dit.plotImageGrid((mypsf.computeKernelImage(afwGeom.Point2D(24, 24)),
                   mypsf.computeKernelImage(afwGeom.Point2D(24, 237)),
                   mypsf.computeKernelImage(afwGeom.Point2D(24, 480)),))









import testPsfexPsfEDITED as tpp

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

psfs2b = [exposure2.getPsf().computeImage(afwGeom.Point2D(c[0], c[1])) for c in centers]
titles = ['%.0f,%.0f' % (c[0], c[1]) for c in centers]
dit.plotImageGrid(psfs2b, titles=titles)

ALres = dit.tasks.doAlInStack(template2, exposure2, doWarping=False, doDecorr=True, doPreConv=False,
            spatialBackgroundOrder=0, spatialKernelOrder=2)

newExpC = task.run(ALres.subtractedExposure, template=template2, science=exposure2,
                  alTaskResult=ALres, forceEvenSized=True).exposure

psfs3 = [newExpC.getPsf().computeImage(afwGeom.Point2D(c[0], c[1])) for c in centers]
titles = ['%.0f,%.0f' % (c[0], c[1]) for c in centers]
dit.plotImageGrid(psfs3, titles=titles)

psfs3a = [newExpC.getPsf().computeKernelImage(afwGeom.Point2D(c[0], c[1])) for c in centers]
[p.getDimensions() for p in psfs3][0:5]

print psfs3a[0].getDimensions(), psfs3a[0].getBBox(), dit.psf.computeMoments(psfs3a[0].getArray())
print psfs3a[1].getDimensions(), psfs3a[1].getBBox(), dit.psf.computeMoments(psfs3a[1].getArray())
print psfs3a[3].getDimensions(), psfs3a[3].getBBox(), dit.psf.computeMoments(psfs3a[3].getArray())

testObj2 = testObj.clone()
testObj2.ALres.decorrelatedDiffim = newExpC
testObj2.runTest(spatialKernelOrder=2)







testObj2 = testObj.clone()
testObj2.ALres.decorrelatedDiffim = newExpB
testObj2.runTest(spatialKernelOrder=2)

df, _ = testObj2.doPlotWithDetectionsHighlighted(transientsOnly=True, xaxisIsScienceForcedPhot=True);
plt.xlim(1500, 3000); plt.ylim(-0, 20);

print df.shape
print df.ix[df.Zogy_detected == True].shape
print df.ix[df.ALstack_decorr_detected == True].shape
print df.ix[df.ALstack_detected == True].shape
df[(df.Zogy_detected == True) & (df.ALstack_decorr_detected != True)]
#dit.sizeme(df)

testObj2.doPlot(imScale=6, centroidCoord=[424,66]);

psf1 = testObj.variablePsf.getImage(66, 424)
psf2 = newExpB.getPsf().computeImage(afwGeom.Point2D(66, 424)).getArray()
psf3 = testObj2.D_Zogy.psf
print psf1.shape, psf2.shape
print dit.psf.computeMoments(psf1), dit.psf.computeMoments(psf2)
dit.plotImageGrid((psf1, psf2, psf3), clim=(-0.001, 0.01))









config = dit.ZogyMapReduceConfig()
config.gridStepX = config.gridStepY = 9
#config.gridSizeX = config.gridSizeY = 10
config.borderSizeX = config.borderSizeY = 3
config.reducerSubtask.reduceOperation = 'average'
task = dit.ImageMapReduceTask(config=config)
print config

newExpZ = task.run(exposure2, template=template2, inImageSpace=False,
                       Scorr=False, forceEvenSized=True).exposure
newExpZ_Scorr = task.run(exposure2, template=template2, inImageSpace=False,
                       Scorr=True, forceEvenSized=True).exposure

testObj3 = testObj.clone()
testObj3.ALres.decorrelatedDiffim = newExpC
testObj3.D_Zogy = dit.Exposure(ga(newExpZ), testObj.D_Zogy.psf, gv(newExpZ))
testObj3.S_Zogy = dit.Exposure(ga(newExpZ_Scorr), dit.psf.afwPsfToArray(newExpZ_Scorr.getPsf()), gv(newExpZ_Scorr))
testObj3.runTest(spatialKernelOrder=2)

testObj3.doPlot(imScale=6, centroidCoord=[424,66]);

def gp(exp, coord):
    return exp.getPsf().computeImage(afwGeom.Point2D(coord[0], coord[1])).getArray()
    
dit.plotImageGrid((gp(exposure2, [80,80]), gp(newExpC, [80,80]), gp(newExpZ, [80,80]),
                   gp(exposure2, [255,255]), gp(newExpC, [255,255]), gp(newExpZ, [255,255]),
                   gp(exposure2, [480,480]), gp(newExpC, [480,480]), gp(newExpZ, [480,480]),), 
                  clim=(-0.0005, 0.001), nrows_ncols=(3, 3))



