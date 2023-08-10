import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
import diffimTests as dit

import warnings
warnings.filterwarnings('ignore')

testObj = dit.DiffimTest(varFlux2=np.repeat(620., 50), sourceFluxRange=(600, 120000))

testObj.doPlot();

actualPsf1 = testObj.im2.psf.copy()
im2 = testObj.im2.asAfwExposure()
res2 = dit.tasks.doMeasurePsf(im2, detectThresh=5.0, measurePsfAlg='psfex', spatialOrder=1)

res2.cellSet

bbox = im2.getBBox()
xcen = (bbox.getBeginX() + bbox.getEndX()) / 2.
ycen = (bbox.getBeginY() + bbox.getEndY()) / 2.
bbox, xcen, ycen

import lsst.afw.geom as afwGeom
img = res2.psf.computeImage(afwGeom.Point2D(xcen, ycen))
dit.plotImageGrid((img,))

psf = res2.psf

actualPsf1 = testObj.im2.psf.copy()
im2 = testObj.im2.asAfwExposure()
res2 = dit.tasks.doMeasurePsf(im2, detectThresh=20.0, measurePsfAlg='pca', spatialOrder=0)
len(res2.cellSet.getCellList())

img = res2.psf.computeImage(afwGeom.Point2D(xcen, ycen))

dit.plotImageGrid((img,))

n_runs = 1
testResults1 = dit.multi.runMultiDiffimTests(varSourceFlux=620., n_runs=n_runs, 
                                             remeasurePsfs=[False, True])

tr = testResults1[0]

tr['psfInfo']

testObj = dit.DiffimTest(varFlux2=np.repeat(620., 50), n_sources=50, sourceFluxRange=(600, 120000))

testObj.doPlot();

testObj = dit.DiffimTest(varFlux2=np.repeat(620., 10), n_sources=3000, sourceFluxRange=(600, 120000),
                        avoidAllOverlaps=15., templateNoNoise=True, skyLimited=False)

testObj.doPlot();

res = testObj.runTest(returnSources=True)

del res['sources']
res

res1 = dit.multi.runTest(flux=620., n_varSources=50, n_sources=300, returnObj=True,
                        templateNoNoise=True, skyLimited=True)
#del res1['result']['sources']
res1['result']

res2 = dit.multi.runTest(flux=620., n_varSources=50, n_sources=300, 
                         templateNoNoise=True, skyLimited=True,
                         remeasurePsfs=[False, True], returnObj=True)
#del res2['result']['sources']
res2['result']

dit.plotImageGrid((res2['psfInfo']['inputPsf2'], res2['psfInfo']['psf2']))

res1['obj'].doPlot(centroidCoord=[284,235,50]);

res2['obj'].doPlot(centroidCoord=[284,235,50]);



