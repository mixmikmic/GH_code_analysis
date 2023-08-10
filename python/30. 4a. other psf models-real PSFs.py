import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
#%matplotlib notebook
#import matplotlib.pylab as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
num_cores = multiprocessing.cpu_count()
if num_cores == 32:
    num_cores = 24  # lsst-dev - don't use all the cores, man.
elif num_cores == 8:
    num_cores = 3
elif num_cores == 4:
    num_cores = 2
print num_cores

import seaborn as sns
sns.set(style="whitegrid", palette="pastel", color_codes=True)

pd.options.display.max_columns = 9999
pd.set_option('display.width', 9999)

import warnings
warnings.filterwarnings('ignore')

import diffimTests as dit

# Set up console so we can reattach via terminal ipython later. See:
# https://stackoverflow.com/questions/19479645/using-ipython-console-along-side-ipython-notebook

get_ipython().magic('qtconsole')

# Then do `ipython console --existing` in a terminal to connect and have access to same data!
# But note, do not do CTRL-D in that terminal or it will kill the kernel!

reload(dit)
testObj = dit.diffimTests.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=15.)

res = testObj.runTest(returnSources=True)
src = res['sources']
del res['sources']
print res

tmp = dit.catalogToDF(testObj.getCentroidsCatalog(transientsOnly=False))
#dit.sizeme(tmp.tail())
testObj.doPlot(centroidCoord=[tmp.centroid_y.values[300], tmp.centroid_x.values[300]]);

testObj.doPlotWithDetectionsHighlighted(transientsOnly=True, divideByInput=False, addPresub=True)
plt.xlim(0, 2000)
plt.ylim(-2, 20)
plt.title('Default Gaussian PSFs')

reload(dit)
testObj2 = dit.diffimTests.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=15., psfType='doubleGaussian')

res2 = testObj2.runTest(returnSources=True)
src2 = res2['sources']
del res2['sources']
print res2

testObj2.doPlot(centroidCoord=[tmp.centroid_y.values[300], tmp.centroid_x.values[300]]);

testObj2.doPlotWithDetectionsHighlighted(transientsOnly=True, divideByInput=False, addPresub=True)
plt.xlim(0, 2000)
plt.ylim(-2, 20)
plt.title('Double Gaussian PSFs')

reload(dit)
testObj3 = dit.diffimTests.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=15., psfType='moffat')

res3 = testObj3.runTest(returnSources=True)
src3 = res3['sources']
del res3['sources']
print res3

testObj3.doPlot(centroidCoord=[tmp.centroid_y.values[300], tmp.centroid_x.values[300]]);

testObj3.doPlotWithDetectionsHighlighted(transientsOnly=True, divideByInput=False, addPresub=True)
plt.xlim(0, 2000)
plt.ylim(-2, 20)
plt.title('Moffat PSFs')

reload(dit)
testObj4 = dit.diffimTests.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=15., psfType='kolmogorov', psfSize=65)

res4 = testObj4.runTest(returnSources=True)
src4 = res4['sources']
del res4['sources']
print res4

tmp = dit.catalogToDF(testObj4.getCentroidsCatalog(transientsOnly=False))
testObj4.doPlot(centroidCoord=[tmp.centroid_y.values[300], tmp.centroid_x.values[300]]);
#testObj4.doPlot();

testObj4.doPlotWithDetectionsHighlighted(transientsOnly=True, divideByInput=False, addPresub=True)
plt.xlim(0, 2000)
plt.ylim(-2, 20)
plt.title('Kolmogorov PSFs')

import diffimTests as dit
import glob, os

afwData = os.getenv('AFWDATA_DIR')
drpData = os.getenv('HOME') + '/DATA/'

#fnames = glob.glob('./psfLib/*.fits')
#fnames = [fn.replace('_psf', '') for fn in fnames]
filenames = glob.glob(afwData + '/CFHT/D4/*.fits')
# PsfEx doesn't seem to work well on the imsim simulated images.
filenames.extend(glob.glob(afwData + '/ImSim/postISR/v85751839-fr/s1/R23/S11/*.fits'))
filenames.extend(glob.glob(drpData + '/validation_data_decam/data/*/calexp/calexp*.fits'))
filenames.extend(glob.glob(drpData + './validation_data_cfht/data/calexp/06AL01/D3/2006-*/r/calexp*.fits'))
filenames.sort()

psfs = {}
for filename in filenames:
    #print filename
    try:
        psf, source = dit.psf.loadPsf(filename, asArray=False)
        if psf is not None:
            printedFilename = filename.replace(afwData, '')
            printedFilename = printedFilename.replace(drpData, '')
            print printedFilename, dit.afw.afwPsfToShape(psf)
            psfs[filename] = psf
    except Exception as e:
        pass
        #print e
        
print len(psfs)

psftmp = [dit.afw.afwPsfToArray(psfs[key]) for key in sorted(psfs.keys())]
psftitles = [os.path.basename(key) for key in sorted(psfs.keys())]
psftitles = [t for i,t in enumerate(psftitles) if psftmp[i] is not None]
psftmp = [p for p in psftmp if p is not None]
print len(psftmp)
#print psftitles
dit.plotImageGrid(psftmp, titles=psftitles)

afwData = os.getenv('AFWDATA_DIR')
filename = afwData + '/CFHT/D4/cal-53535-i-797722_6_tmpl.fits'
psf, source = dit.psf.loadPsf(filename, asArray=False)
print type(psf)
from lsst.afw.detection import Psf
print isinstance(psf, Psf)
psfImg = dit.afw.afwPsfToArray(psf)
psfImg2 = dit.afw.afwPsfToArray(psf, centroid=[0.0, 0.45])
psfImg3 = dit.afw.afwPsfToArray(psf, centroid=[0.45, 0.0])
print type(psfImg)
print isinstance(psfImg, np.ndarray)

filename2 = afwData + '/CFHT/D4/cal-53535-i-797722_6.fits'
psf2, source = dit.psf.loadPsf(filename2, asArray=False)
psf2Img = dit.afw.afwPsfToArray(psf2)

#psfImg2 = dit.afw.afwPsfToArray(psf2, centroid=[0.0, 0.45])
#psfImg3 = dit.afw.afwPsfToArray(psf2, centroid=[0.45, 0.0])

sh = dit.afw.afwPsfToShape(psf)
print sh, (sh.getIxx() + sh.getIyy()) / 2.
sh = dit.afw.afwPsfToShape(psf2)
print sh, (sh.getIxx() + sh.getIyy()) / 2.

print dit.psf.computeMoments(dit.afw.afwPsfToArray(psf))
print dit.psf.computeMoments(dit.afw.afwPsfToArray(psf2))

print dit.psf.computeMoments(psfImg)
print dit.psf.computeMoments(psfImg2)
print dit.psf.computeMoments(psfImg3)

dit.plotImageGrid((psfImg, psfImg2, psfImg2-psfImg, psf2Img, psfImg3-psfImg))

tmp = dit.psf.makePsf(psfType=psf)
print dit.psf.computeMoments(tmp)

tmp = dit.psf.makePsf(psfType=psf, offset=[0.5, 0.5])
print dit.psf.computeMoments(tmp)

tmp = dit.psf.makePsf(psfType='gaussian')
print dit.psf.computeMoments(tmp)

tmp = dit.psf.makePsf(psfType='gaussian', offset=[0.5, 0.5])
print dit.psf.computeMoments(tmp)

tmp = dit.psf.makePsf(psfType='kolmogorov')
print dit.psf.computeMoments(tmp)

tmp = dit.psf.makePsf(psfType='kolmogorov', offset=[0.5, 0.5])
print dit.psf.computeMoments(tmp)

filename = afwData + '/CFHT/D4/cal-53535-i-797722_6_tmpl.fits'
psf, source = dit.psf.loadPsf(filename, asArray=False)
psf = dit.psf.recenterPsf(dit.afw.afwPsfToArray(psf))
filename2 = afwData + '/CFHT/D4/cal-53535-i-797722_6.fits'
psf2, source = dit.psf.loadPsf(filename2, asArray=False)
psf2 = dit.psf.recenterPsf(dit.afw.afwPsfToArray(psf2))
print dit.afw.arrayToAfwPsf(psf).computeShape(), '\n', dit.psf.computeMoments(psf)
print dit.afw.arrayToAfwPsf(psf2).computeShape(), '\n', dit.psf.computeMoments(psf2)

dit.plotImageGrid((psf, psf2), same_zscale=True)

reload(dit)
testObj5 = dit.diffimTests.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50), #offset=[0.05, 0.],
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=15., psfType=[psf, psf2], 
                         psf1=1., psf2=1., theta1=0.) # to prevent tweaking the PSFs

res5 = testObj5.runTest(returnSources=True)
src5 = res5['sources']
del res5['sources']
print res5

tmp = dit.catalogToDF(testObj5.getCentroidsCatalog(transientsOnly=False))

testObj5.doPlot(centroidCoord=[tmp.centroid_y.values[300], tmp.centroid_x.values[300]], 
                same_zscale=True);

tmp5 = testObj5.doPlotWithDetectionsHighlighted(transientsOnly=True, divideByInput=False, addPresub=True)
plt.xlim(0, 2000)
plt.ylim(-1., 20)
plt.title('Real PSF');

column='base_GaussianCentroid'
fluxCol='base_PsfFlux'

src1 = testObj5.im1.doDetection(asDF=True)
src1 = src1[~src1[column + '_flag'] & ~src1[fluxCol + '_flag']]
src1 = src1[[column + '_x', column + '_y', fluxCol + '_flux']]
src1.reindex()

# src2 = self.im2.doDetection(asDF=True)
# src2 = src2[~src2[column + '_flag'] & ~src2[fluxCol + '_flag']]
# src2 = src2[[column + '_x', column + '_y', fluxCol + '_flux']]
# src2.reindex()

src2 = tmp[['centroid_x', 'centroid_y', 'inputFlux_science']]

dx, dy, _ = dit.catalog.computeOffsets(src1, src2, threshold=2.5)
dx, dy

filename = afwData + '/CFHT/D4/cal-53535-i-797722_6_tmpl.fits'
#psf, source = dit.psf.loadPsf(filename, asArray=False)
filename2 = afwData + '/CFHT/D4/cal-53535-i-797722_6.fits'
#psf2, source = dit.psf.loadPsf(filename2, asArray=False)

reload(dit)
testObj5a = dit.diffimTests.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50), #offset=[0.05, 0.],
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=15., psfType=[filename, filename2],
                         psf1=1., psf2=1., theta1=0.) # to prevent tweaking the PSFs

res5 = testObj5a.runTest(returnSources=True)
src5 = res5['sources']
del res5['sources']
print res5

tmp = dit.catalogToDF(testObj5a.getCentroidsCatalog(transientsOnly=False))

testObj5a.doPlot(centroidCoord=[tmp.centroid_y.values[300], tmp.centroid_x.values[300]], 
                same_zscale=True);

tmp5 = testObj5a.doPlotWithDetectionsHighlighted(transientsOnly=False, divideByInput=False, addPresub=True)
plt.xlim(0, 20000)
plt.ylim(-1., 150)
plt.title('Real PSF');

get_ipython().magic('timeit dit.psf.loadPsf(filename, asArray=False)  # without memoize is about 1 ms.')

def runit(i):
    testObj = dit.diffimTests.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                             varFlux2=np.linspace(200, 2000, 50),
                             #varFlux2=np.repeat(500., 50),
                             templateNoNoise=True, skyLimited=True,
                            avoidAllOverlaps=15., psfType='moffat', seed=666+i,
                            psf1=1.6/1.1, psf2=2.1/1.1, theta1=0.) # to prevent tweaking the PSFs

    res = testObj.runTest(returnSources=True, matchDist=np.sqrt(1.5))
    src = res['sources']
    del res['sources']

    cats = testObj.doForcedPhot(transientsOnly=False)
    sources, fp1, fp2, fp_ZOGY, fp_AL, fp_ALd = cats

    meas = fp2['base_PsfFlux_flux']/fp2['base_PsfFlux_fluxSigma']
    calc = testObj.im2.calcSNR(sources['inputFlux_science'], skyLimited=True)
    print i, np.median(meas/calc), np.median(meas[calc>60]/calc[calc>60])
    return calc, meas

res = Parallel(n_jobs=num_cores, verbose=2)(delayed(runit)(i) for i in range(20))

calcss, meass = zip(*res)
calcss = np.array(calcss).flatten()
meass = np.array(meass).flatten()
#calcs = np.append(calcs, calc)
#meass = np.append(meass, meas)
    
print np.median(meass/calcss), np.median(meass[calcss>60]/calcss[calcss>60])
print dit.mad(meass/calcss), dit.mad(meass[calcss>60]/calcss[calcss>60])
plt.scatter(calcss, meass/calcss)
plt.xlim(0, 170)
plt.ylim(0.8, 1.2);
plt.xlabel('Input SNR (science)')
plt.ylabel('Measured SNR / Input SNR (science)')

import scipy.interpolate
spl = scipy.interpolate.UnivariateSpline(calcss, meass/calcss, s=1000)
xs = np.linspace(calcss.min(), calcss.max(), 1000)
plt.plot(xs, spl(xs), 'g', lw=3);

testObj4 = dit.diffimTests.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=True, skyLimited=True,
                        avoidAllOverlaps=15., psfType='moffat', seed=666,
                        psf1=1.6/1.1, psf2=2.1/1.1, theta1=0.) # to prevent tweaking the PSFs

print testObj4.im1.psf.sum(), testObj4.im1.psf.min(), testObj4.im2.psf.sum(), testObj4.im2.psf.min()
print dit.psf.computeMoments(testObj4.im1.psf), dit.psf.computeMoments(testObj4.im2.psf)
print dit.afw.arrayToAfwPsf(testObj4.im1.psf).computeShape()
print dit.afw.arrayToAfwPsf(testObj4.im2.psf).computeShape()

testObj = dit.diffimTests.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=True, skyLimited=True,
                        avoidAllOverlaps=15., psfType='gaussian', seed=666,
                        psf1=1.6, psf2=2.1, theta1=0.) # to prevent tweaking the PSFs

print testObj.im1.psf.sum(), testObj.im1.psf.min(), testObj.im2.psf.sum(), testObj.im2.psf.min()
print dit.psf.computeMoments(testObj.im1.psf), dit.psf.computeMoments(testObj.im2.psf)
print dit.afw.arrayToAfwPsf(testObj.im1.psf).computeShape()
print dit.afw.arrayToAfwPsf(testObj.im2.psf).computeShape()

dit.plotImageGrid((testObj4.im1.psf, testObj4.im2.psf, testObj.im1.psf, testObj.im2.psf),
                 clim=(-0.000001, 0.000001))

cats = testObj4.doForcedPhot(transientsOnly=False)
sources, fp1, fp2, fp_ZOGY, fp_AL, fp_ALd = cats

meas = fp2['base_PsfFlux_flux']/fp2['base_PsfFlux_fluxSigma']
calc = testObj.im2.calcSNR(sources['inputFlux_science'], skyLimited=True)
print np.median(meas/calc), np.median(meas[calc>60]/calc[calc>60])

psf = testObj4.im2.psf
psf = psf / psf.max()
print psf.min(), psf.max()
nPix = np.sum(psf) * 2.  # not sure where the 2 comes from but it works.
radius = 2.1/1.1
print nPix, np.pi*radius*radius*4  # and it equals pi*r1*r2*4.

1/0.954465334102, 54.0549070053/45.7997474457

tmp = testObj.doPlotWithDetectionsHighlighted(transientsOnly=False, divideByInput=False, addPresub=True)
plt.xlim(0, 20000)
plt.ylim(-1., 150)
plt.title('Default Gaussian PSF');



