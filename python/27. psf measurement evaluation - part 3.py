import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
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

class sizeme():
    """ Class to change html fontsize of object's representation"""
    def __init__(self,ob, size=50, height=120):
        self.ob = ob
        self.size = size
        self.height = height
    def _repr_html_(self):
        repl_tuple = (self.size, self.height, self.ob._repr_html_())
        return u'<span style="font-size:{0}%; line-height:{1}%">{2}</span>'.format(*repl_tuple)

pd.options.display.max_columns = 9999
pd.set_option('display.width', 9999)

import warnings
warnings.filterwarnings('ignore')

import diffimTests as dit
reload(dit)

# Let's try w same parameters as ZOGY paper.
sky = 300.

testObj = dit.DiffimTest(imSize=(512,512), sky=sky, psf1=[1.6,1.6], psf2=[1.8,2.2],
                         offset=[0,0], psf_yvary_factor=0., 
                         varFlux2=[1500., 1600., 1800., 2000., 2200., 2400., 2600., 2800.],
                         variablesNearCenter=False,
                         theta1=0., theta2=-45., im2background=0., n_sources=50, sourceFluxRange=(500,30000), 
                         seed=66, psfSize=21)
testObj.runTest()

reload(dit)
res = dit.measurePsf(testObj.im1.asAfwExposure(), measurePsfAlg='psfex')
testObj.im1.psf = res.psf.computeImage().getArray()

res = dit.measurePsf(testObj.im2.asAfwExposure(), measurePsfAlg='psfex')
testObj.im2.psf = res.psf.computeImage().getArray()

testObj.reset()
testObj.runTest()

reload(dit)
res = dit.measurePsf(testObj.im1.asAfwExposure(), measurePsfAlg='psfex')
testObj.im1.psf = dit.afwPsfToArray(res.psf, testObj.im1.asAfwExposure())

res = dit.measurePsf(testObj.im2.asAfwExposure(), measurePsfAlg='psfex')
testObj.im2.psf = dit.afwPsfToArray(res.psf, testObj.im2.asAfwExposure())

testObj.reset()
testObj.runTest()

reload(dit)
testObj2 = dit.DiffimTest(imSize=(512,512), sky=sky, psf1=[1.6,1.6], psf2=[1.8,2.2],
                         offset=[0,0], psf_yvary_factor=0., 
                         varFlux2=[1500., 1600., 1800., 2000., 2200., 2400., 2600., 2800.],
                         variablesNearCenter=False,
                         theta1=0., theta2=-45., im2background=0., n_sources=500, sourceFluxRange=(500,30000), 
                         seed=66, psfSize=21)
testObj2.psf1_orig = testObj2.im1.psf
testObj2.psf2_orig = testObj2.im2.psf
testObj2.runTest()

res = dit.measurePsf(testObj2.im1.asAfwExposure(), measurePsfAlg='psfex')
testObj2.im1.psf = res.psf.computeImage().getArray()

res = dit.measurePsf(testObj2.im2.asAfwExposure(), measurePsfAlg='psfex')
testObj2.im2.psf = res.psf.computeImage().getArray()

testObj2.reset()
testObj2.runTest()

reload(dit)
res = dit.measurePsf(testObj2.im1.asAfwExposure(), measurePsfAlg='psfex')
testObj2.im1.psf = dit.afwPsfToArray(res.psf, testObj2.im1.asAfwExposure())

res = dit.measurePsf(testObj2.im2.asAfwExposure(), measurePsfAlg='psfex')
testObj2.im2.psf = dit.afwPsfToArray(res.psf, testObj2.im2.asAfwExposure())

testObj2.reset()
testObj2.runTest()

print testObj2.im1.psf.max(), testObj2.psf1_orig.max()
print testObj2.im1.psf.sum(), testObj2.psf1_orig.sum()
dit.plotImageGrid((testObj2.im1.psf, testObj2.psf1_orig, testObj2.im2.psf, testObj2.psf2_orig))

fig = plt.figure(1, (16, 16))
obj = testObj2
dit.plotImageGrid((obj.res.decorrelatedDiffim, obj.D_ZOGY.im, obj.S_corr_ZOGY.im))

reload(dit)
testObj3 = dit.DiffimTest(imSize=(512,512), sky=sky, psf1=[1.6,1.6], psf2=[1.8,2.2],
                         offset=[0,0], psf_yvary_factor=0., 
                         varFlux2=[1500., 1600., 1800., 2000., 2200., 2400., 2600., 2800.],
                         variablesNearCenter=False,
                         theta1=0., theta2=-45., im2background=0., n_sources=5000, sourceFluxRange=(500,30000), 
                         seed=66, psfSize=21)
testObj3.psf1_orig = testObj3.im1.psf
testObj3.psf2_orig = testObj3.im2.psf
testObj3.runTest()

res = dit.measurePsf(testObj3.im1.asAfwExposure(), measurePsfAlg='psfex')
testObj3.im1.psf = res.psf.computeImage().getArray()

res = dit.measurePsf(testObj3.im2.asAfwExposure(), measurePsfAlg='psfex')
testObj3.im2.psf = res.psf.computeImage().getArray()

testObj3.reset()
testObj3.runTest()

print testObj3.im1.psf.max(), testObj3.psf1_orig.max()
print testObj3.im1.psf.sum(), testObj3.psf1_orig.sum()
dit.plotImageGrid((testObj3.im1.psf, testObj3.psf1_orig, testObj3.im2.psf, testObj3.psf2_orig))

reload(dit)
res = dit.measurePsf(testObj3.im1.asAfwExposure(), measurePsfAlg='psfex')
testObj3.im1.psf = dit.afwPsfToArray(res.psf, testObj3.im1.asAfwExposure())

res = dit.measurePsf(testObj3.im2.asAfwExposure(), measurePsfAlg='psfex')
testObj3.im2.psf = dit.afwPsfToArray(res.psf, testObj3.im2.asAfwExposure())

testObj3.reset()
testObj3.runTest()

fig = plt.figure(1, (16, 16))
obj = testObj3
dit.plotImageGrid((obj.res.decorrelatedDiffim, obj.D_ZOGY.im, obj.S_corr_ZOGY.im))

obj = testObj2
psf2 = obj.psf1_orig

res = dit.measurePsf(obj.im1.asAfwExposure(), measurePsfAlg='psfex')
print res.psf.getAveragePosition()
tmp1 = res.psf.computeImage().getArray()
tmp1 /= tmp1.max()

import lsst.afw.geom as afwGeom
img = obj.res.decorrelatedDiffim
bbox = img.getBBox()
xcen = (bbox.getBeginX() + bbox.getEndX()) / 2.
ycen = (bbox.getBeginY() + bbox.getEndY()) / 2.
print xcen, ycen
tmp2 = res.psf.computeImage(afwGeom.Point2D(xcen, ycen)).getArray()
tmp2 /= tmp2.max()
tmp3 = dit.afwPsfToArray(res.psf, obj.im1.asAfwExposure())
tmp3 /= tmp3.max()
dit.plotImageGrid((tmp1 - tmp2, tmp1-psf2/psf2.max(), tmp3-psf2/psf2.max(), tmp3-obj.im1.psf/obj.im1.psf.max()), 
                  clim=(-0.1,0.1))

sh = dit.arrayToAfwPsf(testObj2.psf1_orig).computeShape()
print [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
print dit.computeMoments(testObj2.psf1_orig)
sh = dit.afwPsfToShape(res.psf, obj.im1.asAfwExposure())
print [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
print dit.computeMoments(dit.afwPsfToArray(res.psf, obj.im1.asAfwExposure()))

obj = testObj2
psf2 = obj.psf2_orig

res = dit.measurePsf(obj.im2.asAfwExposure(), measurePsfAlg='psfex')
print res.psf.getAveragePosition()
tmp1 = res.psf.computeImage().getArray()
tmp1 /= tmp1.max()

import lsst.afw.geom as afwGeom
img = obj.res.decorrelatedDiffim
bbox = img.getBBox()
xcen = (bbox.getBeginX() + bbox.getEndX()) / 2.
ycen = (bbox.getBeginY() + bbox.getEndY()) / 2.
print xcen, ycen
tmp2 = res.psf.computeImage(afwGeom.Point2D(xcen, ycen)).getArray()
tmp2 /= tmp2.max()
tmp3 = dit.afwPsfToArray(res.psf, obj.im2.asAfwExposure())
tmp3 /= tmp3.max()
dit.plotImageGrid((tmp1 - tmp2, tmp1-psf2/psf2.max(), tmp3-psf2/psf2.max(), tmp3-obj.im2.psf/obj.im2.psf.max()), 
                  clim=(-0.1,0.1))

dit.plotImageGrid((tmp3 - tmp2.getArray(),))

print testObj2.im2.psf.max(), testObj2.psf2_orig.max()
print testObj2.im2.psf.sum(), testObj2.psf2_orig.sum()
dit.plotImageGrid((testObj2.im1.psf, testObj2.psf1_orig, testObj2.im2.psf, testObj2.psf2_orig), clim=(-0.001,0.001))

sh = dit.arrayToAfwPsf(testObj2.psf2_orig).computeShape()
print [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
print dit.computeMoments(testObj2.psf2_orig)
sh = dit.afwPsfToShape(res.psf, obj.im2.asAfwExposure())
print [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
print dit.computeMoments(dit.afwPsfToArray(res.psf, obj.im2.asAfwExposure()))

moments = dit.computeMoments(tmp3)
sh = dit.arrayToAfwPsf(tmp3).computeShape()
xgrid, ygrid = np.meshgrid(np.arange(0, psf2.shape[0]), np.arange(psf2.shape[1]))
rad = np.sqrt((xgrid - moments[0])**2 + (ygrid - moments[1])**2)
tmp3a = tmp3.copy()
tmp3a[(rad > sh.getDeterminantRadius() * 4.) | (np.abs(tmp3a) <= 1e-3)] = 0.
dit.plotImageGrid((tmp3a - psf2/psf2.max(),), clim=(-0.1,0.1))

plt.plot(tmp3[20,:])
plt.plot(tmp3a[20,:])
plt.plot(psf2[20,:]/psf2.max())

reload(dit)

testObj2 = dit.DiffimTest(imSize=(512,512), sky=sky, psf1=[1.6,1.6], psf2=[1.8,2.2],
                         offset=[0,0], psf_yvary_factor=0., 
                         varFlux2=[1500., 1600., 1800., 2000., 2200., 2400., 2600., 2800.],
                         variablesNearCenter=False,
                         theta1=0., theta2=-45., im2background=0., n_sources=500, sourceFluxRange=(500,30000), 
                         seed=66, psfSize=21)
testObj2.psf1_orig = testObj2.im1.psf
testObj2.psf2_orig = testObj2.im2.psf

res = dit.measurePsf(testObj2.im1.asAfwExposure(), measurePsfAlg='psfex')
testObj2.im1.psf = testObj2.psf1_fitted = dit.afwPsfToArray(res.psf, testObj2.im1.asAfwExposure())

res = dit.measurePsf(testObj2.im2.asAfwExposure(), measurePsfAlg='psfex')
testObj2.im2.psf = testObj2.psf2_fitted = dit.afwPsfToArray(res.psf, testObj2.im2.asAfwExposure())

testObj2.reset()
#print testObj2.runTest()

print testObj2.im1.psf.max(), testObj2.psf1_orig.max()
print testObj2.im1.psf.sum(), testObj2.psf1_orig.sum()
print testObj2.im1.psf.mean(), testObj2.psf1_orig.mean()
dit.plotImageGrid((testObj2.im1.psf, testObj2.psf1_orig, testObj2.im2.psf, testObj2.psf2_orig),
                 clim=(-0.0003,0.0003))

testObj2.im1.psf[testObj2.im1.psf < 0] = 0
testObj2.im1.psf[0:10,:] = testObj2.im1.psf[31:41,:] = testObj2.im1.psf[:,0:10] = testObj2.im1.psf[:,31:41] = 0
testObj2.im2.psf[testObj2.im2.psf < 0] = 0
testObj2.im2.psf[0:10,:] = testObj2.im2.psf[31:41,:] = testObj2.im2.psf[:,0:10] = testObj2.im2.psf[:,31:41] = 0
dit.plotImageGrid((testObj2.im1.psf, testObj2.psf1_orig, testObj2.im2.psf, testObj2.psf2_orig),
                 clim=(-0.0003,0.0003))

reload(dit)
testObj2a = testObj2.clone()
testObj2a.im1.psf = testObj2.psf1_fitted.copy() #orig
testObj2a.im1.psf[testObj2a.im1.psf < 0] = 0
testObj2a.im1.psf[0:10,13:41] = testObj2a.im1.psf[31:41,0:10] = 0
testObj2a.im1.psf /= testObj2a.im1.psf.sum()
testObj2a.im2.psf = testObj2.psf2_fitted.copy() #orig
testObj2a.im2.psf[testObj2a.im2.psf < 0] = 0
testObj2a.im2.psf[0:10,13:41] = testObj2a.im2.psf[31:41,0:10] = 0
testObj2a.im2.psf /= testObj2a.im2.psf.sum()

D_ZOGY1a = testObj2a.doZOGY(inImageSpace=True, padSize=15)
D_ZOGY2a = testObj2a.doZOGY(inImageSpace=False)

fig = plt.figure(1, (8, 8))
print dit.computeClippedImageStats(D_ZOGY2a.im[380:480,50:150])
print dit.computeClippedImageStats(D_ZOGY1a.im[380:480,50:150])
print dit.computeClippedImageStats(D_ZOGY2a.im[380:480,50:150] - D_ZOGY1a.im[380+1:480+1,50+1:150+1])
dit.plotImageGrid((testObj2a.im1.im[380:480,50:150],
                   D_ZOGY2a.im[380:480,50:150], D_ZOGY1a.im[380:480,50:150],
                   D_ZOGY2a.im[380:480,50:150] - D_ZOGY1a.im[380+1:480+1,50+1:150+1]))#,
                 #clim=(-0.5,0.5))

fig = plt.figure(1, (12, 12))
dit.plotImageGrid((D_ZOGY2a.im, D_ZOGY1a.im,
                  D_ZOGY2a.im - D_ZOGY1a.im))

print dit.global_dict['psf1'].shape, np.unravel_index(np.argmax(dit.global_dict['psf1']), dit.global_dict['psf1'].shape)
print dit.global_dict['padded_psf1'].shape, np.unravel_index(np.argmax(dit.global_dict['padded_psf1']), dit.global_dict['padded_psf1'].shape)
print dit.global_dict['psf2'].shape, np.unravel_index(np.argmax(dit.global_dict['psf2']), dit.global_dict['psf2'].shape)
print dit.global_dict['padded_psf2'].shape, np.unravel_index(np.argmax(dit.global_dict['padded_psf2']), dit.global_dict['padded_psf2'].shape)
print dit.global_dict['K_r'].shape, np.unravel_index(np.argmax(dit.global_dict['K_r']), dit.global_dict['K_r'].shape)
print dit.global_dict['K_n'].shape, np.unravel_index(np.argmax(dit.global_dict['K_n']), dit.global_dict['K_n'].shape)
#dit.plotImageGrid((dit.global_dict['padded_psf1'], dit.global_dict['padded_psf2']), clim=(-0.0003,0.0003))

from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
fig = plt.figure(1, (12, 12))
dit.plotImageGrid((dit.global_dict['K_r'], dit.global_dict['K_n'],
                  fftshift(dit.global_dict['K_r_hat'].real), 
                  fftshift(dit.global_dict['K_n_hat'].real)))

reload(dit)
testObj2b = testObj2.clone()
testObj2b.im1.psf = testObj2.psf1_orig
testObj2b.im2.psf = testObj2.psf2_orig
D_ZOGY1b = testObj2a.doZOGY(inImageSpace=True, padSize=0)
D_ZOGY2b = testObj2a.doZOGY(inImageSpace=False)

fig = plt.figure(1, (12, 12))
print dit.computeClippedImageStats(D_ZOGY2b.im[380:480,50:150])
print dit.computeClippedImageStats(D_ZOGY1b.im[380:480,50:150])
print dit.computeClippedImageStats(D_ZOGY2b.im[380:480,50:150] - D_ZOGY1b.im[380+1:480+1,50+1:150+1])
dit.plotImageGrid((D_ZOGY2b.im[380:480,50:150], D_ZOGY1b.im[380:480,50:150],
                  D_ZOGY2b.im[380:480,50:150] - D_ZOGY1b.im[380+1:480+1,50+1:150+1]))

fig = plt.figure(1, (12, 12))
dit.plotImageGrid((D_ZOGY2b.im, D_ZOGY1b.im,
                  D_ZOGY2b.im - D_ZOGY1b.im))

print dit.global_dict['psf1'].shape, np.unravel_index(np.argmax(dit.global_dict['psf1']), dit.global_dict['psf1'].shape)
print dit.global_dict['padded_psf1'].shape, np.unravel_index(np.argmax(dit.global_dict['padded_psf1']), dit.global_dict['padded_psf1'].shape)
print dit.global_dict['psf2'].shape, np.unravel_index(np.argmax(dit.global_dict['psf2']), dit.global_dict['psf2'].shape)
print dit.global_dict['padded_psf2'].shape, np.unravel_index(np.argmax(dit.global_dict['padded_psf2']), dit.global_dict['padded_psf2'].shape)
print dit.global_dict['K_r'].shape, np.unravel_index(np.argmax(dit.global_dict['K_r']), dit.global_dict['K_r'].shape)
print dit.global_dict['K_n'].shape, np.unravel_index(np.argmax(dit.global_dict['K_n']), dit.global_dict['K_n'].shape)
fig = plt.figure(1, (12, 12))

from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
dit.plotImageGrid((dit.global_dict['K_r'], dit.global_dict['K_n'],
                  fftshift(dit.global_dict['K_r_hat'].real), 
                   fftshift(dit.global_dict['K_n_hat'].real)))

from numpy.fft import fft2, ifft2, fftshift, ifftshift
# From astropy.photutils.psf.matching.windows
def _radial_distance(shape):
    """
    Return an array where each value is the Euclidean distance from the
    array center.
    Parameters
    ----------
    shape : tuple of int
        The size of the output array along each axis.
    Returns
    -------
    result : `~numpy.ndarray`
        An array containing the Euclidian radial distances from the
        array center.
    """

    if len(shape) != 2:
        raise ValueError('shape must have only 2 elements')
    position = (np.asarray(shape) - 1) / 2.
    x = np.arange(shape[1]) - position[1]
    y = np.arange(shape[0]) - position[0]
    xx, yy = np.meshgrid(x, y)
    return np.sqrt(xx**2 + yy**2)

K_r_hat = fftshift(dit.global_dict['K_n_hat'])
K_r = dit.global_dict['K_n']
rd = _radial_distance(K_r_hat.shape)
th = (rd <= 65) * 1.0
print th.min(), th.max()
K_r_h = K_r_hat * th
print K_r_hat.real.min(), K_r_hat.real.max()
print K_r_h.real.min(), K_r_h.real.max()
kernel = np.real(ifft2(ifftshift(K_r_h)))
dit.plotImageGrid((K_r_hat.real, K_r_h.real, kernel, K_r))  # rd, th, 



