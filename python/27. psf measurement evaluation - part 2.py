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
breakLimit = 1.050

testObj = dit.DiffimTest(imSize=(512,512), sky=sky, psf1=[1.6,1.6], psf2=[1.8,2.2],
                         offset=[0,0], psf_yvary_factor=0., 
                         #varSourceChange=[1500., 1600., 1800., 2000., 2200., 2400., 2600., 2800.],
                         varFlux2=[1500., 1500., 1500.], variablesNearCenter=False,
                         theta1=0., theta2=-45., im2background=0., n_sources=50, sourceFluxRange=(500,30000), 
                         seed=66, psfSize=21)

reload(dit)
res = dit.measurePsf(testObj.im2.asAfwExposure(), measurePsfAlg='psfex')
print res.psf.getAveragePosition()
psf = dit.afwPsfToArray(res.psf, testObj.im2.asAfwExposure()) #res.psf.computeImage()
print testObj.im2.psf.shape, psf.shape
print psf.sum()

res2 = dit.measurePsf(testObj.im2.asAfwExposure(), measurePsfAlg='pca')
print res2.psf.getAveragePosition()
psf2 = dit.afwPsfToArray(res2.psf, testObj.im2.asAfwExposure()) #res2.psf.computeImage()
print testObj.im2.psf.shape, psf2.shape
print psf2.sum()

#dit.plotImageGrid((res.psf.computeImage().getArray(),)) #, clim=(-0.001,0.001))
dit.plotImageGrid((psf,))

dit.plotImageGrid((psf2,)) #, clim=(-0.001,0.001))

print dit.computeMoments(testObj.im2.psf)
print dit.computeMoments(psf)
print dit.computeMoments(psf2)

reload(dit)
testObj2 = dit.DiffimTest(imSize=(512,512), sky=sky, psf1=[1.6,1.6], psf2=[1.8,2.2],
                         offset=[0,0], psf_yvary_factor=0., 
                         #varSourceChange=[1500., 1600., 1800., 2000., 2200., 2400., 2600., 2800.],
                         varFlux2=[1500., 1500., 1500.], variablesNearCenter=False,
                         theta1=0., theta2=-45., im2background=0., n_sources=500, sourceFluxRange=(500,30000), 
                         seed=66, psfSize=21)

#fig = plt.figure(1, (12, 12))
#dit.plotImageGrid((testObj2.im1.im,))

reload(dit)
res = dit.measurePsf(testObj2.im1.asAfwExposure(), detectThresh=10.0, measurePsfAlg='psfex')

reload(dit)
psf1 = dit.afwPsfToArray(res.psf, testObj2.im1.asAfwExposure()) #res.psf.computeImage()
print testObj2.im1.psf.shape, psf1.shape
print psf1.sum()

psf1a = psf1.copy() #/ np.abs(psf2.getArray()).sum()
psf1anorm = psf1a[np.abs(psf1a)>1e-3].sum()
print psf1a.sum()
psf1a /= psf1anorm

actualPsf1 = dit.makePsf(21, [1.6, 1.6], theta=0.)
print actualPsf1.sum()

print np.unravel_index(np.argmax(psf1a), psf1a.shape)
print np.unravel_index(np.argmax(actualPsf1), actualPsf1.shape)
#print ((actualPsf1 - psf1.getArray())**2.).sum()
print np.sqrt(((psf1a - actualPsf1)**2.).mean()) * 100.  # compare with the violinplots from ...-Copy6.ipynb

dit.plotImageGrid((psf1a, actualPsf1, actualPsf1 - psf1a), clim=(-0.001,0.002))

reload(dit)
res2 = dit.measurePsf(testObj2.im2.asAfwExposure(), detectThresh=10.0, measurePsfAlg='psfex')

print dit.computeMoments(testObj2.im1.psf)
print dit.computeMoments(testObj2.im2.psf)
print dit.computeMoments(dit.afwPsfToArray(res.psf, testObj2.im1.asAfwExposure()))
print dit.computeMoments(dit.afwPsfToArray(res2.psf, testObj2.im2.asAfwExposure()))

reload(dit)
psf2 = dit.afwPsfToArray(res2.psf, testObj2.im2.asAfwExposure())
print testObj2.im2.psf.shape, psf2.shape
print psf2.sum(), dit.computeMoments(psf2)

psf2a = psf2.copy() #/ np.abs(psf2.getArray()).sum()
psf2anorm = psf2a[np.abs(psf2a)>=1e-3].sum()
print psf2anorm, psf2a.sum()
psf2a /= psf2anorm
print psf2a.sum()

actualPsf2 = dit.makePsf(21, [1.8, 2.2], theta=-45.)
print actualPsf2.sum()

print np.unravel_index(np.argmax(psf2a), psf2a.shape)
print np.unravel_index(np.argmax(actualPsf2), actualPsf2.shape)
print np.sqrt(((psf2a - actualPsf2)**2.).mean()) * 100.  # compare with the violinplots from ...-Copy6.ipynb

dit.plotImageGrid((psf2a, actualPsf2, actualPsf2 - psf2a), clim=(-0.001,0.002))

reload(dit)

sh = dit.afwPsfToShape(res.psf, testObj2.im1.asAfwExposure())
print sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()
sh = dit.afwPsfToShape(res2.psf, testObj2.im1.asAfwExposure())
print sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()

sh = dit.arrayToAfwPsf(actualPsf1).computeShape()
print sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()
sh = dit.arrayToAfwPsf(actualPsf2).computeShape()
print sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()

print np.abs(actualPsf2).sum(), psf2.sum(), np.abs(psf2).sum()
psf2a = psf2.copy()
psf2anorm = psf2a[np.abs(psf2a)>=1e-3].sum()
print psf2anorm, psf2a.sum()
psf2a /= psf2anorm
print psf2a.sum()

plt.plot((actualPsf2)[:,20])
plt.plot((psf2a)[:,20])
plt.plot((actualPsf2 - psf2a)[:,20])

reload(dit)
testObj3 = dit.DiffimTest(imSize=(512,512), sky=sky, psf1=[1.6,1.6], psf2=[1.8,2.2],
                         offset=[0,0], psf_yvary_factor=0., 
                         #varSourceChange=[1500., 1600., 1800., 2000., 2200., 2400., 2600., 2800.],
                         varFlux2=[1500., 1500., 1500.], variablesNearCenter=False,
                         theta1=0., theta2=-45., im2background=0., n_sources=5000, sourceFluxRange=(500,30000), 
                         seed=66, psfSize=13)

#fig = plt.figure(1, (12, 12))
#dit.plotImageGrid((testObj2.im1.im,))

reload(dit)
res3 = dit.measurePsf(testObj3.im1.asAfwExposure(), detectThresh=5.0, measurePsfAlg='psfex')

reload(dit)
psf3 = dit.afwPsfToArray(res3.psf, testObj3.im1.asAfwExposure())
print testObj3.im1.psf.shape, psf3.shape
print psf3.sum()

psf3a = psf3.copy() #/ np.abs(psf2.getArray()).sum()
psf3anorm = psf3a[np.abs(psf3a)>1e-3].sum()
print psf3a.sum()
psf3a /= psf3anorm

actualPsf3 = dit.makePsf(21, [1.6, 1.6], offset=[0., 0.], theta=0.)
print actualPsf3.sum()

print np.unravel_index(np.argmax(psf3a), psf3a.shape)
print np.unravel_index(np.argmax(actualPsf3), actualPsf3.shape)
#print ((actualPsf1 - psf1.getArray())**2.).sum()
print np.sqrt(((psf3a - actualPsf3)**2.).mean()) * 100.  # compare with the violinplots from ...-Copy6.ipynb

dit.plotImageGrid((psf3a, actualPsf3, actualPsf3 - psf3a), clim=(-0.001,0.002))

res3a = dit.measurePsf(testObj3.im2.asAfwExposure(), detectThresh=5.0, measurePsfAlg='psfex')
actualPsf3a = dit.makePsf(21, [1.8, 2.2], offset=[0., 0.], theta=-45.)

sh = dit.afwPsfToShape(res3.psf, testObj3.im1.asAfwExposure())
print sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()
sh = dit.afwPsfToShape(res3a.psf, testObj3.im2.asAfwExposure())
print sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()

sh = dit.arrayToAfwPsf(actualPsf3).computeShape(); 
print sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()
sh = dit.arrayToAfwPsf(actualPsf3a).computeShape(); 
print sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()

print np.abs(actualPsf3).sum(), psf3.sum(), np.abs(psf3).sum()
psf3a = psf3.copy() #/ np.abs(psf2.getArray()).sum()
psf3anorm = psf3a[np.abs(psf3a)>=1e-2].sum()
print psf3anorm, psf2a.sum()
psf3a /= psf3anorm
print psf3a.sum()

plt.plot((actualPsf3)[:,20])
plt.plot((psf3a)[:,20])
plt.plot((actualPsf3 - psf3a)[:,20])

xgrid, ygrid = np.meshgrid(np.arange(0, psf3a.shape[0]), np.arange(0, psf3a.shape[1]))
xmoment = np.average(xgrid, weights=psf3a)
ymoment = np.average(ygrid, weights=psf3a)
print xmoment, ymoment

reload(dit)
print dit.computeMoments(psf3a)
print dit.computeMoments(actualPsf3)

def runTest(n_sources=500, seed=66):
    out = None
    try:
        testObj = dit.DiffimTest(imSize=(512,512), sky=sky, psf1=[1.6,1.6], psf2=[1.8,2.2],
                             offset=[0,0], psf_yvary_factor=0., 
                             #varSourceChange=[1500., 1600., 1800., 2000., 2200., 2400., 2600., 2800.],
                             varFlux2=[1500., 1500., 1500.], variablesNearCenter=False,
                             theta1=0., theta2=-45., im2background=0., n_sources=n_sources, 
                             sourceFluxRange=(500,30000), 
                             seed=seed, psfSize=21)
        
        try:
            im1 = testObj.im1.asAfwExposure()
            res1 = dit.measurePsf(im1, detectThresh=5.0, measurePsfAlg='psfex')
            psf1 = dit.afwPsfToArray(res1.psf, im1) #.computeImage()
            psf1a = psf1.copy() #/ np.abs(psf2.getArray()).sum()
            psf1anorm = psf1a[np.abs(psf1a)>=1e-3].sum()
            psf1a /= psf1anorm
            actualPsf1 = testObj.im1.psf #dit.makePsf(21, [1.6, 1.6], offset=[0., 0.], theta=0.)
            rms1 = np.sqrt(((psf1a - actualPsf1)**2.).mean()) #* 100.
            sh = dit.arrayToAfwPsf(actualPsf1).computeShape()
            inputShape1 = [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
            sh = dit.afwPsfToShape(res1.psf, im1)
            shape1 = [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
            moments1 = dit.computeMoments(psf1)
        except:
            rms1 = shape1 = inputShape1 = None

        try:
            im2 = testObj.im2.asAfwExposure()
            res2 = dit.measurePsf(im2, detectThresh=5.0, measurePsfAlg='psfex')
            psf2 = dit.afwPsfToArray(res2.psf, im2) #.computeImage()
            psf2a = psf2.copy() #/ np.abs(psf2.getArray()).sum()
            psf2anorm = psf2a[np.abs(psf2a)>=1e-3].sum()
            psf2a /= psf2anorm
            actualPsf2 = testObj.im2.psf #dit.makePsf(21, [1.8, 2.2], offset=[0., 0.], theta=-45.)
            rms2 = np.sqrt(((psf2a - actualPsf2)**2.).mean()) #* 100.
            sh = dit.arrayToAfwPsf(actualPsf2).computeShape()
            inputShape2 = [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
            sh = dit.afwPsfToShape(res2.psf, im2)
            shape2 = [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
            moments2 = dit.computeMoments(psf2)
        except:
            rms2 = shape2 = inputShape2 = None

        out = {'psf1': psf1, 'psf2': psf2,
               'inputPsf1': actualPsf1, 'inputPsf2': actualPsf2,
               'rms1': rms1, 'rms2': rms2, 
               'shape1': shape1, 'shape2': shape2,
               'inputShape1': inputShape1, 'inputShape2': inputShape2,
               'moments1': moments1, 'moments2': moments2,
               'nSources': n_sources, 'seed': seed}
    except Exception as e:
        pass
    return out

tmp = runTest(n_sources=50, seed=1);
#tmp

inputs = [(i, j) for i in np.insert(np.arange(50, 5000, 100), 0, [5,10,25,35]) for j in np.arange(1, 10)]
print len(inputs)
testResults1 = Parallel(n_jobs=num_cores, verbose=2)(delayed(runTest)(n_sources=i[0], seed=i[1])                                                      for i in inputs)
import cPickle; import gzip
cPickle.dump(testResults1, gzip.GzipFile('27. psf measurement evaluation - part 2.p.gz', 'wb'))

import cPickle; import gzip
testResults1 = cPickle.load(gzip.GzipFile('27. psf measurement evaluation - part 2.p.gz', 'rb'))
inputs = [(i, j) for i in np.insert(np.arange(50, 5000, 100), 0, [5,10,25,35]) for j in np.arange(1, 10)]

ns = np.array([inputs[i][0] for i in np.arange(len(inputs)) if testResults1[i] is not None])
tr = [t for t in testResults1 if t is not None]
print len(tr)
tr = {'nSources': ns,
      'rms1': np.array([t['rms1'] for t in tr])*100.,
      'rms2': np.array([t['rms2'] for t in tr])*100.,
      'rad1diff': np.array([t['shape1'][0] - t['inputShape1'][0] for t in tr]),
      'rad2diff': np.array([t['shape2'][0] - t['inputShape2'][0] for t in tr])}
tr = pd.DataFrame(tr)
sizeme(tr.head())

matplotlib.rcParams['figure.figsize'] = (20.0, 6.0)
fig, axes = plt.subplots(nrows=1, ncols=2)

g = sns.violinplot(x='nSources', y='rms1', data=tr, inner="box", cut=0, linewidth=0.3, bw=0.5, ax=axes[0],
                  scale='width')
g.set_title('RMS (template PSF)')
g.set_ylabel('PSF measurement error (RMS*100)')
g.set_xlabel('N sources')
g.set_xticklabels(g.get_xticklabels(), rotation=60);

g = sns.violinplot(x='nSources', y='rms2', data=tr, inner="box", cut=0, linewidth=0.3, bw=0.5, ax=axes[1],
                  scale='width')
g.set_title('RMS (science PSF)')
g.set_ylabel('PSF measurement error (RMS*100)')
g.set_xlabel('N sources')
g.set_xticklabels(g.get_xticklabels(), rotation=60);

matplotlib.rcParams['figure.figsize'] = (20.0, 6.0)
fig, axes = plt.subplots(nrows=1, ncols=2)

g = sns.violinplot(x='nSources', y='rad1diff', data=tr, inner="box", cut=0, linewidth=0.3, bw=0.5, ax=axes[0],
                  scale='width')
g.set_title('Radius error (template PSF)')
g.set_ylabel('Measured PSF radius - input PSF radius (pixels)')
g.set_xlabel('N sources')
g.set_xticklabels(g.get_xticklabels(), rotation=60);

g = sns.violinplot(x='nSources', y='rad2diff', data=tr, inner="box", cut=0, linewidth=0.3, bw=0.5, ax=axes[1],
                  scale='width')
g.set_title('Radius error (science PSF)')
g.set_ylabel('Measured PSF radius - input PSF radius (pixels)')
g.set_xlabel('N sources')
g.set_xticklabels(g.get_xticklabels(), rotation=60);

def computeNormedPsfRms(psf1, psf2):
    psf1a = psf1.copy() / psf1.max()
    psf2a = psf2.copy() / psf2.max()
    weights = psf1a**2.
    weights /= weights.mean()
    rms1weighted = np.sqrt(((psf1a - psf2a)**2. * weights).mean())
    return rms1weighted

rms1s = [computeNormedPsfRms(t['psf1'], t['inputPsf1']) for t in testResults1 if t is not None]
rms2s = [computeNormedPsfRms(t['psf2'], t['inputPsf2']) for t in testResults1 if t is not None]
tr['rms1'] = np.array(rms1s)
tr['rms2'] = np.array(rms2s)

matplotlib.rcParams['figure.figsize'] = (20.0, 6.0)
fig, axes = plt.subplots(nrows=1, ncols=2)

g = sns.violinplot(x='nSources', y='rms1', data=tr, inner="box", cut=0, linewidth=0.3, bw=0.5, ax=axes[0],
                  scale='width')
g.set_title('RMS (template PSF)')
g.set_ylabel('PSF measurement error (RMS)')
g.set_xlabel('N sources')
g.set_xticklabels(g.get_xticklabels(), rotation=60);

g = sns.violinplot(x='nSources', y='rms2', data=tr, inner="box", cut=0, linewidth=0.3, bw=0.5, ax=axes[1],
                  scale='width')
g.set_title('RMS (science PSF)')
g.set_ylabel('PSF measurement error (RMS)')
g.set_xlabel('N sources')
g.set_xticklabels(g.get_xticklabels(), rotation=60);

reload(dit)
n_sources = 5600  # scale up from 4000 by (512-80)**2 / 512**2 because we are not avoiding 
# the edges (this is for show, not for our actual work)
testObj = dit.DiffimTest(imSize=(512,512), sky=sky, psf1=[1.6,1.6], psf2=[1.8,2.2],
                             offset=[0,0], psf_yvary_factor=0., 
                             #varSourceChange=[1500., 1600., 1800., 2000., 2200., 2400., 2600., 2800.],
                             varFlux2=[1500., 1500., 1500.], variablesNearCenter=False,
                             theta1=0., theta2=-45., im2background=0., n_sources=n_sources, 
                             sourceFluxRange=(500,30000), avoidBorder=False,
                             seed=66, psfSize=21)

fig = plt.figure(1, (12, 12))
dit.plotImageGrid((testObj.im1.im, testObj.im2.im), clim=(-10,1000))

print 0.2*512  # arcsec per axis on the image
print 0.2*512/60/60  # degrees per axis on the image
print (0.2*512/60/60)**2  # sq. degrees on the image?
# Sanity check: if it scales to LSST focal plane 
print 189*4096*4096/512/512
print 189*4096*4096/512/512 * (0.2*512/60/60)**2  # OK, looks good.

print n_sources/((0.2*512/60/60)**2)

n_sources/0.000809086419753

12960000  # sq arcsec per sq deg
12960000 * 0.1

print ((0.2*512)**2 / 12960000)
print n_sources
n_sources / ((0.2*512)**2 / 12960000)

print 1.6*2.355*0.2  # template
print 2.2*2.355*0.2  # science



