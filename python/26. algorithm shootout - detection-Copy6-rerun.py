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
                         varFlux2=[1500., 1500., 1500.],
                         theta1=0., theta2=-45., im2background=0., n_sources=50, sourceFluxRange=(500,30000), 
                         seed=66, psfSize=13)

# Tweak the saved PSFs so the science PSF sigma is slightly wrong ("mis-measured"):
P_n = testObj.im2.psf
testObj.im2.psf = dit.makePsf(psfSize=13, sigma=[1.8*breakLimit,2.2*breakLimit], theta=-45.*breakLimit)

print dit.computeClippedImageStats(testObj.im1.im)
print dit.computeClippedImageStats(testObj.im2.im)
print dit.computeClippedImageStats(testObj.im1.var)
print dit.computeClippedImageStats(testObj.im2.var)

print P_n.sum(), testObj.im2.psf.sum()
dit.plotImageGrid((P_n, testObj.im2.psf, P_n-testObj.im2.psf))

print testObj.runTest()

#testObj.doAL(spatialKernelOrder=0, spatialBackgroundOrder=1)
D_ZOGY = testObj.doZOGY(inImageSpace=False)
testObj.doZOGY(inImageSpace=True)
res = testObj.doALInStack()

xim = np.arange(-256, 256, 1)
yim = xim.copy()

D_stack = res.decorrelatedDiffim.getMaskedImage().getImage().getArray()

fig = plt.figure(1, (12, 12))
x1d, x2d, y1d, y2d = 150, 512-150, 150, 512-150   # limits for display
extent = (xim.min()+150, xim.max()-150, yim.min()+150, yim.max()-150)
dit.plotImageGrid((testObj.im1.im[x1d:x2d,y1d:y2d]/10., 
                   D_ZOGY.im[x1d:x2d,y1d:y2d], 
                   testObj.S_corr_ZOGY.im[x1d:x2d,y1d:y2d], 
                   #testObj.D_AL.im[x1d:x2d,y1d:y2d]), 
                   D_stack[x1d:x2d,y1d:y2d]/20.),
                  clim=(-2,2), titles=['Template', 'ZOGY(FT)', 'S_ZOGY(real)', 'A&L'])

reload(dit)
testObj = dit.DiffimTest(imSize=(512,512), sky=sky, psf1=[1.6,1.6], psf2=[1.8,2.2],
                         offset=[0,0], psf_yvary_factor=0., 
                         #varSourceChange=[1500., 1600., 1800., 2000., 2200., 2400., 2600., 2800.],
                         varFlux2=[1500., 1500., 1500.],
                         scintillation=0.1,
                         theta1=0., theta2=-45., im2background=0., n_sources=50, sourceFluxRange=(500,30000), 
                         seed=66, psfSize=13)

# Tweak the saved PSFs so the science PSF sigma is slightly wrong ("mis-measured"):
P_n = testObj.im2.psf
testObj.im2.psf = dit.makePsf(psfSize=13, sigma=[1.8*breakLimit,2.2*breakLimit], theta=-45.*breakLimit)
print testObj.astrometricOffsets, np.sqrt(np.sum(testObj.astrometricOffsets))

print testObj.runTest()

#testObj.doAL(spatialKernelOrder=0, spatialBackgroundOrder=1)
D_ZOGY = testObj.doZOGY(inImageSpace=False)
S_corr_ZOGY = testObj.S_corr_ZOGY
S_ZOGY = testObj.S_ZOGY
testObj.doZOGY(inImageSpace=True)
res = testObj.doALInStack()

xim = np.arange(-256, 256, 1)
yim = xim.copy()

D_stack = res.decorrelatedDiffim.getMaskedImage().getImage().getArray()

fig = plt.figure(1, (12, 12))
x1d, x2d, y1d, y2d = 150, 512-150, 150, 512-150   # limits for display
extent = (xim.min()+150, xim.max()-150, yim.min()+150, yim.max()-150)
print dit.computeClippedImageStats(S_corr_ZOGY.im[x1d:x2d,y1d:y2d])
print dit.computeClippedImageStats(S_ZOGY.im[x1d:x2d,y1d:y2d])
dit.plotImageGrid((testObj.im1.im[x1d:x2d,y1d:y2d]/10., 
                   D_ZOGY.im[x1d:x2d,y1d:y2d], 
                   S_corr_ZOGY.im[x1d:x2d,y1d:y2d], 
                   #testObj.D_AL.im[x1d:x2d,y1d:y2d]), 
                   D_stack[x1d:x2d,y1d:y2d]/20.),
                  clim=(-2,2), titles=['Template', 'ZOGY(FT)', 'S_ZOGY(real)', 'A&L'])

# Default 10 sources with same flux
def runTest(flux, seed=66, sky=300., n_sources=50, n_varSources=1, percentagePsfMismeasure=0., scint=0.1):
    #methods = ['ALstack', 'ZOGY', 'ZOGY_S', 'ALstack_noDecorr']
    testObj = dit.DiffimTest(imSize=(512,512), sky=sky, psf1=[1.6,1.6], psf2=[1.8,2.2],
                             offset=[0,0], psf_yvary_factor=0., 
                             scintillation=scint,
                             varFlux2=np.repeat(flux, n_varSources),
                             theta1=0., theta2=-45., im2background=0., n_sources=n_sources, 
                             sourceFluxRange=(500,30000), seed=seed, psfSize=13)
    #testObj.im2.psf = dit.makePsf(psfSize=13, sigma=[1.9,2.1], theta=-35.)
    ppm = 1.0 + percentagePsfMismeasure / 100.
    origPsf = testObj.im2.psf
    testObj.im2.psf = dit.makePsf(psfSize=13, sigma=[1.8*ppm,2.2*ppm], theta=-45.*ppm)
    det = testObj.runTest() #subtractMethods=methods) #, 'AL'])
    det['flux'] = flux
    det['ppm'] = percentagePsfMismeasure
    det['rms'] = np.sqrt(((origPsf - testObj.im2.psf)**2.).mean())
    #print percentagePsfMismeasure, det['rms']
    return det

methods = ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']

def plotResults(tr):
    TP = pd.DataFrame({key: np.array([t[key]['TP'] for t in tr]) for key in methods})
    FN = pd.DataFrame({key: np.array([t[key]['FN'] for t in tr]) for key in methods})
    FP = pd.DataFrame({key: np.array([t[key]['FP'] for t in tr]) for key in methods})

    TP['rms'] = np.array([t['rms'] for t in tr])
    FP['rms'] = np.array([t['rms'] for t in tr])
    FN['rms'] = np.array([t['rms'] for t in tr])

    matplotlib.rcParams['figure.figsize'] = (20.0, 6.0)
    fig, axes = plt.subplots(nrows=1, ncols=2)

    tmp1 = TP[['rms','ALstack']]
    tmp2 = TP[['rms', 'SZOGY']]
    tmp1['method'] = np.repeat('ALstack', tmp1.shape[0])
    tmp2['method'] = np.repeat('SZOGY', tmp2.shape[0])
    tmp1.columns.values[1] = tmp2.columns.values[1] = 'TP'
    tmp = pd.concat((tmp1, tmp2))
    tmp['rms'] = np.round(tmp['rms']*100., 3)  # make it RMS instead of RMS

    g = sns.violinplot(x='rms', y='TP', data=tmp, split=True, hue='method', inner="box", cut=0, 
                   linewidth=0.3, bw=0.5, ax=axes[0])
    g.set_title('True Positives')
    g.set_ylim((3, 10))
    g.set_xlabel('PSF measurement error (RMS*100)')
    g.set_xticklabels(g.get_xticklabels(), rotation=30);

    tmp1 = FP[['rms','ALstack']]
    tmp2 = FP[['rms', 'SZOGY']]
    tmp1['method'] = np.repeat('ALstack', tmp1.shape[0])
    tmp2['method'] = np.repeat('SZOGY', tmp2.shape[0])
    tmp1.columns.values[1] = tmp2.columns.values[1] = 'FP'
    tmp = pd.concat((tmp1, tmp2))
    tmp['rms'] = np.round(tmp['rms']*100., 3)

    g = sns.violinplot(x='rms', y='FP', data=tmp, split=True, hue='method', inner="box", cut=0, 
                   linewidth=0.3, bw=0.5, ax=axes[1], width=0.98)
    g.set_title('False Positives')
    g.set_ylim((-1, 15))
    g.set_xlabel('PSF measurement error (RMS*100)')
    g.set_xticklabels(g.get_xticklabels(), rotation=30);

errorVals = [0,2,4,6,8,10,12]
inputs = [(f, seed, ppm) for f in [1500] for seed in np.arange(66, 116, 1) for ppm in errorVals]
print len(inputs)
testResults1 = Parallel(n_jobs=num_cores, verbose=2)(delayed(runTest)(i[0], i[1], n_sources=100, n_varSources=10,                                                                       percentagePsfMismeasure=i[2])                                                      for i in inputs)

plotResults(testResults1)

errorVals = [0,2,4,6,8,10,12]
inputs = [(f, seed, ppm) for f in [1500] for seed in np.arange(66, 116, 1) for ppm in errorVals]
print len(inputs)
testResults2 = Parallel(n_jobs=num_cores, verbose=2)(delayed(runTest)(i[0], i[1], n_sources=100, n_varSources=10,                                                                       percentagePsfMismeasure=i[2], scint=0.05)                                                      for i in inputs)

plotResults(testResults2)



