import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
num_cores = multiprocessing.cpu_count()

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

import diffimTests as dit
reload(dit)

# Let's try w same parameters as ZOGY paper.
sky = 300.

testObj = dit.DiffimTest(imSize=(512,512), sky=sky, offset=[0,0], psf_yvary_factor=0., 
                         varSourceChange=[1500., 1600., 1800., 2000., 2200., 2400., 2600., 2800.],
                         theta1=0., theta2=-45., im2background=0., n_sources=50, sourceFluxRange=(500,30000), 
                         seed=66, psfSize=13)

print dit.computeClippedImageStats(testObj.im1.im)
print dit.computeClippedImageStats(testObj.im2.im)
print dit.computeClippedImageStats(testObj.im1.var)
print dit.computeClippedImageStats(testObj.im2.var)

src = testObj.runTest(returnSources=True)

changedCentroid = np.array(testObj.centroids[testObj.changedCentroidInd, :])
print changedCentroid

#print src['AL'][['base_NaiveCentroid_x', 'base_NaiveCentroid_y', 'base_PsfFlux_fluxSigma', 'base_PsfFlux_flag']]
print src['ALstack'][['base_NaiveCentroid_x', 'base_NaiveCentroid_y', 'base_PsfFlux_fluxSigma', 'base_PsfFlux_flag']]
print src['ALstack_noDecorr'][['base_NaiveCentroid_x', 'base_NaiveCentroid_y', 'base_PsfFlux_fluxSigma', 'base_PsfFlux_flag']]
print src['ZOGY'][['base_NaiveCentroid_x', 'base_NaiveCentroid_y', 'base_PsfFlux_fluxSigma', 'base_PsfFlux_flag']]
# For SZOGY, the correct flux measurement is PeakLikelihoodFlux
print src['SZOGY'][['base_NaiveCentroid_x', 'base_NaiveCentroid_y', 'base_PeakLikelihoodFlux_fluxSigma', 'base_PsfFlux_flag']]

dist = np.sqrt(np.add.outer(src['ALstack'].base_NaiveCentroid_x, -changedCentroid[:, 0])**2. +                np.add.outer(src['ALstack'].base_NaiveCentroid_y, -changedCentroid[:, 1])**2.) # in pixels
print dist
matches = np.where(dist <= 1.5)
print matches
true_pos = len(matches[0])
false_neg = changedCentroid.shape[0] - len(matches[0])
false_pos = src['ALstack'].shape[0] - len(matches[0])
print true_pos, false_neg, false_pos

dist = np.sqrt(np.add.outer(src['SZOGY'].base_NaiveCentroid_x, -changedCentroid[:, 0])**2. +                np.add.outer(src['SZOGY'].base_NaiveCentroid_y, -changedCentroid[:, 1])**2.) # in pixels
print dist
matches = np.where(dist <= 1.5)
true_pos = len(matches[0])
false_neg = changedCentroid.shape[0] - len(matches[0])
false_pos = src['SZOGY'].shape[0] - len(matches[0])
print true_pos, false_neg, false_pos

get_ipython().magic('pinfo2 dit.runTest')

reload(dit)

# Let's try w same parameters as ZOGY paper.
sky = 300.

testObj = dit.DiffimTest(imSize=(512,512), sky=sky, offset=[0,0], psf_yvary_factor=0., 
                         varSourceChange=[1500., 1600., 1800., 2000., 2200., 2400., 2600., 2800.],
                         theta1=0., theta2=-45., im2background=0., n_sources=50, sourceFluxRange=(500,30000), 
                         seed=66, psfSize=13)

det = testObj.runTest()
print det

# Default 10 sources with same flux
def runTest(flux, seed=66, sky=300., n_sources=50, n_varSources=10):
    sky = 300.
    testObj = dit.DiffimTest(imSize=(512,512), sky=sky, offset=[0,0], psf_yvary_factor=0., 
                             varSourceChange=np.repeat(flux, n_varSources),
                             theta1=0., theta2=-45., im2background=0., n_sources=n_sources, 
                             sourceFluxRange=(500,30000), seed=seed, psfSize=13)
    det = testObj.runTest(subtractMethods=['ALstack', 'ZOGY', 'ZOGY_S', 'ALstack_noDecorr']) #, 'AL'])
    det['flux'] = flux
    return det

#testResults = [runTest(f, seed) for f in [1500, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000] for \
#              seed in [66, 67, 68, 69, 70]]

inputs = [(f, seed) for f in np.arange(1500, 3001, 100) for seed in np.arange(66, 86, 1)]
print len(inputs)
testResults = Parallel(n_jobs=num_cores, verbose=2)(delayed(runTest)(i[0], i[1]) for i in inputs)

tr = testResults
FN = {key: np.array([t[key]['FN'] for t in tr]).mean() for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']}
FP = {key: np.array([t[key]['FP'] for t in tr]).mean() for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']}
TP = {key: np.array([t[key]['TP'] for t in tr]).mean() for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']}
print 'FN:', FN
print 'FP:', FP
print 'TP:', TP

FN = pd.DataFrame({key: np.array([t[key]['FN'] for t in tr]) for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']})
FP = pd.DataFrame({key: np.array([t[key]['FP'] for t in tr]) for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']})
TP = pd.DataFrame({key: np.array([t[key]['TP'] for t in tr]) for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']})
#sizeme(FN)

matplotlib.rcParams['figure.figsize'] = (18.0, 6.0)
fig, axes = plt.subplots(nrows=1, ncols=2)

sns.violinplot(data=TP, inner="quart", cut=True, linewidth=0.3, bw=0.5, ax=axes[0])
axes[0].set_title('True positives')
sns.violinplot(data=FP, inner="quart", cut=True, linewidth=0.3, bw=0.5, ax=axes[1])
axes[1].set_title('False positives')

inputs = [(f, seed) for f in np.repeat(1500, 10) for seed in np.arange(66, 86, 1)]
print len(inputs)
testResults2 = Parallel(n_jobs=num_cores, verbose=2)(delayed(runTest)(i[0], i[1]) for i in inputs)

tr = testResults2
FN = {key: np.array([t[key]['FN'] for t in tr]).mean() for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']}
FP = {key: np.array([t[key]['FP'] for t in tr]).mean() for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']}
TP = {key: np.array([t[key]['TP'] for t in tr]).mean() for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']}
print 'FN:', FN
print 'FP:', FP
print 'TP:', TP

FN = pd.DataFrame({key: np.array([t[key]['FN'] for t in tr]) for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']})
FP = pd.DataFrame({key: np.array([t[key]['FP'] for t in tr]) for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']})
TP = pd.DataFrame({key: np.array([t[key]['TP'] for t in tr]) for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']})
#sizeme(FN)

matplotlib.rcParams['figure.figsize'] = (18.0, 6.0)
fig, axes = plt.subplots(nrows=1, ncols=2)

sns.violinplot(data=TP, inner="quart", cut=True, linewidth=0.3, bw=0.5, ax=axes[0])
axes[0].set_title('True positives')
sns.violinplot(data=FP, inner="quart", cut=True, linewidth=0.3, bw=0.5, ax=axes[1])
axes[1].set_title('False positives')

inputs = [(f, seed) for f in np.arange(1000, 2000, 25) for seed in np.arange(66, 86, 1)]
print len(inputs)
testResults3 = Parallel(n_jobs=num_cores, verbose=2)(delayed(runTest)(i[0], i[1]) for i in inputs)

tr = testResults3
FN = {key: np.array([t[key]['FN'] for t in tr]).mean() for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']}
FP = {key: np.array([t[key]['FP'] for t in tr]).mean() for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']}
TP = {key: np.array([t[key]['TP'] for t in tr]).mean() for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']}
print 'FN:', FN
print 'FP:', FP
print 'TP:', TP

FN = pd.DataFrame({key: np.array([t[key]['FN'] for t in tr]) for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']})
FP = pd.DataFrame({key: np.array([t[key]['FP'] for t in tr]) for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']})
TP = pd.DataFrame({key: np.array([t[key]['TP'] for t in tr]) for key in ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']})
#sizeme(FN)

matplotlib.rcParams['figure.figsize'] = (18.0, 6.0)
fig, axes = plt.subplots(nrows=1, ncols=2)

sns.violinplot(data=TP, inner="quart", cut=True, linewidth=0.3, bw=0.5, ax=axes[0])
axes[0].set_title('True positives')
sns.violinplot(data=FP, inner="quart", cut=True, linewidth=0.3, bw=0.5, ax=axes[1])
axes[1].set_title('False positives')



