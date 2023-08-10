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
n_sources = 100
n_varSources = 50
seed = 66

reload(dit)

testObj = dit.DiffimTest(imSize=(512,512), sky=sky, psf1=[1.6,1.6], psf2=[1.8,2.2],
                         offset=[0,0], psf_yvary_factor=0., sourceFluxDistrib='uniform',
                         varFlux2=np.repeat(500., n_varSources), variablesNearCenter=False,
                         theta1=0., theta2=-45., im2background=0., n_sources=n_sources, 
                         sourceFluxRange=(500,30000), avoidBorder=False, 
                         seed=seed, psfSize=21)

res = testObj.runTest()
print res

reload(dit)
fig = plt.figure(1, (8, 8))
testObj.doPlot()

reload(dit)
src1 = testObj.im1.doDetection(threshold=5.0)
src2 = testObj.im2.doDetection(threshold=5.0)

src1 = src1[~src1['base_PsfFlux_flag']]
src2 = src2[~src2['base_PsfFlux_flag']]

print src1.shape
print src2.shape

src = src1
dist = np.sqrt(np.add.outer(src.base_NaiveCentroid_x, -testObj.centroids[:, 0])**2. +    np.add.outer(src.base_NaiveCentroid_y, -testObj.centroids[:, 1])**2.) # in pixels
matches = np.where(dist <= 1.5)
true_pos = len(np.unique(matches[0]))
false_neg = testObj.centroids.shape[0] - len(np.unique(matches[1]))
false_pos = src.shape[0] - len(np.unique(matches[0]))
detections = {'TP': true_pos, 'FN': false_neg, 'FP': false_pos}
print detections

src_hits1 = src.iloc[matches[0],:]
input_hits1 = testObj.centroids[matches[1],:]

sizeme(src_hits1.head())

plt.scatter(input_hits1[:,2], src_hits1.base_PsfFlux_flux.values)
plt.scatter(input_hits1[:,2], src_hits1.base_PeakLikelihoodFlux_flux.values, color='r')
plt.xlim(0, 32000)
plt.ylim(0, 65000);

src = src2
dist = np.sqrt(np.add.outer(src.base_NaiveCentroid_x, -testObj.centroids[:, 0])**2. +    np.add.outer(src.base_NaiveCentroid_y, -testObj.centroids[:, 1])**2.) # in pixels
matches = np.where(dist <= 1.5)
true_pos = len(np.unique(matches[0]))
false_neg = testObj.centroids.shape[0] - len(np.unique(matches[1]))
false_pos = src.shape[0] - len(np.unique(matches[0]))
detections = {'TP': true_pos, 'FN': false_neg, 'FP': false_pos}
print detections

src_hits2 = src.iloc[matches[0],:]
input_hits2 = testObj.centroids[matches[1],:]

plt.scatter(input_hits2[:,3], src_hits2.base_PsfFlux_flux.values)
plt.scatter(input_hits2[:,3], src_hits2.base_PeakLikelihoodFlux_flux.values, color='r')
plt.xlim(0, 32000)
plt.ylim(0, 65000);

print input_hits1[:,2].min(), (src_hits1.base_PsfFlux_flux.values/src_hits1.base_PsfFlux_fluxSigma.values).min()
print input_hits2[:,3].min(), (src_hits2.base_PsfFlux_flux.values/src_hits2.base_PsfFlux_fluxSigma.values).min()
plt.scatter(input_hits1[:,3], src_hits1.base_PsfFlux_flux.values/src_hits1.base_PsfFlux_fluxSigma.values)
plt.scatter(input_hits2[:,3], src_hits2.base_PsfFlux_flux.values/src_hits2.base_PsfFlux_fluxSigma.values)
#plt.scatter(input_hits1[:,2], src_hits.base_PeakLikelihoodFlux_flux.values/src_hits.base_PeakLikelihoodFlux_fluxSigma.values, color='r')
plt.xlim(0, 32000);
#plt.ylim(0, 65000);



