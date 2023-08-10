import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
import diffimTests as dit

import warnings
warnings.filterwarnings('ignore')

n_runs = 10
ns = np.append(np.insert(np.arange(500, 5001, 250), 0, [50, 100, 250]), [7500, 10000, 15000])
testResults1 = dit.multi.runMultiDiffimTests(varSourceFlux=620., 
                                             n_varSources=50, nStaticSources=ns,
                                             #templateNoNoise=True, skyLimited=True,
                                             sky=[30., 300.],
                                             avoidAllOverlaps=0.,
                                             n_runs=n_runs, remeasurePsfs=[False, False])

import pandas as pd
methods = ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_decorr']

tr1 = [tr for tr in testResults1 if tr['result'] is not None]
print len(testResults1), len(tr1)

TP1 = []; FP1 = []; FN1 = []
for i, tr in enumerate(tr1):    
    FN = {key: tr['result'][key]['FN'] for key in methods}
    FN['n_sources'] = tr['n_sources']
    FN1.append(pd.DataFrame(FN, index=[0]))

    TP = {key: tr['result'][key]['TP'] for key in methods}
    TP['n_sources'] = tr['n_sources']
    TP1.append(pd.DataFrame(TP, index=[0]))

    FP = {key: tr['result'][key]['FP'] for key in methods}
    FP['n_sources'] = tr['n_sources']
    FP1.append(pd.DataFrame(FP, index=[0]))

FN1mn = pd.concat(FN1, axis=0).groupby('n_sources').median()
TP1mn = pd.concat(TP1, axis=0).groupby('n_sources').median()
FP1mn = pd.concat(FP1, axis=0).groupby('n_sources').median()

FN1err = pd.concat(FN1, axis=0).groupby('n_sources').std()
TP1err = pd.concat(TP1, axis=0).groupby('n_sources').std()
FP1err = pd.concat(FP1, axis=0).groupby('n_sources').std()
dit.sizeme(FN1mn.head())

plt.subplots(1, 2, figsize=(20, 8))
ax = plt.subplot(121)
#TP1mn.drop('SZOGY', 1).reset_index().plot(x='n_sources', yerr=TP1err, alpha=0.5, lw=5, ax=ax)
TP1mn.reset_index().plot(x='n_sources', yerr=TP1err, alpha=0.5, lw=5, ax=ax)
ax.set_ylabel('True positives (out of 50)')
ax.set_xlim(0, 6000)
ax.set_ylim(0, 25)
ax = plt.subplot(122)    
#FP1mn.drop('SZOGY', 1).reset_index().plot(x='n_sources', yerr=FP1err, alpha=0.5, lw=5, ax=ax)
FP1mn.reset_index().plot(x='n_sources', yerr=FP1err, alpha=0.5, lw=5, ax=ax)
ax.set_ylabel('False positives')
ax.set_title('Low-noise template, sky-limited')
ax.set_xlim(0, 6000)
ax.set_ylim(0, 25);

n_runs = 10
ns = np.append(np.insert(np.arange(500, 5001, 250), 0, [50, 100, 250]), [7500, 10000, 15000])
testResults2 = dit.multi.runMultiDiffimTests(varSourceFlux=620., 
                                             n_varSources=50, nStaticSources=ns,
                                             #templateNoNoise=True, skyLimited=True,
                                             sky=[30., 300.],
                                             avoidAllOverlaps=0.,
                                             n_runs=n_runs, remeasurePsfs=[True, True])

dit.dumpObjects((testResults1, testResults2), "tmp5_pkl")

testResults1, testResults2 = dit.loadObjects('tmp5_pkl')

import pandas as pd
methods = ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_decorr']

tr2 = [tr for tr in testResults2 if tr['result'] is not None]
print len(testResults2), len(tr2)

TP1 = []; FP1 = []; FN1 = []
for i, tr in enumerate(tr2):    
    rms1 = tr['psfInfo']['normedRms1']
    if rms1 is None:
        rms1 = np.nan
    rms2 = tr['psfInfo']['normedRms2']
    if rms2 is None:
        rms2 = np.nan

    TP = {key: tr['result'][key]['TP'] for key in methods}
    TP['n_sources'] = tr['n_sources']
    TP['normedRms1'] = rms1
    TP['normedRms2'] = rms2
    TP1.append(pd.DataFrame(TP, index=[0]))

    FN = {key: tr['result'][key]['FN'] for key in methods}
    FN['n_sources'] = tr['n_sources']
    FN1.append(pd.DataFrame(FN, index=[0]))

    FP = {key: tr['result'][key]['FP'] for key in methods}
    FP['n_sources'] = tr['n_sources']
    FP1.append(pd.DataFrame(FP, index=[0]))

FN1mn = pd.concat(FN1, axis=0).groupby('n_sources').median()
TP1mn = pd.concat(TP1, axis=0).groupby('n_sources').median()
FP1mn = pd.concat(FP1, axis=0).groupby('n_sources').median()

FN1err = pd.concat(FN1, axis=0).groupby('n_sources').std()
TP1err = pd.concat(TP1, axis=0).groupby('n_sources').std()
FP1err = pd.concat(FP1, axis=0).groupby('n_sources').std()
dit.sizeme(FN1mn.head())

plt.subplots(3, 2, figsize=(20, 16))
ax = plt.subplot(221)    
TP1mn.drop(['normedRms1','normedRms2'], 1).reset_index().plot(x='n_sources', 
                                               yerr=TP1err.drop(['normedRms1','normedRms2'], 1), 
                                               alpha=0.5, lw=5, ax=ax)
ax.set_ylabel('True positives (out of 50)')
ax.set_xlim(0, 6000)
ax.set_ylim(0, 25)
ax = plt.subplot(222)    
FP1mn.reset_index().plot(x='n_sources', yerr=FP1err, alpha=0.5, lw=5, ax=ax)
ax.set_ylabel('False positives')
ax.set_title('Low-noise template, sky-limited')
ax.set_xlim(0, 6000)
ax.set_ylim(0, 25)

ax = plt.subplot(223)    
TP1mn[['normedRms1', 'normedRms2']].reset_index().plot(x='n_sources', 
                                                       yerr=TP1err[['normedRms1', 'normedRms2']],
                                                       alpha=0.5, lw=5, ax=ax)
ax.set_xlim(0, 6000)
ax.set_ylabel('PSF measurement error (RMS)')

TP1a = pd.concat(TP1, axis=0).drop(['normedRms2'], 1)
TP1a = TP1a[TP1a.n_sources <= 6000]
TP1a.boxplot(column=methods, by='n_sources');

FP1a = pd.concat(FP1, axis=0).drop(['SZOGY'], 1)
FP1a = FP1a[FP1a.n_sources <= 6000]
FP1a.boxplot(column=['ALstack', 'ZOGY', 'ALstack_decorr'], by='n_sources');

TP1a.head()

import seaborn as sns

TP1a = pd.concat(TP1, axis=0)
ax = plt.subplot(321)
g = sns.violinplot(x='n_sources', y='normedRms1', 
                   data=TP1a[['n_sources', 'normedRms1', 'normedRms2']], 
                   inner="box", cut=0, linewidth=0.3, bw=0.5, ax=ax)
g.set_xticklabels(g.get_xticklabels(), rotation=60);
ax = plt.subplot(322)
g = sns.violinplot(x='n_sources', y='normedRms2', 
                   data=TP1a[['n_sources', 'normedRms1', 'normedRms2']], 
                   inner="box", cut=0, linewidth=0.3, bw=0.5, ax=ax)
g.set_xticklabels(g.get_xticklabels(), rotation=60);

ax = plt.subplot(323)
g = sns.violinplot(x='n_sources', y='ALstack_decorr', 
                   data=TP1a[['n_sources', 'ALstack_decorr', 'ZOGY']], 
                   inner="box", cut=0, linewidth=0.3, bw=0.5, ax=ax)
g.set_xticklabels(g.get_xticklabels(), rotation=60);
ax = plt.subplot(324)
g = sns.violinplot(x='n_sources', y='ZOGY', 
                   data=TP1a[['n_sources', 'ALstack_decorr', 'ZOGY']], 
                   inner="box", cut=0, linewidth=0.3, bw=0.5, ax=ax)
g.set_xticklabels(g.get_xticklabels(), rotation=60);

FP1a = pd.concat(FP1, axis=0)
ax = plt.subplot(325)
g = sns.violinplot(x='n_sources', y='ALstack_decorr', 
                   data=FP1a[['n_sources', 'ALstack_decorr', 'ZOGY']], 
                   inner="box", cut=0, linewidth=0.3, bw=0.5, ax=ax)
g.set_xticklabels(g.get_xticklabels(), rotation=60);
ax.set_ylim(0, 40)
ax = plt.subplot(326)
g = sns.violinplot(x='n_sources', y='ZOGY', 
                   data=FP1a[['n_sources', 'ALstack_decorr', 'ZOGY']], 
                   inner="box", cut=0, linewidth=0.3, bw=0.5, ax=ax)
g.set_xticklabels(g.get_xticklabels(), rotation=60);
ax.set_ylim(0, 40)



