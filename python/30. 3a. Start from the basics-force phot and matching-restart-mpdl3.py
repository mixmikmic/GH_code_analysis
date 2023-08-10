import numpy as np
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

pd.options.display.max_columns = 9999
pd.set_option('display.width', 9999)

import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import mpld3
import mpld3.plugins
print mpld3.__version__

mpld3.enable_notebook()

import diffimTests as dit

# Set up console so we can reattach via terminal ipython later. See:
# https://stackoverflow.com/questions/19479645/using-ipython-console-along-side-ipython-notebook

get_ipython().magic('qtconsole')

# Then do `ipython console --existing` in a terminal to connect and have access to same data!
# But note, do not do CTRL-D in that terminal or it will kill the kernel!

reload(dit)
testObj = dit.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=False, skyLimited=True)
res = testObj.runTest(returnSources=True, matchDist=np.sqrt(1.5))
src = res['sources']
del res['sources']
print res

cats = testObj.doForcedPhot(transientsOnly=True)
sources, fp1, fp2, fp_ZOGY, fp_AL, fp_ALd = cats
dit.sizeme(dit.catalogToDF(sources).head())

fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))

x = np.vstack((sources['inputFlux_science'], sources['inputFlux_science'],
               sources['inputFlux_science']))
data = np.vstack((fp_ZOGY['base_PsfFlux_flux']/fp_ZOGY['base_PsfFlux_fluxSigma'],
                  fp_AL['base_PsfFlux_flux']/fp_AL['base_PsfFlux_fluxSigma'],
                  fp_ALd['base_PsfFlux_flux']/fp_ALd['base_PsfFlux_fluxSigma']))

#ax.scatter(sources['inputFlux_science'], 
#           fp_AL['base_PsfFlux_flux']/fp_AL['base_PsfFlux_fluxSigma'], label='AL', color='r',
#           alpha=0.3, cmap=plt.cm.jet)
#ax.scatter(sources['inputFlux_science'], 
#           fp_ALd['base_PsfFlux_flux']/fp_ALd['base_PsfFlux_fluxSigma'], label='ALd', color='g',
#           alpha=0.3, cmap=plt.cm.jet)

ax.set_xlabel('input flux')
ax.set_ylabel('measured SNR')
ax.set_xlim(0, 2000)

labels = ['ZOGY', 'AL', 'ALd']
line_collections = ax.plot(x.T, data.T, 'o', ms=10, alpha=0.3)
interactive_legend = mpld3.plugins.InteractiveLegendPlugin(line_collections, labels)
mpld3.plugins.connect(fig, interactive_legend)

scatter = ax.scatter(x.T, data.T, alpha=0.01)
labs = ['id {0}'.format(i) for i in dit.catalogToDF(sources).id.values]
labs = labs + labs + labs
tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labs)
mpld3.plugins.connect(fig, tooltip)

mpld3.display()

cats = testObj.doForcedPhot(transientsOnly=False)
sources, fp1, fp2, fp_ZOGY, fp_AL, fp_ALd = cats
get_ipython().magic('matplotlib notebook')
plt.scatter(sources['inputFlux_science'], fp_ZOGY['base_PsfFlux_flux']/fp_ZOGY['base_PsfFlux_fluxSigma'], label='ZOGY')
plt.scatter(sources['inputFlux_science'], fp_AL['base_PsfFlux_flux']/fp_AL['base_PsfFlux_fluxSigma'], label='AL', color='r')
plt.scatter(sources['inputFlux_science'], fp_ALd['base_PsfFlux_flux']/fp_ALd['base_PsfFlux_fluxSigma'], label='ALd', color='g')
plt.legend(loc='upper left')
plt.xlabel('input flux')
plt.ylabel('measured SNR')
plt.xlim(0, 20000);

cats = testObj.doForcedPhot(transientsOnly=True)
sources, fp1, fp2, fp_ZOGY, fp_AL, fp_ALd = cats
dit.sizeme(dit.catalogToDF(sources).head())

import lsst.afw.table as afwTable
import lsst.afw.table.catalogMatches as catMatch
import lsst.daf.base as dafBase
reload(dit)

matches = afwTable.matchXy(sources, src['ZOGY'], 1.0)
print len(matches)

metadata = dafBase.PropertyList()
matchCat = catMatch.matchesToCatalog(matches, metadata)
tmp = dit.catalogToDF(matchCat)
dit.sizeme(tmp.head())

dit.sizeme(tmp[np.in1d(tmp['ref_id'], [1,2,3,4,5])])

def plotWithDetectionsHighlighted(fp_DIFFIM=fp_ZOGY, label='ZOGY', color='b', alpha=1.0,
                                 addPresub=False, xaxisIsScienceForcedPhot=False):
    if not xaxisIsScienceForcedPhot:
        srces = sources['inputFlux_science']
    else:
        srces = fp2['base_PsfFlux_flux']

    plt.scatter(srces, 
                fp_DIFFIM['base_PsfFlux_flux']/fp_DIFFIM['base_PsfFlux_fluxSigma'], 
                color=color, alpha=alpha, label=None, s=10)
    plt.scatter(srces, 
                fp_DIFFIM['base_PsfFlux_flux']/fp_DIFFIM['base_PsfFlux_fluxSigma'], 
                color='k', marker='x', label=None, s=10)

    # TBD: if xaxisIsScienceForcedPhot is True, then don't use sources['inputFlux_science'] --
    #    use fp2['base_PsfFlux_flux'] instead.

    if not xaxisIsScienceForcedPhot:
        matches = afwTable.matchXy(sources, src[label], 1.0)
        metadata = dafBase.PropertyList()
        matchCat = catMatch.matchesToCatalog(matches, metadata)
        sources_detected = dit.catalogToDF(sources)
        sources_detected = sources_detected[np.in1d(sources_detected['id'], matchCat['ref_id'])]
        sources_detected = sources_detected['inputFlux_science']
        fp_ZOGY_detected = dit.catalogToDF(fp_DIFFIM)
        fp_ZOGY_detected = fp_ZOGY_detected[np.in1d(fp_ZOGY_detected['id'], matchCat['ref_id'])]
    else:
        matches = afwTable.matchXy(fp2, src[label], 1.0)
        metadata = dafBase.PropertyList()
        matchCat = catMatch.matchesToCatalog(matches, metadata)
        sources_detected = dit.catalogToDF(fp2)
        sources_detected = sources_detected[np.in1d(sources_detected['id'], matchCat['ref_id'])]
        sources_detected = sources_detected['base_PsfFlux_flux']
        fp_ZOGY_detected = dit.catalogToDF(fp_DIFFIM)
        fp_ZOGY_detected = fp_ZOGY_detected[np.in1d(fp_ZOGY_detected['id'], matchCat['ref_id'])]

    plt.scatter(sources_detected, 
                fp_ZOGY_detected['base_PsfFlux_flux']/fp_ZOGY_detected['base_PsfFlux_fluxSigma'], 
                label=label, s=20, color=color, alpha=alpha, edgecolors='r')
    
    if addPresub: # Add measurements in original science and template images
        srces = sources['inputFlux_science']
        if xaxisIsScienceForcedPhot:
            srces = fp2['base_PsfFlux_flux']
        plt.scatter(srces, 
                    fp1['base_PsfFlux_flux']/fp1['base_PsfFlux_fluxSigma'], 
                    label='template', color='y')
        plt.scatter(srces, 
                    fp2['base_PsfFlux_flux']/fp2['base_PsfFlux_fluxSigma'], 
                    label='science', color='orange', alpha=0.2)

get_ipython().magic('matplotlib notebook')
plotWithDetectionsHighlighted(fp_ZOGY)
plotWithDetectionsHighlighted(fp_AL, label='ALstack', color='r', alpha=0.2)
plotWithDetectionsHighlighted(fp_ALd, label='ALstack_decorr', color='g', addPresub=True)
plt.scatter([10000], [10], color='k', marker='x', label='Missed')
plt.legend(loc='upper left', scatterpoints=3)
plt.xlabel('input flux')
plt.ylabel('measured SNR')
plt.xlim(0, 2010)
plt.ylim(-2, 20);

get_ipython().magic('matplotlib notebook')
plotWithDetectionsHighlighted(fp_ZOGY, xaxisIsScienceForcedPhot=True)
plotWithDetectionsHighlighted(fp_AL, label='ALstack', color='r', xaxisIsScienceForcedPhot=True)
plotWithDetectionsHighlighted(fp_ALd, label='ALstack_decorr', color='g', 
                              xaxisIsScienceForcedPhot=True, addPresub=True)
plt.scatter([10000], [10], color='k', marker='x', label='Missed')
plt.legend(loc='upper left', scatterpoints=3)
plt.xlabel('science flux (measured)')
plt.ylabel('measured SNR')
plt.xlim(0, 2010)
plt.ylim(-2, 20);

reload(dit)
testObj = dit.DiffimTest(n_sources=100, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.repeat(750., 50),
                         templateNoNoise=True, skyLimited=True)
res = testObj.runTest(returnSources=True, matchDist=2.) #np.sqrt(1.5))
src = res['sources']
del res['sources']
print res

cats = testObj.doForcedPhot(transientsOnly=True)
sources, fp1, fp2, fp_ZOGY, fp_AL, fp_ALd = cats

get_ipython().magic('matplotlib notebook')
plotWithDetectionsHighlighted(fp_ZOGY, xaxisIsScienceForcedPhot=True)
plotWithDetectionsHighlighted(fp_AL, label='ALstack', color='r', alpha=0.2, xaxisIsScienceForcedPhot=True)
plotWithDetectionsHighlighted(fp_ALd, label='ALstack_decorr', color='g', 
                              xaxisIsScienceForcedPhot=True, addPresub=True)
plt.scatter([1000], [10], color='k', marker='x', label='Missed')
legend = plt.legend(loc='lower right', scatterpoints=3, frameon=1)
#frame = legend.get_frame()
#frame.set_facecolor('lightgrey')
plt.xlabel('science flux (measured)')
plt.ylabel('measured SNR')
plt.xlim(400, 1000)
plt.ylim(-0.2, 8);

print testObj.im1.sig, testObj.im2.sig
print dit.computeClippedImageStats(testObj.im1.var)
print dit.computeClippedImageStats(testObj.im1.im)



