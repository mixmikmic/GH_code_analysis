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
print sns.__version__
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
testObj = dit.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=False, skyLimited=True)
res = testObj.runTest(returnSources=True, matchDist=np.sqrt(1.5))
src = res['sources']
del res['sources']
print res

reload(dit)
cats = testObj.doForcedPhot(transientsOnly=True)

sources, mc1, mc2, mc_ZOGY, mc_AL, mc_ALd = cats
plt.scatter(sources['inputFlux_science'], mc_ZOGY['base_PsfFlux_flux']/mc_ZOGY['base_PsfFlux_fluxSigma'], label='ZOGY')
plt.scatter(sources['inputFlux_science'], mc_AL['base_PsfFlux_flux']/mc_AL['base_PsfFlux_fluxSigma'], label='AL', color='r')
plt.scatter(sources['inputFlux_science'], mc_ALd['base_PsfFlux_flux']/mc_ALd['base_PsfFlux_fluxSigma'], label='ALd', color='g')
plt.legend(loc='upper left')
plt.xlabel('input flux')
plt.ylabel('measured SNR')
plt.xlim(0, 2000);

cats = testObj.doForcedPhot(transientsOnly=False)
sources, mc1, mc2, mc_ZOGY, mc_AL, mc_ALd = cats
plt.scatter(sources['inputFlux_science'], mc_ZOGY['base_PsfFlux_flux']/mc_ZOGY['base_PsfFlux_fluxSigma'], label='ZOGY')
plt.scatter(sources['inputFlux_science'], mc_AL['base_PsfFlux_flux']/mc_AL['base_PsfFlux_fluxSigma'], label='AL', color='r')
plt.scatter(sources['inputFlux_science'], mc_ALd['base_PsfFlux_flux']/mc_ALd['base_PsfFlux_fluxSigma'], label='ALd', color='g')
plt.legend(loc='upper left')
plt.xlabel('input flux')
plt.ylabel('measured SNR')
plt.xlim(0, 20000);

cats = testObj.doForcedPhot(transientsOnly=False)
sources, mc1, mc2, mc_ZOGY, mc_AL, mc_ALd = cats
print len(sources), len(mc_ZOGY)

import lsst.afw.table as afwTable
import lsst.afw.table.catalogMatches as catMatch
import lsst.daf.base as dafBase
reload(dit)

matches = afwTable.matchXy(src['ZOGY'], sources, 1.0)
print len(matches)

metadata = dafBase.PropertyList()
matchCat = catMatch.matchesToCatalog(matches, metadata)
dit.sizeme(dit.catalogToDF(matchCat).head())

reload(dit)
testObj = dit.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=False, skyLimited=True)
res = testObj.runTest(returnSources=True, matchDist=np.sqrt(1.5))
src = res['sources']
del res['sources']
print res

sources, fp1, fp2, fp_ZOGY, fp_AL, fp_ALd = testObj.doForcedPhot(transientsOnly=True)

sourcesA, fp1A, fp2A, fp_ZOGYA, fp_ALA, fp_ALdA = testObj.doForcedPhot(centroids=src['ZOGY'])

print len(sources)
dit.sizeme(dit.catalogToDF(sources).head())

print len(fp_ZOGY)
dit.sizeme(dit.catalogToDF(fp_ZOGY).head())

print len(fp_ZOGYA)
dit.sizeme(dit.catalogToDF(fp_ZOGYA).head())

matches = afwTable.matchXy(sources, src['ZOGY'], 1.0)
metadata = dafBase.PropertyList()
matchedCat = catMatch.matchesToCatalog(matches, metadata)

tmp = dit.catalogToDF(matchedCat)
print tmp.shape
dit.sizeme(tmp.ix[np.in1d(tmp['ref_id'], [1,2,3,4,5])])
#dit.sizeme(tmp.head())

matchesA = afwTable.matchXy(sources, fp_ZOGYA, 1.0)
metadata = dafBase.PropertyList()
matchedCatA = catMatch.matchesToCatalog(matchesA, metadata)

tmp = dit.catalogToDF(matchedCatA)
print tmp.shape
dit.sizeme(tmp.ix[np.in1d(tmp['ref_id'], [1,2,3,4,5])])
#dit.sizeme(tmp.head())

plt.plot(matchedCat['src_base_PsfFlux_flux'], matchedCatA['src_base_PsfFlux_flux'])

reload(dit)
import lsst.afw.table.catalogMatches as catMatch
import lsst.daf.base as dafBase

# First, forced-photometry centered on all 50 positions of input transients

# Convert the input centroids into a catalog...
inputCatalog = dit.centroidsToCatalog(testObj.centroids, testObj.im1.asAfwExposure().getWcs(),
                                     transientsOnly=True)
# Run the forced phot. fp_ZOGY is forced-phot of 50 input transient locations in ZOGY diffim.
cats = testObj.doForcedPhot(inputCatalog)
sources, fp1, fp2, fp_ZOGY, fp_AL, fp_ALd = cats
print len(fp_ZOGY)

# Cross-ref the 50 input transient locations (and their forced phot measurements) 
#    with tbe 27 ZOGY detections
catalog = src['ZOGY'].copy(deep=True)
catalog = catalog[~catalog['base_PsfFlux_flag']]  # this works!
catalog = catalog.copy(deep=True)
zogyDetections = catalog
print len(zogyDetections)
matches = afwTable.matchXy(zogyDetections, inputCatalog, 1.0)
metadata = dafBase.PropertyList()
matchCat = catMatch.matchesToCatalog(matches, metadata)
print len(matchCat)  # matchCat has matches betw. input transients and detected transients -> 26.

dit.sizeme(dit.catalogToDF(matchCat).head())

# Take the 31 ZOGY detections and force-phot them on the images (fp1, fp2, fp_ZOGY).
cats2 = testObj.doForcedPhot(zogyDetections)
#_, fp1, fp2, fp_ZOGY, _, _ = cats2
print len(cats2[2]), len(cats2[3])

# ... now we need to find the 5 records in cats2 that are not among the 50 input transients.
# We'll do it by (1) matching cats2[2] with inputCatalog, and then finding the rows of cats[2]
# that are *not* in the resulting set of matches.
matches2 = afwTable.matchXy(inputCatalog, cats2[2], 1.0)   # cats2[3] is fp_ZOGY centered at zogyDetections
metadata = dafBase.PropertyList()
matchCat2 = catMatch.matchesToCatalog(matches2, metadata)
print len(matchCat2)

hits_x = np.in1d(cats2[2]['base_TransformedCentroid_x'], matchCat2['src_base_TransformedCentroid_x'])
hits_y = np.in1d(cats2[2]['base_TransformedCentroid_y'], matchCat2['src_base_TransformedCentroid_y'])
tmp_cat = cats2[2][(~hits_x) & (~hits_y)]
tmp_cat = tmp_cat.copy(deep=True)
print len(tmp_cat)

dit.sizeme(dit.catalogToDF(tmp_cat))

reload(dit)
df = dit.catalogToDF(tmp_cat)
#tmp = dit.dfToCatalog(df)
#dt2 = dit.catalogToDF(tmp)
#dit.sizeme(dt2.head())



