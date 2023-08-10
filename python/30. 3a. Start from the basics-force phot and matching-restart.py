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
testObj = dit.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=True, skyLimited=True,
                        avoidAllOverlaps=15.)
res = testObj.runTest(returnSources=True, matchDist=np.sqrt(1.5))
src = res['sources']
del res['sources']
print res

cats = testObj.doForcedPhot(transientsOnly=True)
sources, fp1, fp2, fp_ZOGY, fp_AL, fp_ALd = cats

#%matplotlib notebook
plt.scatter(sources['inputFlux_science']+10, fp_ZOGY['base_PsfFlux_flux']/fp_ZOGY['base_PsfFlux_fluxSigma'], label='ZOGY')
plt.scatter(sources['inputFlux_science'], fp_AL['base_PsfFlux_flux']/fp_AL['base_PsfFlux_fluxSigma'], label='AL', color='r')
plt.scatter(sources['inputFlux_science'], fp_ALd['base_PsfFlux_flux']/fp_ALd['base_PsfFlux_fluxSigma'], label='ALd', color='g')
plt.legend(loc='upper left')
plt.xlabel('input flux')
plt.ylabel('measured SNR')
plt.xlim(0, 2000);

cats = testObj.doForcedPhot(transientsOnly=False)
sources, fp1, fp2, fp_ZOGY, fp_AL, fp_ALd = cats
#%matplotlib notebook
plt.scatter(sources['inputFlux_science']+10, fp_ZOGY['base_PsfFlux_flux']/fp_ZOGY['base_PsfFlux_fluxSigma'], label='ZOGY')
plt.scatter(sources['inputFlux_science'], fp_AL['base_PsfFlux_flux']/fp_AL['base_PsfFlux_fluxSigma'], label='AL', color='r', alpha=0.4)
plt.scatter(sources['inputFlux_science'], fp_ALd['base_PsfFlux_flux']/fp_ALd['base_PsfFlux_fluxSigma'], label='ALd', color='g')
plt.legend(loc='upper left')
plt.xlabel('input flux')
plt.ylabel('measured SNR')
plt.xlim(0, 20000);

flux = 750.
psf = testObj.im2.psf
sky = testObj.im2.sig**2
print sky

nPix = np.sum(psf/psf.max()) * 2.  # not sure where the 2 comes from but it works.
print nPix, np.pi*1.8*2.2*4  # and it equals pi*r1*r2*4.

def snr(flux, sky, psf, skyLimited=True):
    psf = psf / psf.max()
    nPix = np.sum(psf) * 2.
    if skyLimited:  #  only sky noise matters 
        return flux / (np.sqrt(nPix * sky))
    else:
        return flux / (np.sqrt(flux + nPix * sky))

print snr(flux, sky, psf)
print testObj.im2.calcSNR(flux, skyLimited=True)  # moved the func here.

meas = fp2['base_PsfFlux_flux']/fp2['base_PsfFlux_fluxSigma']
calc = testObj.im2.calcSNR(sources['inputFlux_science'], skyLimited=True)
print np.median(meas/calc)
plt.scatter(sources['inputFlux_science'], meas/calc)
plt.xlim(0, 20000)
plt.ylim(0.8, 1.2);

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

def plotWithDetectionsHighlighted(testObj, transientsOnly=True, addPresub=False, 
                                  xaxisIsScienceForcedPhot=False, alpha=0.5):
    
    #fp_DIFFIM=fp_ZOGY, label='ZOGY', color='b', alpha=1.0,
    
    res = testObj.runTest(zogyImageSpace=True, returnSources=True, matchDist=np.sqrt(1.5))
    src = res['sources']
    del res['sources']
    print res
    
    cats = testObj.doForcedPhot(transientsOnly=transientsOnly)
    sources, fp1, fp2, fp_ZOGY, fp_AL, fp_ALd = cats

    # if xaxisIsScienceForcedPhot is True, then don't use sources['inputFlux_science'] --
    #    use fp2['base_PsfFlux_flux'] instead.
    if not xaxisIsScienceForcedPhot:
        srces = sources['inputFlux_science']
    else:
        srces = fp2['base_PsfFlux_flux']
        
    df = pd.DataFrame()
    df['inputFlux'] = sources['inputFlux_science']
    df['templateFlux'] = fp1['base_PsfFlux_flux']
    df['scienceFlux'] = fp2['base_PsfFlux_flux']
    df['inputId'] = sources['id']
    df['inputCentroid_x'] = sources['centroid_x']
    df['inputCentroid_y'] = sources['centroid_y']
    
    fp_DIFFIM = [fp_ZOGY, fp_AL, fp_ALd]
    label = ['ZOGY', 'ALstack', 'ALstack_decorr']
    color = ['b', 'r', 'g']
    
    for i, fp_d in enumerate(fp_DIFFIM):
        df[label[i] + '_SNR'] = fp_d['base_PsfFlux_flux']/fp_d['base_PsfFlux_fluxSigma']
        df[label[i] + '_flux'] = fp_d['base_PsfFlux_flux']
        df[label[i] + '_fluxSigma'] = fp_d['base_PsfFlux_fluxSigma']

        plt.scatter(srces, 
                    fp_d['base_PsfFlux_flux']/fp_d['base_PsfFlux_fluxSigma'], 
                    color=color[i], alpha=alpha, label=None, s=10)
        plt.scatter(srces, 
                    fp_d['base_PsfFlux_flux']/fp_d['base_PsfFlux_fluxSigma'], 
                    color='k', marker='x', alpha=alpha, label=None, s=10)

        if not xaxisIsScienceForcedPhot:
            matches = afwTable.matchXy(sources, src[label[i]], 1.0)
            metadata = dafBase.PropertyList()
            matchCat = catMatch.matchesToCatalog(matches, metadata)
            sources_detected = dit.catalogToDF(sources)
            detected = np.in1d(sources_detected['id'], matchCat['ref_id'])
            sources_detected = sources_detected[detected]
            sources_detected = sources_detected['inputFlux_science']
            fp_ZOGY_detected = dit.catalogToDF(fp_d)
            detected = np.in1d(fp_ZOGY_detected['id'], matchCat['ref_id'])
            fp_ZOGY_detected = fp_ZOGY_detected[detected]
        else:
            matches = afwTable.matchXy(fp2, src[label[i]], 1.0)
            metadata = dafBase.PropertyList()
            matchCat = catMatch.matchesToCatalog(matches, metadata)
            sources_detected = dit.catalogToDF(fp2)
            detected = np.in1d(sources_detected['id'], matchCat['ref_id'])
            sources_detected = sources_detected[detected]
            sources_detected = sources_detected['base_PsfFlux_flux']
            fp_ZOGY_detected = dit.catalogToDF(fp_d)
            detected = np.in1d(fp_ZOGY_detected['id'], matchCat['ref_id'])
            fp_ZOGY_detected = fp_ZOGY_detected[detected]

        df[label[i] + '_detected'] = detected
        plt.scatter(sources_detected, 
                    fp_ZOGY_detected['base_PsfFlux_flux']/fp_ZOGY_detected['base_PsfFlux_fluxSigma'], 
                    label=label[i], s=20, color=color[i], alpha=alpha) #, edgecolors='r')
    
    if addPresub: # Add measurements in original science and template images
        df['templateSNR'] = fp1['base_PsfFlux_flux']/fp1['base_PsfFlux_fluxSigma']
        plt.scatter(srces, 
                    fp1['base_PsfFlux_flux']/fp1['base_PsfFlux_fluxSigma'], 
                    label='template', color='y', alpha=alpha)
        df['scienceSNR'] = fp2['base_PsfFlux_flux']/fp2['base_PsfFlux_fluxSigma']
        plt.scatter(srces, 
                    fp2['base_PsfFlux_flux']/fp2['base_PsfFlux_fluxSigma'], 
                    label='science', color='orange', alpha=alpha-0.2)
        
    snrCalced = testObj.im2.calcSNR(sources['inputFlux_science'], skyLimited=True)
    df['inputSNR'] = snrCalced
    plt.scatter(srces, snrCalced, color='k', alpha=alpha-0.2, s=7, label='Input SNR')
    plt.scatter([10000], [10], color='k', marker='x', label='Missed')
    plt.legend(loc='upper left', scatterpoints=3)
    if not xaxisIsScienceForcedPhot:
        plt.xlabel('input flux')
    else:
        plt.xlabel('science flux (measured)')
    plt.ylabel('measured SNR')

    return df

## Just re-create the testObj here:
reload(dit)
testObj = dit.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=5.)

testObj.doPlotWithDetectionsHighlighted(transientsOnly=False, addPresub=True)
plt.xlim(0, 20010)
plt.ylim(-2, 205);

#%matplotlib notebook
df = testObj.doPlotWithDetectionsHighlighted(transientsOnly=True, addPresub=True)
plt.xlim(0, 2010)
plt.ylim(-0.2, 20);

tmp = df.ix[(df.scienceSNR > 12) & (df.scienceSNR < 15) & (df.inputFlux < 1500)]
dit.sizeme(tmp)

testObj.doPlot(centroidCoord=[tmp.inputCentroid_y.values[0], tmp.inputCentroid_x.values[0]]);

## Just re-create the testObj here:
reload(dit)
testObj = dit.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=15.)

testObj.doPlotWithDetectionsHighlighted(transientsOnly=False, addPresub=True)
plt.xlim(0, 20010)
plt.ylim(-2, 205);

#%matplotlib notebook
df = testObj.doPlotWithDetectionsHighlighted(transientsOnly=True, addPresub=True)
plt.xlim(0, 2010)
plt.ylim(-2, 20);

dit.sizeme(df.head())

plt.scatter(df.ALstack_SNR, df.ZOGY_SNR / df.ALstack_SNR)
plt.ylim(0.90, 1.02)

#%matplotlib notebook
testObj.doPlotWithDetectionsHighlighted(transientsOnly=True, addPresub=True,
                                        xaxisIsScienceForcedPhot=True)
plt.xlim(0, 2010)
plt.ylim(-2, 20);

reload(dit)
testObj2 = dit.DiffimTest(n_sources=91, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.repeat(610., 50),
                         templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=15.)

testObj2.doPlotWithDetectionsHighlighted(transientsOnly=True, addPresub=True,
                                         xaxisIsScienceForcedPhot=True)
plt.xlim(400, 900)
plt.ylim(-0.2, 8);

print testObj2.im1.sig, testObj2.im2.sig
print dit.computeClippedImageStats(testObj2.im1.var)
print dit.computeClippedImageStats(testObj2.im1.im)
print dit.computeClippedImageStats(testObj2.D_ZOGY.im)

reload(dit)
testObj3 = dit.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         templateNoNoise=False, skyLimited=False,
                         avoidAllOverlaps=15.)

testObj3.doPlotWithDetectionsHighlighted(transientsOnly=True, addPresub=True,
                                         xaxisIsScienceForcedPhot=True)
plt.xlim(0, 2000)
plt.ylim(-2, 18);

## Just re-create the testObj here:
reload(dit)
testObj = dit.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(1000., 50),
                         templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=15.)

df = testObj.doPlotWithDetectionsHighlighted(transientsOnly=True, 
                                             #xaxisIsScienceForcedPhot=True,
                                             addPresub=True)
plt.xlim(200, 2000)
plt.ylim(-0.2, 18);

dit.sizeme(df.head())

tmp = df.ix[(df.ZOGY_detected == False) & (df.ALstack_detected == True)]
dit.sizeme(tmp)

tmp = df.ix[(df.ZOGY_detected == True) & (df.ALstack_detected == False)]
dit.sizeme(tmp)

# Since detections are identical, let's just look at a transient missed by both methods: 
tmp = df.ix[(df.ZOGY_detected == False) & (df.ALstack_detected == False) &
           (df.templateFlux < 1.)]
dit.sizeme(tmp)

imagesToPlot, _ = testObj.doPlot(centroidCoord=[tmp.inputCentroid_y.values[0], 
                                  tmp.inputCentroid_x.values[0]]);

for img in imagesToPlot:
    print dit.computeClippedImageStats(img)
    
p1 = testObj.ALres.subtractedExposure.getPsf().computeImage().getArray()
p2 = testObj.ALres.decorrelatedDiffim.getPsf().computeImage().getArray()
p2a = dit.fixEvenKernel(p2)
print np.unravel_index(np.argmax(p1), p1.shape), p1.shape
print np.unravel_index(np.argmax(p2), p2.shape), p2.shape
print np.unravel_index(np.argmax(p2a), p2a.shape), p2a.shape

dit.plotImageGrid((p1, p2, p2a), clim=(-0.01, 0.01)) #,
                  #testObj.ALres.decorrelationKernel))

print (df.ALstack_decorr_SNR / df.ALstack_SNR).median()
plt.scatter(df.inputFlux, df.ALstack_decorr_flux / df.ALstack_flux, c='r')
plt.scatter(df.inputFlux, df.ALstack_decorr_fluxSigma / df.ALstack_fluxSigma, c='g')
plt.scatter(df.inputFlux, df.ALstack_decorr_SNR / df.ALstack_SNR, c='b')
plt.ylim(0.995, 1.005);

plt.scatter(df.inputFlux, df.ALstack_decorr_flux / df.inputFlux, c='r')
plt.scatter(df.inputFlux+10, df.ALstack_flux / df.inputFlux, c='g')
plt.scatter(df.inputFlux+20, df.ZOGY_flux / df.inputFlux * np.sqrt(300.), c='b')
plt.ylim(-0.05, 2.05)

img = testObj.ALres.decorrelatedDiffim.getMaskedImage().getImage().getArray()
img2 = testObj.ALres.subtractedExposure.getMaskedImage().getImage().getArray()
#dit.plotImageGrid((img, img2), imScale=8)

print dit.computeClippedImageStats(img)
print dit.computeClippedImageStats(img2)

tmp = img.flatten()
print len(tmp[~np.isnan(tmp)])
plt.hist(tmp[~np.isnan(tmp)], bins=200);
tmp2 = img2.flatten()
print len(tmp2[(~np.isnan(tmp2))])
plt.hist(tmp2[(~np.isnan(tmp2))], bins=200, color='b', alpha=0.5);

img = testObj.ALres.decorrelatedDiffim.getMaskedImage().getVariance().getArray()
img2 = testObj.ALres.subtractedExposure.getMaskedImage().getVariance().getArray()
#dit.plotImageGrid((img, img2), imScale=8)

print dit.computeClippedImageStats(img)
print dit.computeClippedImageStats(img2)
# print img[0:80,80]

tmp = img.flatten()
print len(tmp[~np.isnan(tmp)])
plt.hist(tmp[~np.isnan(tmp)], bins=200);
tmp2 = img2.flatten()
print len(tmp2[(~np.isnan(tmp2)) & (np.isfinite(tmp2))])
plt.hist(tmp2[(~np.isnan(tmp2)) & (np.isfinite(tmp2))], bins=200, color='b', alpha=0.5);

reload(dit)
testObj = dit.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(1000., 50),
                         #templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=15.)

df = testObj.doPlotWithDetectionsHighlighted(transientsOnly=True, 
                                             #xaxisIsScienceForcedPhot=True,
                                             addPresub=True)
plt.xlim(200, 2000)
plt.ylim(-0.2, 18);

reload(dit)
testObj2 = dit.DiffimTest(n_sources=5000, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(1000., 50),
                         #templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=15.)

df = testObj2.doPlotWithDetectionsHighlighted(transientsOnly=True, 
                                             #xaxisIsScienceForcedPhot=True,
                                             addPresub=True)
plt.xlim(200, 2000)
plt.ylim(-0.2, 18);

## Just re-create the testObj here:
reload(dit)
testObj = dit.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(1000., 50),
                         templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=15.)

df = testObj.doPlotWithDetectionsHighlighted(transientsOnly=True, 
                                             #xaxisIsScienceForcedPhot=True,
                                             addPresub=True)
plt.xlim(200, 2000)
plt.ylim(-0.2, 18);



