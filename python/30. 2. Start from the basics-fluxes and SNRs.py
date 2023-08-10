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

class sizeme():
    """ Class to change html fontsize of object's representation"""
    def __init__(self, ob, size=50, height=120):
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

# Set up console so we can reattach via terminal ipython later. See:
# https://stackoverflow.com/questions/19479645/using-ipython-console-along-side-ipython-notebook

get_ipython().magic('qtconsole')

# Then do `ipython console --existing` in a terminal to connect and have access to same data!
# But note, do not do CTRL-D in that terminal or it will kill the kernel!

def plotInputVsMeasuredFluxes(testObj, img=1, plotSNR=False):
    src = None
    if img == 1:
        src1 = testObj.im1.doDetection(threshold=3.0)
        src1 = src1[~src1['base_PsfFlux_flag']]
        src = src1
    elif img == 2:
        src2 = testObj.im2.doDetection(threshold=3.0)
        src2 = src2[~src2['base_PsfFlux_flag']]
        src = src2
    elif img == 3:
        src3 = dit.doDetection(testObj.res.subtractedExposure, threshold=3.0)
        src3 = src3[~src3['base_PsfFlux_flag']]
        src = src3
    elif img == 4:
        src4 = testObj.D_ZOGY.doDetection(threshold=3.0)
        src4 = src4[~src4['base_PsfFlux_flag']]
        src = src4

    dist = np.sqrt(np.add.outer(src.base_NaiveCentroid_x, -testObj.centroids[:, 0])**2. +                    np.add.outer(src.base_NaiveCentroid_y, -testObj.centroids[:, 1])**2.) # in pixels
    matches = np.where(dist <= 1.5)
    true_pos = len(np.unique(matches[0]))
    false_neg = testObj.centroids.shape[0] - len(np.unique(matches[1]))
    false_pos = src.shape[0] - len(np.unique(matches[0]))
    detections = {'TP': true_pos, 'FN': false_neg, 'FP': false_pos}
    print detections

    src_hits1 = src.iloc[matches[0],:]
    input_hits1 = testObj.centroids[matches[1],:]
    fluxes = input_hits1[:,2]
    if img == 2 or img == 3 or img == 4:
        fluxes = input_hits1[:,3]

    if not plotSNR:
        plt.scatter(fluxes, src_hits1.base_PsfFlux_flux.values, label='PsfFlux')
        plt.scatter(fluxes, src_hits1.base_PeakLikelihoodFlux_flux.values, label='PeakLikelihoodFlux', color='r')
        plt.legend(loc='upper left', shadow=True)
        #plt.xlim(0, 20000)
        #plt.ylim(0, 40000)
    else:
        try:
            print fluxes[fluxes>0].min(), (src_hits1.base_PsfFlux_flux.values/src_hits1.base_PsfFlux_fluxSigma.values).min()
        except:
            pass
        plt.scatter(fluxes, src_hits1.base_PsfFlux_flux.values/src_hits1.base_PsfFlux_fluxSigma.values)
        #plt.xlim(0, 32000);
    
    return {'input': fluxes, 'psfFlux': src_hits1.base_PsfFlux_flux.values,
            'psfFluxSigma': src_hits1.base_PsfFlux_fluxSigma.values,
            'peakLikelihoodFlux': src_hits1.base_PeakLikelihoodFlux_flux.values}

reload(dit)

testObj = dit.DiffimTest(n_sources=500, sourceFluxRange=(200, 20000), 
                         varFlux2=np.linspace(200, 2000, 200),
                        templateNoNoise=False, skyLimited=True)
res = testObj.runTest()
print res

tmp = plotInputVsMeasuredFluxes(testObj, img=1)

plotInputVsMeasuredFluxes(testObj, img=2);

print np.median(tmp['peakLikelihoodFlux']/tmp['psfFlux'])
print np.median(tmp['psfFlux']/tmp['input'])
plt.scatter(tmp['input'], tmp['peakLikelihoodFlux']/tmp['psfFlux'])
plt.scatter(tmp['input'], tmp['psfFlux']/tmp['input'], color='r')
plt.ylim(-0.1,5)

plotInputVsMeasuredFluxes(testObj, img=1, plotSNR=True);

plotInputVsMeasuredFluxes(testObj, img=2, plotSNR=True);

tmp = plotInputVsMeasuredFluxes(testObj, img=3, plotSNR=True)

tmp = plotInputVsMeasuredFluxes(testObj, img=4, plotSNR=True)

reload(dit)

testObj = dit.DiffimTest(n_sources=510, sourceFluxRange=(500, 120000), 
                         varFlux2=np.linspace(500, 5000, 500),
                         templateNoNoise=False, skyLimited=True,
                         variablesNearCenter=False,
                         verbose=False)
res = testObj.runTest(returnSources=True)
sources = res['sources']
del res['sources']
print res

fig = plt.figure(1, (15, 15))
testObj.doPlot(nrows_ncols=(5, 2))

tmp = plotInputVsMeasuredFluxes(testObj, img=3, plotSNR=False)

tmp = plotInputVsMeasuredFluxes(testObj, img=3, plotSNR=True)

tmp = plotInputVsMeasuredFluxes(testObj, img=4, plotSNR=True)

centroids = testObj.centroids
sizeme(pd.DataFrame(centroids).head())

import lsst.afw.table as afwTable
schema = afwTable.SourceTable.makeMinimalSchema()
centroidKey = afwTable.Point2DKey.addFields(schema, 'centroid', 'centroid', 'pixel')
schema.getAliasMap().set('slot_Centroid', 'centroid')
#schema.addField('centroid_x', type=float, doc='x pixel coord')
#schema.addField('centroid_y', type=float, doc='y pixel coord')
schema.addField('inputFlux_template', type=float, doc='input flux in template')
schema.addField('inputFlux_science', type=float, doc='input flux in science image')
table = afwTable.SourceTable.make(schema)
sources = afwTable.SourceCatalog(table)

import lsst.afw.geom as afwGeom
import lsst.afw.detection as afwDetection

footprint_radius = 5  # pixels
exposure = testObj.im2.asAfwExposure()
expWcs = exposure.getWcs()

for row in centroids:
    record = sources.addNew()
    coord = expWcs.pixelToSky(row[0], row[1])
    record.setCoord(coord)
    record.set(centroidKey, afwGeom.Point2D(row[0], row[1]))
    record.set('inputFlux_template', row[2])
    record.set('inputFlux_science', row[3])
    #print record.getCentroid()
    
    #print expWcs.skyToPixel(coord), type(expWcs.skyToPixel(coord)), row
    fpCenter = afwGeom.Point2I(afwGeom.Point2D(row[0], row[1])) #expWcs.skyToPixel(coord))
    footprint = afwDetection.Footprint(fpCenter, footprint_radius)
    record.setFootprint(footprint)
    
print sources.schema.getNames()
am = sources.schema.getAliasMap()
print am.keys(), am.values()    

sources = sources.copy(deep=True)  # make it contiguous
tmp = pd.DataFrame({col: sources.columns[col] for col in sources.schema.getNames()})
sizeme(tmp.head())

plt.scatter(sources['centroid_x'], sources['centroid_y'])

plt.scatter(sources['coord_ra'], sources['coord_dec'])
plt.xlim(3.7613, 3.7622)
plt.ylim(0.9278, 0.9284)

config = measBase.ForcedMeasurementTask.ConfigClass() #ForcedExternalCatalogMeasurementTask.ConfigClass()
config.plugins.names = ["base_TransformedCentroid", "base_PsfFlux"]
config.slots.shape = None
config.slots.centroid = "base_TransformedCentroid"
config.slots.modelFlux = "base_PsfFlux"
measurement = measBase.ForcedMeasurementTask(schema, config=config)
measCat = measurement.generateMeasCat(exposure, sources, expWcs)
measurement.attachTransformedFootprints(measCat, sources, exposure, expWcs)
measurement.run(measCat, exposure, sources, expWcs)

tmp = pd.DataFrame({col: measCat.columns[col] for col in measCat.schema.getNames()})
sizeme(tmp.head())

tmp = pd.DataFrame({col: sources.columns[col] for col in sources.schema.getNames()})
sizeme(tmp.head())

plt.scatter(sources['inputFlux_science'], measCat['base_PsfFlux_flux'])

plt.scatter(sources['inputFlux_science'], measCat['base_PsfFlux_flux']/measCat['base_PsfFlux_fluxSigma'])

def doForcedPhotometry(testObj, exposure=None, asDF=False):
    import lsst.afw.table as afwTable
    import lsst.afw.geom as afwGeom
    import lsst.afw.detection as afwDetection

    centroids = testObj.centroids
    
    schema = afwTable.SourceTable.makeMinimalSchema()
    centroidKey = afwTable.Point2DKey.addFields(schema, 'centroid', 'centroid', 'pixel')
    schema.getAliasMap().set('slot_Centroid', 'centroid')
    #schema.addField('centroid_x', type=float, doc='x pixel coord')
    #schema.addField('centroid_y', type=float, doc='y pixel coord')
    schema.addField('inputFlux_template', type=float, doc='input flux in template')
    schema.addField('inputFlux_science', type=float, doc='input flux in science image')
    table = afwTable.SourceTable.make(schema)
    sources = afwTable.SourceCatalog(table)
    
    footprint_radius = 5  # pixels
    if exposure is None:
        exposure = testObj.im2.asAfwExposure()
    expWcs = exposure.getWcs()

    for row in centroids:
        record = sources.addNew()
        coord = expWcs.pixelToSky(row[0], row[1])
        record.setCoord(coord)
        record.set(centroidKey, afwGeom.Point2D(row[0], row[1]))
        record.set('inputFlux_template', row[2])
        record.set('inputFlux_science', row[3])

        fpCenter = afwGeom.Point2I(afwGeom.Point2D(row[0], row[1])) #expWcs.skyToPixel(coord))
        footprint = afwDetection.Footprint(fpCenter, footprint_radius)
        record.setFootprint(footprint)

    sources = sources.copy(deep=True)  # make it contiguous
    
    config = measBase.ForcedMeasurementTask.ConfigClass() #ForcedExternalCatalogMeasurementTask.ConfigClass()
    config.plugins.names = ["base_TransformedCentroid", "base_PsfFlux"]
    config.slots.shape = None
    config.slots.centroid = "base_TransformedCentroid"
    config.slots.modelFlux = "base_PsfFlux"
    measurement = measBase.ForcedMeasurementTask(schema, config=config)
    measCat = measurement.generateMeasCat(exposure, sources, expWcs)
    measurement.attachTransformedFootprints(measCat, sources, exposure, expWcs)
    measurement.run(measCat, exposure, sources, expWcs)
    
    if asDF:
        measCat = pd.DataFrame({col: measCat.columns[col] for col in measCat.schema.getNames()})
    return measCat

mc1 = doForcedPhotometry(testObj, testObj.im1.asAfwExposure())
mc2 = doForcedPhotometry(testObj, testObj.im2.asAfwExposure())

plt.scatter(sources['inputFlux_science'], mc1['base_PsfFlux_flux'])
plt.scatter(sources['inputFlux_science'], mc2['base_PsfFlux_flux'], color='r')



