import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from moSlicer import MoSlicer
import moMetrics as MoMetrics
import moSummaryMetrics as MoSummaryMetrics
import moStackers as MoStackers
import moPlots as moPlots
import moMetricBundle as mmb
import lsst.sims.maf.plots as plots

# Set up MoSlicer.  We use the moving object slicer to read the observation data from disk.
test = True
mbas = False
if test:
    orbitfile = 'test.des'
    obsfile = 'test_out.txt'
    runName = 'enigma_1189'
    metadata = '8 NEOs'
    outDir = 'test'
else:
    orbitfile = 'pha20141031.des'
    obsfile = 'pha_withChip_allObs.txt'
    runName = 'enigma_1189'
    metadata = 'PHAS with camera footprint'
    outDir = 'pha'
if mbas:
    orbitfile = 'mbas_10k.des'
    obsfile = 'mbas10k_allObs.txt'
    runName = 'enigma_1189'
    metadata = '10K MBAs'
    outDir = 'mba'

mos = MoSlicer(orbitfile, Hrange=np.arange(13, 26, 0.5))
mos.readObs(obsfile)
print mos.slicePoints['H'], len(mos.slicePoints['H'])
mos.allObs.tail(10)

mos.orbits.tail()

# Set up an example metric bundle.
metric = MoMetrics.NObsMetric()
slicer = mos
pandasConstraint = None
if test:
    plotDict = {'nxbins':20, 'nybins':20}
else:
    plotDict = {'nxbins':100, 'nybins':100}
nobs = mmb.MoMetricBundle(metric, slicer, pandasConstraint, 
                          runName=runName, metadata=metadata, plotDict=plotDict)

# Calculate completeness. First we must calculate "DiscoveryChances". 
# Set up an example metric bundle.
metric = MoMetrics.DiscoveryMetric()
slicer = mos
pandasConstraint = None
discovery = mmb.MoMetricBundle(metric, slicer, pandasConstraint, 
                               runName=runName, metadata=metadata, plotDict=plotDict)

# try a different kind of metric, looking at chances of finding activity
metric = MoMetrics.ActivityOverTimeMetric(window=6*30.)
slicer = mos
pandasConstraint = None
activity6month = mmb.MoMetricBundle(metric, slicer, pandasConstraint, 
                               runName=runName, metadata=metadata, plotDict=plotDict)

bdict = {'nobs':nobs, 'discovery':discovery, 'activity':activity6month}
bg = mmb.MoMetricBundleGroup(bdict, outDir=outDir)
bg.runAll()

bg.plotAll(closefigs=False)

ph = plots.PlotHandler(outDir=outDir)
ph.setMetricBundles([discovery, discovery])
ph.setPlotDicts(plotDicts=[{'npReduce':np.mean, 'color':'b', 'label':'Mean'},
                           {'npReduce':np.median, 'color':'g', 'label':'Median'}])
ph.plot(plotFunc=moPlots.MetricVsH(), plotDicts={'ylabel':'Discovery Chances @ H'})

# Then calculate 'completeness' as function of H, as a secondary metric.
completeness = discovery.reduceMetric(discovery.metric.reduceFuncs['Completeness'])
completeness.plot(plotHandler=ph)

# And we can make an 'integrated over H distribution' version. 
completenessInt = completeness.reduceMetric(completeness.metric.reduceFuncs['CumulativeH'])
completenessInt.plot()

Hmark = 22.0
for c in [completeness, completenessInt]:
    print c
    summaryMetric = ValueAtHMetric(Hmark=Hmark)
    c.setSummaryMetrics(summaryMetric)
    c.computeSummaryStats()
    label = "Completeness at H=%.1f: %.2f" %(Hmark, c.summaryValues['Value At H=%.1f' %Hmark])
    c.setPlotDict({'label':label})
    c.plot(plotFunc = moPlots.MetricVsH())
    plt.axvline(Hmark, color='r', linestyle=':')
    plt.axhline(c.summaryValues['Value At H=%.1f' %(Hmark)], color='r', linestyle='-')
    plt.legend(loc=(0.9, 0.2))

### side note.. illustrating detection probability curve (50% at m5)
# this probability curve is used if snrLimit is NOT set (otherwise it's just a cutoff at snrLimit)
metric = MoMetrics.BaseMoMetric()
# Check what 'visibility' looks like. 
nobjs = 10000
appMag = np.random.rand(nobjs) * 5.0 + 20.0
magLimit = 22.8
vis = metric._calcVis(appMag, magLimit)
bins = np.arange(appMag.min(), appMag.max(), 0.1)
vn, b = np.histogram(appMag[vis], bins=bins)
n, b = np.histogram(appMag, bins=bins)
plt.plot(b[:-1], vn / n.astype(float), 'k-')
plt.xlabel('apparent magnitude')
plt.ylabel('probability of detection')





