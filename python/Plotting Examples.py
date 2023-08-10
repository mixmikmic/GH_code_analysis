# Import modules.
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.plots as plots
from lsst.sims.maf.metricBundles import MetricBundle, MetricBundleGroup, makeBundlesDictFromList

# Connect to databases.
runName = 'enigma_1189'
opsdb = db.OpsimDatabase(runName + '_sqlite.db')
outDir = 'allfilters_test'
resultsDb = db.ResultsDb(outDir=outDir)

nside = 128
# Set up metrics, slicer and summaryMetrics.
m1 = metrics.CountMetric('expMJD', metricName='Nvisits')
m2 = metrics.Coaddm5Metric()
slicer = slicers.HealpixSlicer(nside=nside)
summaryMetrics = [metrics.MinMetric(), metrics.MeanMetric(), metrics.MaxMetric(), 
                  metrics.MedianMetric(), metrics.RmsMetric(), 
                 metrics.PercentileMetric(percentile=25), metrics.PercentileMetric(percentile=75)]
# And I'll set a plotDict for the nvisits and coadded depth, because otherwise the DD fields throw the 
#  scale in the plots into too wide a range. 
#  (we could also generate plots, see this effect, then set the dict and regenerate the plots)
nvisitsPlotRanges = {'xMin':0, 'xMax':300, 'colorMin':0, 'colorMax':300, 'binsize':5}
coaddPlotRanges = {'xMin':24, 'xMax':28, 'colorMin':24, 'colorMax':28, 'binsize':0.02}

filterlist = ['u', 'g', 'r', 'i', 'z', 'y']
filterorder = {'u':0, 'g':1, 'r':2, 'i':3, 'z':4, 'y':5}

# Create metricBundles for each filter. 
# For ease of access later, I want to make a dictionary with 'nvisits[filter]' and 'coadd[filter]' first.
nvisits = {}
coadd = {}
for f in filterlist:
    sqlconstraint = 'filter = "%s"' %(f)
    # Add displayDict stuff that's useful for showMaf to put things in "nice" order.
    displayDict = {'subgroup':'Undithered', 'order':filterorder[f], 'group':'Nvisits'}
    nvisits[f] = MetricBundle(m1, slicer, sqlconstraint=sqlconstraint, runName=runName,
                              summaryMetrics=summaryMetrics, plotDict=nvisitsPlotRanges,
                              displayDict=displayDict)
    displayDict['group'] = 'Coadd'
    coadd[f] = MetricBundle(m2, slicer, sqlconstraint=sqlconstraint, runName=runName,
                            summaryMetrics=summaryMetrics, plotDict=coaddPlotRanges,
                            displayDict=displayDict)
blistAll = []
for f in filterlist:
    blistAll.append(nvisits[f])
    blistAll.append(coadd[f])
bdict = makeBundlesDictFromList(blistAll)
# Set the metricBundleGroup up with all metricBundles, in all filters.
bgroup = MetricBundleGroup(bdict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.writeAll()
bgroup.plotAll()

print 'Nvisits --'
for f in filterlist:
    print nvisits[f].summaryValues
print 'Coadd --'
for f in filterlist:
    print coadd[f].summaryValues

# Set more complicated plot labels directly in the bundles.
for f in filterlist:
    nvisits[f].setPlotDict({'label':'%s  %1.f/%.1f/%1.f' %(f, nvisits[f].summaryValues['25th%ile'], 
                                                    nvisits[f].summaryValues['Median'], 
                                                   nvisits[f].summaryValues['75th%ile'])})
    coadd[f].setPlotDict({'label':'%s %.2f/%.2f/%.2f' %(f, coadd[f].summaryValues['25th%ile'],
                                                 coadd[f].summaryValues['Median'],
                                                 coadd[f].summaryValues['75th%ile'])})

# Set up the plotHandler.
ph = plots.PlotHandler(outDir=outDir, resultsDb=resultsDb)
# Instantiate the healpix histogram plotter, since we'll use it a lot.
healpixhist = plots.HealpixHistogram()
ph.setMetricBundles(nvisits)
# Add min/max values to the plots, which will be used for the combo histogram for nvisits.
ph.setPlotDicts(nvisitsPlotRanges)
ph.plot(plotFunc=healpixhist)
# And generate the coadd combo histogram too. 
ph.setMetricBundles(coadd)
ph.setPlotDicts(coaddPlotRanges)
ph.plot(plotFunc=healpixhist)

# Set up to calculate the same metrics, but using the dithered pointings.
slicer = slicers.HealpixSlicer(nside=nside, lonCol='ditheredRA', latCol='ditheredDec')

# run dithered bundles
nvisitsDith = {}
coaddDith = {}
for f in filterlist:
    sqlconstraint = 'filter = "%s"' %(f)
    displayDict = {'subgroup':'Dithered', 'order':filterorder[f], 'group':'Nvisits'}
    nvisitsDith[f] = MetricBundle(m1, slicer, sqlconstraint=sqlconstraint, runName=runName,
                                  metadata = '%s Dithered' %(f), 
                                  summaryMetrics=summaryMetrics, plotDict=nvisitsPlotRanges,
                                 displayDict=displayDict)
    displayDict['group'] = 'Coadd'
    coaddDith[f] = MetricBundle(m2, slicer, sqlconstraint=sqlconstraint, runName=runName,
                                metadata = '%s Dithered' %(f), 
                                summaryMetrics=summaryMetrics, plotDict=coaddPlotRanges,
                               displayDict=displayDict)
bListAll = []
for f in filterlist:
    bListAll.append(nvisitsDith[f])
    bListAll.append(coaddDith[f])
bdict = makeBundlesDictFromList(bListAll)
bgroup = MetricBundleGroup(bdict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.writeAll()
bgroup.plotAll()

# set plot labels for dithered bundles
for f in filterlist:
    nvisitsDith[f].setPlotDict({'label':'%s dithered  %1.f/%.1f/%1.f' %(f, nvisitsDith[f].summaryValues['25th%ile'], 
                                                    nvisitsDith[f].summaryValues['Median'], 
                                                   nvisitsDith[f].summaryValues['75th%ile'])})
    coaddDith[f].setPlotDict({'label':'%s dithered %.2f/%.2f/%.2f' %(f, coaddDith[f].summaryValues['25th%ile'],
                                                 coaddDith[f].summaryValues['Median'],
                                                 coaddDith[f].summaryValues['75th%ile'])})    

# Plot all filters, dithered version
ph.setMetricBundles(nvisitsDith)
# Add min/max values to the plots, which will be used for the combo histogram.
ph.setPlotDicts(nvisitsPlotRanges)
ph.plot(plotFunc=healpixhist)
ph.setMetricBundles(coaddDith)
ph.setPlotDicts(coaddPlotRanges)
ph.plot(plotFunc=healpixhist)

# plot dithered vs. non-dithered. Note that this resets xMin/xMax, so that they are set dynamically for each plot.
for f in filterlist:
    ph.setMetricBundles([nvisits[f], nvisitsDith[f]])
    plotDicts = [{'color':'b'}, {'color':'r'}]
    commonDict = {'percentileClip':96, 'binsize':2}
    for pd in plotDicts:
        pd.update(commonDict)
    ph.plot(plotFunc=healpixhist, plotDicts=plotDicts)
    ph.setMetricBundles([coadd[f], coaddDith[f]])
    commonDict = {'percentileClip':96, 'binsize':0.02, 'legendloc':'upper left'}
    for pd in plotDicts:
        pd.update(commonDict)
    ph.plot(plotFunc=healpixhist, plotDicts=plotDicts)

# Save some information on the opsim run itself to disk. 
#  This helps 'showMaf' look pretty and tracks information about the opsim run.
from lsst.sims.maf.utils import writeConfigs
writeConfigs(opsdb, outDir)



