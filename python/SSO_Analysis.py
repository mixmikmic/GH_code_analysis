import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from moSlicer import MoSlicer
import moMetrics as moMetrics
from moSummaryMetrics import ValueAtHMetric
import moPlots as moPlots
import moMetricBundle as mmb
import lsst.sims.maf.plots as plots
import lsst.sims.maf.db as db

slicers = {}
objtypes = ['mba_10k', 'mba_2k']

obj = 'mba_2k'
Hrange = np.arange(15, 26, 0.25)
orbitfile = 'ssm_2k/mbas_2k.des'
obsfile = 'ssm_2k/mbas_2k_allObs.txt'
slicers[obj] = MoSlicer(orbitfile, Hrange=Hrange)
slicers[obj].readObs(obsfile)

obj = 'mba_10k'
orbitfile = 'mbas_10k.des'
obsfile = 'mbas_10k_allObs.txt'
slicers[obj] = MoSlicer(orbitfile, Hrange=Hrange)
slicers[obj].readObs(obsfile)

runName = 'enigma_1189'

mBundles = {}

for obj in objtypes:
    mbundles = {}
    slicer = slicers[obj]
    plotDict = {}
    pandasConstraint = None
    metadata = obj
    metric = moMetrics.NObsMetric()
    mbundles['NObs'] = mmb.MoMetricBundle(metric, slicer, pandasConstraint, 
                                          runName=runName, metadata=metadata, plotDict=plotDict)
    metric = moMetrics.DiscoveryChancesMetric()
    mbundles['Discoveries'] = mmb.MoMetricBundle(metric, slicer, pandasConstraint, 
                                   runName=runName, metadata=metadata, plotDict=plotDict)

    metric = moMetrics.ObsArcMetric()
    mbundles['Arclength'] = mmb.MoMetricBundle(metric, slicer, pandasConstraint,
                                              runName=runName, metadata=metadata, plotDict=plotDict)
    metric = moMetrics.NNightsMetric()
    mbundles['NNights'] = mmb.MoMetricBundle(metric, slicer, pandasConstraint,
                                            runName=runName, metadata=metadata, plotDict=plotDict)
    mBundles[obj] = mbundles

outDir = 'mba_comp'
resultsDb = db.ResultsDb(outDir=outDir)
for obj in objtypes:
    mbg = mmb.MoMetricBundleGroup(mBundles[obj], outDir=outDir, resultsDb=resultsDb)
    mbg.runAll()
    discovery = mBundles[obj]['Discoveries']
    completeness = discovery.reduceMetric(discovery.metric.reduceFuncs['Completeness'])
    mBundles[obj]['completeness'] = completeness
    completenessInt = completeness.reduceMetric(completeness.metric.reduceFuncs['CumulativeH'])
    mBundles[obj]['completenessInt'] = completenessInt

ph = plots.PlotHandler()
plotFunc = moPlots.MetricVsH()
mplot = 'completeness'
ph.setMetricBundles([mBundles['mba_2k'][mplot], mBundles['mba_2k'][mplot],
                     mBundles['mba_10k'][mplot], mBundles['mba_10k'][mplot]])
ph.setPlotDicts([{'color':'b', 'npReduce':np.mean, 'label':'2k mean'}, 
                 {'color':'g', 'npReduce':np.median, 'label':'2k median'},
                {'color':'r', 'npReduce':np.mean, 'label':'10k mean'}, 
                {'color':'m', 'npReduce':np.median, 'label':'10k median'}])
ph.plot(plotFunc=plotFunc, plotDicts={'ylabel':'%s @H' %(mplot)})

Hmark = 21.0
summaryMetric = ValueAtHMetric(Hmark=Hmark)
mBundles['mba_10k']['completeness'].setSummaryMetrics(summaryMetric)
mBundles['mba_10k']['completeness'].computeSummaryStats()
print '10k', np.mean(mBundles['mba_10k']['completeness'].summaryValues['Value At H=%.1f' %(Hmark)])
mBundles['mba_2k']['completeness'].setSummaryMetrics(summaryMetric)
mBundles['mba_2k']['completeness'].computeSummaryStats()
print '2k', np.mean(mBundles['mba_2k']['completeness'].summaryValues['Value At H=%.1f' %(Hmark)])

# Go through all types of objects, with 2k results. 
moslicers = {}
objtypes = ['neos', 'mbas', 'trojans', 'tnos', 'sdos', 'comets']
Hrange = np.arange(5, 27, 0.25)
for obj in objtypes:
    orbitfile = os.path.join('ssm_2k', obj+'_2k.des')
    moslicers[obj] = MoSlicer(orbitfile, Hrange)
    obsfile = os.path.join('ssm_2k', obj+'_2k_allObs.txt')
    moslicers[obj].readObs(obsfile)

runName = 'enigma_1189'
allBundles = {}
for obj in objtypes:
    bundles = {}
    slicer = moslicers[obj]
    plotDict = {}
    pandasConstraint = None
    metadata = obj
    metric = moMetrics.NObsMetric()
    bundles['NObs'] = mmb.MoMetricBundle(metric, slicer, pandasConstraint, 
                                          runName=runName, metadata=metadata, plotDict=plotDict)
    metric = moMetrics.NObsNoSinglesMetric()
    bundles['NObs NoSingles'] = mmb.MoMetricBundle(metric, slicer, pandasConstraint, 
                                          runName=runName, metadata=metadata, plotDict=plotDict)
    metric = moMetrics.DiscoveryChancesMetric()
    bundles['Discoveries'] = mmb.MoMetricBundle(metric, slicer, pandasConstraint, 
                                   runName=runName, metadata=metadata, plotDict=plotDict)

    metric = moMetrics.ObsArcMetric()
    bundles['Arclength'] = mmb.MoMetricBundle(metric, slicer, pandasConstraint,
                                              runName=runName, metadata=metadata, plotDict=plotDict)
    metric = moMetrics.NNightsMetric()
    bundles['N Nights'] = mmb.MoMetricBundle(metric, slicer, pandasConstraint,
                                            runName=runName, metadata=metadata, plotDict=plotDict)
    window = 180.0
    metric = moMetrics.ActivityOverTimeMetric(window, metricName='Chances of Detecting Activity lasting %.0f days' %(window))
    bundles['Chance of Detecting Activity (6 mnth)'] = mmb.MoMetricBundle(metric, slicer, pandasConstraint,
                                                runName=runName, metadata=metadata, plotDict=plotDict)
    window = 90.
    metric = moMetrics.ActivityOverTimeMetric(window, metricName='Chances of Detecting Activity lasting %0.f days' %(window))
    bundles['Chance of Detecting Activity (3 mnth)'] = mmb.MoMetricBundle(metric, slicer, pandasConstraint,
                                                runName=runName, metadata=metadata, plotDict=plotDict)
    allBundles[obj] = bundles

outDir = 'ssm_2k_comp'
resultsDb = db.ResultsDb(outDir=outDir)
for obj in objtypes:
    mbg = mmb.MoMetricBundleGroup(allBundles[obj], outDir=outDir, resultsDb=resultsDb)
    mbg.runAll()
    discovery = allBundles[obj]['Discoveries']
    completeness = discovery.reduceMetric(discovery.metric.reduceFuncs['Completeness'])
    allBundles[obj]['completeness'] = completeness
    completenessInt = completeness.reduceMetric(completeness.metric.reduceFuncs['CumulativeH'])
    allBundles[obj]['completenessInt'] = completenessInt

plotdicts = {}
plotdicts['neos'] = {'color':'cyan', 'label':'NEOs'}
plotdicts['mbas'] = {'color':'g', 'label':'MBAs'}
plotdicts['trojans'] = {'color':'y', 'label':'TR5s'}
plotdicts['tnos'] = {'color':'r', 'label':'TNOs'}
plotdicts['sdos'] = {'color':'m', 'label':'SDOs'}
plotdicts['comets'] = {'color':'k', 'label':'Comets'}

reload(moPlots)
# PLOT the MEDIAN values of each of these metrics as function of H.
objtypes = ['neos', 'mbas', 'trojans', 'tnos', 'sdos', 'comets']
Hrange = np.arange(5, 27, 0.25)
for mplot in allBundles['neos'].keys():
    ph = plots.PlotHandler(outDir=outDir, figformat='png', thumbnail=False)
    plotFunc = moPlots.MetricVsH()
    bundles = []
    pds = []
    for obj in objtypes:
        bundles.append(allBundles[obj][mplot])
        pds.append(plotdicts[obj])
    ph.setMetricBundles(bundles)
    ph.setPlotDicts(pds)
    ylabel = '%s %s' %(mplot, bundles[0].plotDict['units'])
    ph.plot(plotFunc=plotFunc, plotDicts={'npReduce':np.median, 'ylabel':ylabel, 'albedo':0.14,
                                         'xMin':Hrange[0], 'xMax':Hrange[-1]})

# Take a closer look at the NEO discovery chances. 
ph = plots.PlotHandler(outDir=outDir)
ph.setMetricBundles([allBundles['neos']['Discoveries'], 
                     allBundles['neos']['Discoveries'], 
                     allBundles['neos']['Discoveries']])
ph.setPlotDicts([{'npReduce':np.median, 'label':'median'}, 
                 {'npReduce':np.mean, 'label':'mean'}, 
                 {'npReduce':np.max, 'label':'max'}])
ph.plot(plotFunc=moPlots.MetricVsH(), plotDicts={'yMin':0, 'yMax':80, 'xMin':15, 'xMax':27, 
                                                 'ylabel':'Discovery chances @H'})

mval = 'NObs'
Hmark = 5
summaryMetric = ValueAtHMetric(Hmark=Hmark)
for obj in objtypes:
    allBundles[obj][mval].setSummaryMetrics(summaryMetric)
    allBundles[obj][mval].computeSummaryStats()
    print obj, Hmark, mval, np.median(allBundles[obj][mval].summaryValues['Value At H=%.1f' %(Hmark)])

mag_sun = -27.1 
km_per_au = 1.496e8
albedo = 0.14
diameter = 2.0 * np.sqrt(10**((mag_sun - Hrange - 2.5*np.log10(albedo))/2.5))
diameter = diameter * km_per_au
for H, d in zip(Hrange, diameter):
    print H, d

#reload(moMetrics)
slicer = moslicers['mbas']
windows = np.arange(1, 40, 3)*5
bins = np.arange(5, 95, 10.)
print windows, bins
abundles = {}
for w in windows:
    metric = moMetrics.ActivityOverTimeMetric(w, metricName='Chances of Detecting Activity lasting %.0f days' %(w))
    abundles['TActivity %.0f' %w] = mmb.MoMetricBundle(metric, slicer, pandasConstraint,
                                                    runName=runName, metadata=metadata, plotDict=plotDict)
for b in bins:
    metric = moMetrics.ActivityOverPeriodMetric(b, metricName=
                                                'Chances of Detecting Activity lasting %.0f of the period' %(b/360.0))
    abundles['PActivity %.0f' %b] = mmb.MoMetricBundle(metric, slicer, pandasConstraint,
                                                    runName=runName, metadata=metadata, plotDict=plotDict)

outDir = 'ssm_2k_comp'
resultsDb = db.ResultsDb(outDir=outDir)
mbg = mmb.MoMetricBundleGroup(abundles, outDir=outDir, resultsDb=resultsDb)
mbg.runAll()

# Plot the min/mean/max of the fraction of activity detection opportunities, over all objects
# Need to make more summary statistics to do this more elegantly.
meanFraction = np.zeros(len(windows), float)
minFraction = np.zeros(len(windows), float)
maxFraction = np.zeros(len(windows), float)

Hmark = 15.0
Hidx = np.where(moslicers['neos'].Hrange  == Hmark)[0]
for i, w in enumerate(windows):
    b = abundles['TActivity %.0f' %w]
    meanFraction[i] = np.mean(b.metricValues.swapaxes(0, 1)[Hidx])
    minFraction[i] = np.min(b.metricValues.swapaxes(0, 1)[Hidx])
    maxFraction[i] = np.max(b.metricValues.swapaxes(0, 1)[Hidx])
    
plt.figure()
plt.plot(windows, meanFraction, 'r', label='Mean')
plt.plot(windows, minFraction, 'b--', label='Min')
plt.plot(windows, maxFraction, 'g--', label='Max')
plt.xlabel('Length of activity (days)')
plt.ylabel('Chance of detecting activity lasting X days')
plt.title('Chances of detecting activity (for H=%.1f MBA)' %(Hmark))
plt.grid()
plt.savefig(os.path.join(outDir, 'activity.png'), format='png')

meanFraction = {}
Hs = [15, 18.5, 20]
for Hmark in  Hs:
    meanFraction[Hmark] = np.zeros(len(windows), float)
    Hidx = np.where(moslicers['neos'].Hrange == Hmark)[0]
    for i, w in enumerate(windows):
        b = abundles['TActivity %.0f' %w]
        meanFraction[Hmark][i] = np.mean(b.metricValues.swapaxes(0, 1)[Hidx])

for H in Hs:
    plt.plot(windows, meanFraction[H], label='H=%.1f' %H)
plt.xlabel('Length of activity (days)')
plt.ylabel('Chance of detecting activity lasting X days')
plt.title('Chances of detecting activity')
plt.grid()
plt.legend(loc='lower right', fancybox=True, numpoints=1)
plt.savefig(os.path.join(outDir, 'activity2.png'), format='png')

meanFraction = {}
Hs = [15, 18.5, 20]
for Hmark in  Hs:
    meanFraction[Hmark] = np.zeros(len(bins), float)
    Hidx = np.where(moslicers['neos'].Hrange  == Hmark)[0]
    for i, w in enumerate(bins):
        b = abundles['PActivity %.0f' %w]
        meanFraction[Hmark][i] = np.mean(b.metricValues.swapaxes(0, 1)[Hidx])
plt.figure()
for H in Hs:
    plt.plot(bins, meanFraction[H], label='H=%.1f' %H)
plt.xlabel('Length of activity (fraction of period)')
plt.ylabel('Chance of detecting activity lasting X of the period')
plt.title('Chances of detecting activity')
plt.grid()
plt.legend(loc='lower right', fancybox=True, numpoints=1)
plt.savefig(os.path.join(outDir, 'activity3.png'), format='png')

reload(moPlots)
Hrange = np.arange(15, 26, 0.25)
m = 'NObs'
ph = plots.PlotHandler(outDir = 'mba_comp', figformat='png', dpi=600, thumbnail=False)
plotFuncs = [moPlots.MetricVsH(), moPlots.MetricVsOrbit(xaxis='a', yaxis='e'), 
             moPlots.MetricVsOrbit(xaxis='a', yaxis='inc')]
mBundles['mba_10k'][m].setPlotFuncs(plotFuncs)
mBundles['mba_10k'][m].setPlotDict({'nxbins':300, 'nybins':300, 
                                         'Hval':18, 'label':m, 'colorMin':1, 'colorMax':400})
mBundles['mba_10k'][m].plot(plotHandler=ph, savefig=True)

m = 'completeness'
ph = plots.PlotHandler(outDir = 'mba_comp', figformat='png', dpi=600, thumbnail=False)
plotFuncs = [moPlots.MetricVsH()]
mBundles['mba_10k'][m].setPlotFuncs(plotFuncs)
mBundles['mba_10k'][m].setPlotDict({})
mBundles['mba_10k'][m].plot(plotHandler=ph, savefig=True)

# Turn completeness into number of objects.
Hrange = np.arange(15, 26, 0.25)
Hbinsize = np.unique(np.diff(Hrange))[0]
Hextension = np.arange(7, Hrange.min()-Hbinsize/2.0, Hbinsize)
bigHrange = np.concatenate([Hextension, Hrange])
bigcompleteness = np.concatenate([np.ones(len(Hextension)), mBundles['mba_10k']['completeness'].metricValues[0]])

diffHrange = np.concatenate([(bigHrange - Hbinsize/2.0), np.array([Hrange.max() + Hbinsize/2.0])])
x = diffHrange - 15.7
ncum = 267000 * (np.power(10, 0.43*x))/(np.power(10, 0.18*x) + np.power(10, -0.18*x))
ndiff = ncum[1:] - ncum[:-1]
nfound = ndiff * bigcompleteness

ncum2 = np.zeros(len(diffHrange))
C = np.log10(50000) # @ H=14.5
condition = np.where(diffHrange>14.5)
ncum2[condition] = np.power(10, C + 0.38 * (diffHrange[condition] - 14.5) )
condition = np.where(diffHrange<=14.5)
ncum2[condition] = np.power(10, C + 0.51 * (diffHrange[condition] - 14.5))
ndiff2 = ncum2[1:] - ncum2[:-1]
nfound2 = ndiff2 * bigcompleteness
#for H, N, Nf in zip(Hrange, ncum, nfound):
#    print H, N, Nf
fig = plt.figure()
plt.semilogy(bigHrange, nfound)
plt.semilogy(bigHrange, nfound2, 'r-')
ax = fig.gca()
plt.xlim(3, 27)

import pandas as pd
ast = pd.read_table('asteroidDb/mpc_mba.dat', delim_whitespace=True, error_bad_lines=False)

ast.columns.values

ast['H'].plot(kind='hist', bins=50, logy=True, alpha=0.5, color='r', edgecolor='none', label='Known MBAs')
plt.xlabel('H Mag')
plt.ylabel('Number of known MBAs')
plt.xlim(3, 27)

# Differential count of number of MBAs
plt.plot(bigHrange[:-9], nfound[:-9],  color='k', label='LSST (SDSS)')
plt.plot(bigHrange[:-9], nfound2[:-9], color='k', linestyle=':', label='LSST (SKADS)')
plt.hist(ast['H'], bins=60, alpha=0.6, color='r', edgecolor='none', label='MPC MBAs')
leg = plt.legend(loc=(0.77, 0.77), fancybox=True, numpoints=1, fontsize='smaller')
leg.get_frame().set_zorder(0)
leg.get_frame().set_facecolor('white')
plt.ylabel("Number of MBAs @ H")
plt.xlabel("H mag")
plt.xlim(2, 28.5)

mag_sun = -27.1 
km_per_au = 1.496e8
albedo = 0.14
ax = plt.axes()
ax2 = ax.twiny()
hmin, hmax = ax.get_xlim()
dmax = 2.0 * np.sqrt(10**((mag_sun - hmin - 2.5*np.log10(albedo))/2.5))
dmin = 2.0 * np.sqrt(10**((mag_sun - hmax - 2.5*np.log10(albedo))/2.5))
dmax = dmax * km_per_au
dmin = dmin * km_per_au
ax2.set_xlim(dmax, dmin)
ax2.set_xscale('log')
ax2.set_xlabel('D (km)', labelpad=-10, horizontalalignment='center')
plt.sca(ax)
plt.grid(True)
plt.yscale('log', nonposy='clip')
plt.figtext(0.2, 0.72, '>826M observations')
plt.tight_layout()
plt.savefig(os.path.join(outDir, 'N_MBAs.png'), dpi=600)

# cumulative number of objects (<=H)
plt.plot(bigHrange, nfound.cumsum(),  color='k', label='LSST (SDSS)')
plt.plot(bigHrange, nfound2.cumsum(), color='k', linestyle=':', label='LSST (SKADS)')
plt.hist(ast['H'], bins=60, alpha=0.6, color='r', edgecolor='none', label='MPC MBAs', cumulative=True)
leg = plt.legend(loc=(0.77, 0.3), fancybox=True, numpoints=1, fontsize='smaller')
leg.get_frame().set_zorder(0)
leg.get_frame().set_facecolor('white')
plt.ylabel("Number of MBAs <= H")
plt.xlabel("H mag")
plt.xlim(2, 28.5)

mag_sun = -27.1 
km_per_au = 1.496e8
albedo = 0.14
ax = plt.axes()
ax2 = ax.twiny()
hmin, hmax = ax.get_xlim()
dmax = 2.0 * np.sqrt(10**((mag_sun - hmin - 2.5*np.log10(albedo))/2.5))
dmin = 2.0 * np.sqrt(10**((mag_sun - hmax - 2.5*np.log10(albedo))/2.5))
dmax = dmax * km_per_au
dmin = dmin * km_per_au
ax2.set_xlim(dmax, dmin)
ax2.set_xscale('log')
ax2.set_xlabel('D (km)', labelpad=-10, horizontalalignment='center')
plt.sca(ax)
plt.grid(True)
plt.yscale('log', nonposy='clip')
plt.figtext(0.2, 0.72, '>826M observations')
plt.tight_layout()
plt.savefig(os.path.join(outDir, 'N_MBAs_cum.png'), dpi=600)

# Calculate completeness & number of objects as a function of time. 
tBundles = {}
times = np.arange(1, 11, 1)
for t in times:
    tBundles[t] = {}
    slicer = slicers['mba_10k']
    plotDict = {}
    pandasConstraint = 'night<=%d' %(t*365)
    metadata = 'year %d' %t
    metric = moMetrics.NObsMetric()
    tBundles[t]['NObs'] = mmb.MoMetricBundle(metric, slicer, pandasConstraint, 
                                          runName=runName, metadata=metadata, plotDict=plotDict)
    metric = moMetrics.DiscoveryChancesMetric()
    tBundles[t]['Discoveries'] = mmb.MoMetricBundle(metric, slicer, pandasConstraint, 
                                   runName=runName, metadata=metadata, plotDict=plotDict)

outDir = 'mba_comp'
resultsDb = db.ResultsDb(outDir)
for t in times:
    mbg = mmb.MoMetricBundleGroup(tBundles[t], outDir=outDir, resultsDb=resultsDb)
    mbg.runAll()
    discovery = tBundles[t]['Discoveries']
    completeness = discovery.reduceMetric(discovery.metric.reduceFuncs['Completeness'])
    tBundles[t]['completeness'] = completeness
    completenessInt = completeness.reduceMetric(completeness.metric.reduceFuncs['CumulativeH'])
    tBundles[t]['completenessInt'] = completenessInt

# Turn completeness into number of objects.
Hrange = slicer.slicePoints['H']
Hbinsize = np.unique(np.diff(Hrange))[0]
Hextension = np.arange(10, Hrange.min()-Hbinsize/2.0, Hbinsize)
bigHrange = np.concatenate([Hextension, Hrange])
diffHrange = np.concatenate([(bigHrange - Hbinsize/2.0), np.array([Hrange.max() + Hbinsize/2.0])])
x = diffHrange - 15.7
ncum = 267000 * (np.power(10, 0.43*x))/(np.power(10, 0.18*x) + np.power(10, -0.18*x))
ndiff = ncum[1:] - ncum[:-1]
ncum2 = np.zeros(len(diffHrange))
C = np.log10(50000) # @ H=14.5
condition = np.where(diffHrange>14.5)
ncum2[condition] = np.power(10, C + 0.38 * (diffHrange[condition] - 14.5) )
condition = np.where(diffHrange<=14.5)
ncum2[condition] = np.power(10, C + 0.51 * (diffHrange[condition] - 14.5))
ndiff2 = ncum2[1:] - ncum2[:-1]

ncounts = np.zeros(len(times))
for i, t in enumerate(times):
    bigcompleteness = np.concatenate([np.ones(len(Hextension)), tBundles[t]['completeness'].metricValues[0]])
    nfound = ndiff * bigcompleteness
    ncounts[i] = nfound.sum()
for t, c in zip(times, ncounts):
    print tBundles[t]['completeness'].metadata, t, c, c/ncounts[-1]
plt.plot(times, ncounts/1000000.)
plt.xlabel('Years into survey')
plt.ylabel('Approximate # of MBAs (Millions)')
#plt.axvline(3, color='b', linestyle=':')
plt.axhline(ncounts[-1]*.75/1000000.0, color='r', linestyle='-')
plt.figtext(0.5, 0.5, '75%s discovered after 3 years' %('%'))
plt.savefig(os.path.join(outDir, 'N_MBAPerYear.png'))

tb = []
tbpd = []
for t in times:
    tb.append(tBundles[t]['completeness'])
    tbpd.append({'label':"Year %d" %t})
ph.setMetricBundles(tb)
ph.setPlotDicts(tbpd)
ph.plot(plotFunc=moPlots.MetricVsH(), plotDicts={'legendloc':(0.8, 0.2), 'albedo':0.14})
plt.savefig(os.path.join(outDir,'Completeness_time.png'))

# Total expected number of observations
med_nobsH = np.median(mBundles['mba_10k']['NObs'].metricValues, axis=0)
# Extend nobsH down to smaller H sizes, like above. Assume it's approximately a straight line.
extend_nobsH = np.concatenate([np.ones(len(Hextension))*med_nobsH[0], med_nobsH])
totalNObs = np.sum(extend_nobsH * ndiff)
print totalNObs/1000000.

outDir = 'mba_comp'
m = 'Discoveries'
ph = plots.PlotHandler(outDir=outDir)
try:
    del mBundles['mba_10k'][m].plotDict['label']
except KeyError:
    pass
bundles = [mBundles['mba_10k'][m], mBundles['mba_10k'][m], 
           mBundles['mba_10k'][m], mBundles['mba_10k'][m]]
pds = [{'npReduce':np.median, 'label':'Median', 'color':'b'}, {'npReduce':np.mean, 'label':'Mean', 'color':'g'}, 
       {'npReduce':np.max, 'label':'Max', 'color':'r', 'linestyle':':'}, 
       {'npReduce':np.min, 'label':'Min', 'color':'m', 'linestyle':':'}]
ph.setMetricBundles(bundles)
ph.setPlotDicts(pds)
ph.plot(plotFunc=moPlots.MetricVsH(), plotDicts={'ylabel':'Arc length', 'yMin':0, 'yMax':150})

obs = slicers['mba_10k'].allObs.query('magFilter>(fiveSigmaDepth - dmagDetect)')
obs['velocity'].plot(kind='hist', bins=100)
plt.xlabel('Velocity (deg/day)')
plt.ylabel('# MBAs')

Hmark = 14
# 300 and 500
islice = slicers['mba_10k']._sliceObs(500)
ssoObs = islice['obs']
orbit = islice['orbit']
Href = orbit['H']
base = moMetrics.BaseMoMetric()
allappMag, magLimit, vis, snr = base._prep(ssoObs, orbit, Hmark)
print len(vis)

appMag = allappMag[vis]
fcolor = ssoObs['filter'][vis]
phase = ssoObs['phase'][vis]
time = ssoObs['expMJD'][vis] - ssoObs['expMJD'][vis][0]
colors = {'u':'m', 'g':'b', 'r':'g', 'i':'y', 'z':'r', 'y':'k'}
for f in ('u', 'g' ,'r', 'i', 'z', 'y'):
    match = np.where(fcolor == f)
    plt.plot(time[match], appMag[match], linestyle='', marker='.', color=colors[f], label='%s' %f)
plt.xlabel('Time from first observation (days)')
plt.ylabel('Magnitude (in filter)')
plt.xlim(0, 4000)
plt.legend(loc=(0.9, 0.1), fancybox=True, fontsize='smaller', numpoints=1)
plt.title('MBA with $a/e/i$ $%.2f/%.1f/%.1f$ at H=%.1f' %(orbit['a'], orbit['e'], orbit['inc'], Hmark))
plt.savefig(os.path.join(outDir,'mba_phot_%.0f.png' %(Hmark)))

plt.figure()
for f in ('u', 'g' ,'r', 'i', 'z', 'y'):
    match = np.where(fcolor == f)
    plt.plot(time[match], appMag[match], linestyle='', marker='.', color=colors[f], label='%s' %f)
plt.xlabel('Time from first observation (days)')
plt.ylabel('Magnitude (in filter)')
#plt.xlim(1850, 2050)
plt.xlim(3200, 3500)
plt.legend(loc=(0.9, 0.1), fancybox=True, fontsize='smaller', numpoints=1)
plt.title('MBA with $a/e/i$ $%.2f/%.1f/%.1f$ at H=%.1f' %(orbit['a'], orbit['e'], orbit['inc'], Hmark))
plt.savefig(os.path.join(outDir, 'mba_phot_zoom_%.0f.png' %(Hmark)))

plt.figure()
for f in ('u', 'g' ,'r', 'i', 'z', 'y'):
    match = np.where(fcolor == f)
    plt.plot(phase[match], appMag[match], linestyle='', marker='.', color=colors[f], label='%s' %f)
plt.xlabel('Phase angle')
plt.ylabel('Magnitude (in filter)')
plt.xlim(0, 28)
plt.legend(loc=(0.9, 0.1), fancybox=True, fontsize='smaller', numpoints=1)
plt.title('MBA with $a/e/i$ $%.2f/%.1f/%.1f$ at H=%.1f' %(orbit['a'], orbit['e'], orbit['inc'], Hmark))
plt.savefig(os.path.join(outDir, 'mba_phase%.0f.png' %(Hmark)))

# Make plot for Mario showing "how many times did you have to find objects that you already know you could have seen?"
outDir = 'ssm_2k_comp'
b = allBundles['neos']['Discoveries']
Hrange = b.slicer.Hrange
n_med_discoveries = np.zeros(len(Hrange), float)
n_mean_discoveries = np.zeros(len(Hrange), float)
n_min_discoveries = np.zeros(len(Hrange), float)
mVals = b.metricValues.swapaxes(0, 1)
for i, (H, mVal) in enumerate(zip(Hrange, mVals)):  
    match = np.where(mVal>0)
    #print H, len(match[0]), mVal[match]
    n_med_discoveries[i] = np.median(mVal[match])
    n_mean_discoveries[i] = np.mean(mVal[match])
    n_min_discoveries[i] = np.min(mVal[match])
plt.plot(Hrange, n_med_discoveries, label='Median')
plt.plot(Hrange, n_mean_discoveries, label='Mean')
plt.plot(Hrange, n_min_discoveries, label='Minimum')
plt.legend(loc='upper right', fancybox=True, fontsize='smaller')
#plt.ylim(0, 5)
#plt.xlim(20, Hrange.max())
plt.xlabel('H mag')
plt.ylabel('Median number of discovery chances (chance>0)')

# Histogram of discovery chances at a given H
plt.style.use('ggplot')
plt.figure(figsize=(10, 6))
Hmark = 22
Hidx = np.where(Hrange == Hmark)[0][0]
mVals = b.metricValues.swapaxes(0, 1)[Hidx]
#print Hrange[Hidx], np.mean(mVals)
plt.hist(mVals[np.where(mVals>0)], bins=100, cumulative=True, normed=True, edgecolor='none')
plt.xlabel('# discovery chances')
plt.ylabel('Fraction of NEOs')
plt.xlim(0, 20)
plt.ylim(0, 1)
plt.title('NEOs with X or more discovery chances @ H=%.1f' %(Hmark))

# fraction of objects with only X discovery chances
plt.figure(figsize=(10,6))
b = allBundles['neos']['Discoveries']
Hrange = b.slicer.Hrange
nchances = np.arange(1, 2, 1)
mVals = b.metricValues.swapaxes(0, 1)
for n in nchances:
    fraction = np.zeros(len(Hrange), float)
    for i, (H, mVal) in enumerate(zip(Hrange, mVals)):  
        match = np.where(mVal>0)
        n_available = float(len(match[0]))
        fraction[i] = len(np.where(mVal[match]<=n)[0]) / n_available
    plt.plot(Hrange, fraction, label='N chances %d' %(n))
plt.legend(loc='upper left', fancybox=True)
#plt.ylim(0, 5)
plt.xlim(15, Hrange.max())
plt.xlabel('H mag')
plt.ylabel('Fraction of NEOs @H')
plt.title('NEOs with <= N chances of discovery')



