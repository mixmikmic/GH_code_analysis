import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd

# Import the readOrbits function, etc from ProtoMakeObs
from ProtoMakeObs import readOrbits

#orbitfile = 'mbas_1000.txt'
#orbitfile = 'tnos_1000.txt'
orbitfile = 'pha20141031.des'
outroot = orbitfile.replace('.des', '').replace('.txt', '')

obsfile = outroot + '_allObs.txt'
obsfile = 'pha_withChip_allObs.txt'

orbits = readOrbits(orbitfile)

obs = pd.read_table(obsfile, sep='\s*', engine='python')
newcols = obs.columns.values
newcols[0] = 'objId'
obs.columns = newcols
obs = obs.to_records()

def visibility(mag, mag5):
    # have to calculate SNR for each object
    # then convert this to 'observed magnitude' (adding errors)
    # then calculate likelihood of being seen at that 'observed magnitude'
    # then return yes/no
    pass

import os
from lsst.sims.photUtils import Sed, Bandpass, SignalToNoise, PhotometricParameters

fdir = os.getenv('LSST_THROUGHPUTS_DEFAULT')
lsst = {}
photParams = {}
for f in ('u', 'g', 'r', 'i', 'z', 'y'):
    lsst[f] = Bandpass()
    lsst[f].readThroughput(os.path.join(fdir, 'total_' + f + '.dat'))
    photParams[f] = PhotometricParameters(bandpass=f)

for f in ('u', 'g', 'r', 'i', 'z', 'y'):
    # Calculate gamma for a range of m5 values
    print 'Gamma values for %s' %f
    m5s = np.arange(20, 26, 0.5)
    gammas = np.zeros(len(m5s), float)
    for i, m5 in enumerate(m5s):
        gammas[i] = SignalToNoise.calcGamma(lsst[f], m5, photParams[f])
    print 'Min', gammas.min(), 'Max', gammas.max()
print "Let's take 0.038 as an average for now"
gamma = 0.038

def nObsMetric(ssoObs, magH=20., Hrange=np.arange(13, 27, 0.25), snrLimit=5):
    # Given the observations for a particular object and the opsim metadata:
    # Return the number of observations above SNR (in any band) as a function of H
    gamma = 0.038
    countObs = np.zeros(len(Hrange), int)
    if len(ssoObs) == 0:
        return countObs
    # Calculate the magnitude of this object in this filter, each H magnitude. 
    for i, H in enumerate(Hrange):
        magObs = H - magH + ssoObs['magV'] + ssoObs['dmagColor']
        magLimitWithTrailing = ssoObs['fiveSigmaDepth'] - ssoObs['dmagTrailing']
        # Calculate snr with approximate gamma
        xval = np.power(10, 0.4*(magObs-magLimitWithTrailing))
        snr = 1.0/np.sqrt((0.04-gamma)*xval + gamma*xval*xval)
        vis = np.where(snr>=snrLimit)[0]
        countObs[i] = np.where(snr >= snrLimit)[0].size
    return countObs

def DiscoveryMetric(ssoObs, magH=20., Hrange=np.arange(13, 27, 0.25), snrLimit=5, 
                    nObsPerNight=2, tNight=90.*60.0, nNightsPerWindow=3, tWindow=15):
    # Given the observations for a particular object and the opsim metadata (join using joinObs)
    # Return the number possibilities for 'discovery' of the object, as a function of H
    # nObsPerNight = number of observations per night required for tracklet
    # tNight = max time start/finish for the tracklet (seconds)
    # nNightsPerWindow = number of nights with observations required for track
    # tWindow = max number of nights in track (days)
    gamma = 0.038
    eps = 1e-10
    discoveryChances = np.zeros(len(Hrange), int)
    if len(ssoObs) == 0:
        return discoveryChances
    # Calculate the magnitude of this object in this filter, each H magnitude. 
    for i, H in enumerate(Hrange):
        magObs = H - magH + ssoObs['magV'] + ssoObs['dmagColor']
        magLimitWithTrailing = ssoObs['fiveSigmaDepth'] - ssoObs['dmagDetect']
        # Calculate SNR, including approximate 'gamma' 
        xval = np.power(10, 0.4*(magObs-magLimitWithTrailing))
        snr = 1.0/np.sqrt((0.04-gamma)*xval + gamma*xval*xval)
        vis = np.where(snr>=snrLimit)[0]

        if len(vis) == 0:
            discoveryChances[i] = 0
        else:
            # Now to identify where observations meet the timing requirements.
            #  Identify visits where the 'night' changes. 
            visSort = np.argsort(ssoObs['night'])[vis]
            n = np.unique(ssoObs['night'][visSort])
            # Identify all the indexes where the night changes (swap from one night to next)
            nIdx = np.searchsorted(ssoObs['night'][visSort], n)
            # Add index pointing to last observation.
            nIdx = np.concatenate([nIdx, np.array([len(visSort)-1])])
            # Find the nights & indexes where there were more than nObsPerNight observations.
            obsPerNight = (nIdx - np.roll(nIdx, 1))[1:]
            nWithXObs = n[np.where(obsPerNight >= nObsPerNight)]
            nIdxMany = np.searchsorted(ssoObs['night'][visSort], nWithXObs)
            nIdxManyEnd = np.searchsorted(ssoObs['night'][visSort], nWithXObs, side='right') - 1
            # Check that nObsPerNight observations are within tNight
            timesStart = ssoObs['expMJD'][visSort][nIdxMany]
            timesEnd = ssoObs['expMJD'][visSort][nIdxManyEnd]
            # Identify the nights where the total time interval may exceed tNight 
            # (but still have a subset of nObsPerNight which are within tNight)
            check = np.where((timesEnd - timesStart > tNight) & (nIdxManyEnd + 1 - nIdxMany > nObsPerNight))[0]
            bad = []
            for i, j, c in zip(visSort[nIdxMany][check], visSort[nIdxManyEnd][check], check):
                t = ssoObs['expMJD'][i:j+1]
                dtimes = (np.roll(t, 1-nObsPerNight) - t)[:-1]
                if np.all(dtimes > tnight+eps):
                    bad.append(c)
            goodIdx = np.delete(visSort[nIdxMany], bad)
            # Now (with indexes of start of 'good' nights with nObsPerNight within tNight),
            # look at the intervals between 'good' nights (for tracks)
            if len(goodIdx) < nNightsPerWindow:
                discoveryChances[i] = 0
            else:
                dnights = (np.roll(ssoObs['night'][goodIdx], 1-nNightsPerWindow) - ssoObs['night'][goodIdx])
                discoveryChances[i] = len(np.where((dnights >= 0) & (dnights <= tWindow))[0])
    return discoveryChances

def Completeness(discoveryChances, Hrange, numSsos, requiredChances=1):
    completeness = np.zeros(len(Hrange), float)
    discoveries = discoveryChances.swapaxes(0, 1)
    for i, H in enumerate(Hrange):
        completeness[i] = np.where(discoveries[i] >= requiredChances)[0].size
    completeness = completeness/float(numSsos)
    return completeness

def metricVsH(metricVals, Hrange, npmethod=np.mean, label='Mean', fignum=None):
    fig = plt.figure(fignum)
    if npmethod is not None:
        vals = npmethod(metricVals, axis=0)
    else:
        vals = metricVals
    plt.plot(Hrange, vals, label=label)
    plt.xlabel('H (mag)')
    return fig.number

def PeriodCoverageMetric(ssoObs, magH=20., Hrange=np.arange(13, 27, 0.25), snrLimit=5,
                       periods=np.arange(2.0, 12.0, 0.5)):
    # Given the observations for a particular object and the opsim metadata:
    # Return the number of observations above SNR (in any band) as a function of H
    phaseGaps = np.ones([len(Hrange), len(periods)], float)
    if len(ssoObs) == 0:
        return phaseGaps
    # Calculate the magnitude of this object in this filter, each H magnitude. 
    for i, H in enumerate(Hrange):
        magObs = H - magH + ssoObs['magV'] + ssoObs['dmagColor']
        magLimitWithTrailing = ssoObs['fiveSigmaDepth'] - ssoObs['dmagTrailing']
        snr = 5.0 * np.power(10., 0.4*(magLimitWithTrailing - magObs))
        visObs = ssoObs[np.where(snr >= snrLimit)]
        for j, period in enumerate(periods):
            if len(visObs) <= 1:
                phaseGaps[i][j] = 1.0
                continue
            phases = (visObs['expMJD'] % period)/float(period)
            phases = np.sort(phases)
            gaps = np.diff(phases)
            start_to_end = np.array([1.0 - phases[-1] + phases[0]], float)
            gaps = np.concatenate([gaps, start_to_end])
            maxGap = np.max(gaps)
            phaseGaps[i][j] = maxGap
    return phaseGaps    

def ActivityOverTimeMetric(ssoObs, magH=20., Hrange=np.arange(13, 27, 0.25), snrLimit=5, window=30.0):
    # For cometary activity, expect activity at the same point in its orbit at the same time, mostly
    # For collisions, expect activity at random times
    windowBins = np.arange(0, 10.0*365 + window/2.0, window)
    nWindows = len(windowBins)
    activityWindows = np.zeros(len(Hrange), int)
    if len(ssoObs) == 0:
        return activityWindows
    # Calculate the magnitude of this object in this filter, each H magnitude. 
    for i, H in enumerate(Hrange):
        magObs = H - magH + ssoObs['magV'] + ssoObs['dmagColor']
        magLimitWithTrailing = ssoObs['fiveSigmaDepth'] - ssoObs['dmagTrailing']
        snr = 5.0 * np.power(10., 0.4*(magLimitWithTrailing - magObs))
        vis = np.where(snr>=snrLimit)[0]
        if len(vis) == 0:
            activityWindows[i] = 0
        else:
            n, b = np.histogram(ssoObs[vis]['night'], bins=windowBins)
            activityWindows[i] = np.where(n>0)[0].size
    return activityWindows[i] / float(nWindows)

def ActivityOverPeriodMetric(ssoObs, orbit, magH=20., Hrange=np.arange(13, 27, 0.25), snrLimit=5, nBins=10):
    # For cometary activity, expect activity at the same point in its orbit at the same time, mostly
    # For collisions, expect activity at random times
    period = np.power(orbit['q']/(1-orbit['e']), 3/2) * 365.25
    meanAnomaly = ((ssoObs['expMJD'] - orbit['t_p']) / period) % (2*np.pi)
    binsize = 2*np.pi / float(nBins)
    anomalyBins = np.arange(0, 2*np.pi + binsize/2.0, binsize)
    activityWindows = np.zeros(len(Hrange), int)
    if len(ssoObs) == 0:
        return activityWindows
    # Calculate the magnitude of this object in this filter, each H magnitude. 
    for i, H in enumerate(Hrange):
        magObs = H - magH + ssoObs['magV'] + ssoObs['dmagColor']
        magLimitWithTrailing = ssoObs['fiveSigmaDepth'] - ssoObs['dmagTrailing']
        snr = 5.0 * np.power(10., 0.4*(magLimitWithTrailing - magObs))
        vis = np.where(snr>=snrLimit)[0]
        if len(vis) == 0:
            activityWindows[i] = 0
        else:
            n, b = np.histogram(meanAnomaly[vis], bins=anomalyBins)
            activityWindows[i] = np.where(n>0)[0].size
    return activityWindows[i] / float(nBins)

ssoID = 'objId'
ssoID_orbit = ssoID
ssoids = orbits[ssoID_orbit]
if len(ssoids) != len(np.unique(orbits[ssoID_orbit])):
    print "Orbit id's are repeated!"

# Set up to run metrics over wider range of H.
Hrange = np.arange(13, 27.2, 0.5)
sNum = float(len(ssoids))

nobsSsos = np.zeros([len(ssoids), len(Hrange)], int)
discoveries = np.zeros([len(ssoids), len(Hrange)], int)
discoveries30 = np.zeros([len(ssoids), len(Hrange)], int)

# Run Metrics.
for i, sso in enumerate(ssoids):
    ssoObs = obs[np.where(obs[ssoID] == sso)]
    orbit = orbits[np.where(orbits[ssoID_orbit] == sso)]
    nobsSsos[i] = nObsMetric(ssoObs, Hrange=Hrange)
    discoveries[i] = DiscoveryMetric(ssoObs, Hrange=Hrange, tWindow=15)
    discoveries30[i] = DiscoveryMetric(ssoObs, Hrange=Hrange, tWindow=30)

completeness = Completeness(discoveries, Hrange, sNum)
completeness30 = Completeness(discoveries30, Hrange, sNum)

completeness_2 = Completeness(discoveries, Hrange, sNum, requiredChances=2)
completeness30_2 = Completeness(discoveries30, Hrange, sNum, requiredChances=2)

completeness_3 = Completeness(discoveries, Hrange, sNum, requiredChances=3)
completeness30_3 = Completeness(discoveries30, Hrange, sNum, requiredChances=3)

fignum = metricVsH(nobsSsos, Hrange)
fignum = metricVsH(nobsSsos, Hrange, np.median, label='Median', fignum=fignum)
fignum = metricVsH(nobsSsos, Hrange, np.max, label='Max', fignum=fignum)
plt.legend(fontsize='small', fancybox=True)
plt.ylabel('Number of observations')
plt.title('Enigma_1189: %s : %s' %(outroot, 'N obs'))

fignum = metricVsH(discoveries, Hrange)
fignum = metricVsH(discoveries, Hrange, np.min, label='Min', fignum=fignum)
fignum = metricVsH(discoveries, Hrange, np.max, label='Max', fignum=fignum)
plt.legend(fontsize='small', fancybox=True)
plt.ylabel('Discovery chances')
plt.title('Enigma_1189: %s : %s' %(outroot, 'N discovery chances'))

#fignum = metricVsH(discoveriesSimple, Hrange)
#fignum = metricVsH(discoveriesSimple, Hrange, np.min, label='Min', fignum=fignum)
#fignum = metricVsH(discoveriesSimple, Hrange, np.max, label='Max', fignum=fignum)
#plt.legend(fontsize='small', fancybox=True)
#plt.ylabel('Simple Discovery chances')
#plt.title('Enigma_1189: %s : %s' %(outroot, 'N discovery chances'))

fignum = metricVsH(completeness, Hrange, None)
plt.ylabel('Completeness')
plt.title('Enigma_1189: %s : %s' %(outroot, 'Completeness at H'))

fignum = metricVsH(completeness, Hrange, None)
plt.ylabel('Completeness')
plt.title('Enigma_1189: %s : %s' %(outroot, 'Completeness at H, 30 day window'))

# Set up to run activity metrics. 
Hrange = np.arange(10, 10.2, 0.5)
sNum = float(len(ssoids))

nobsSsos = np.zeros([len(ssoids), len(Hrange)], int)
discoveries = np.zeros([len(ssoids), len(Hrange)], int)
discoveriesSimple = np.zeros([len(ssoids), len(Hrange)], int)
completeness = np.zeros([len(ssoids), len(Hrange)], int)

periods = np.arange(2.0, 12.0, 0.5)
phaseGaps = np.zeros([len(ssoids), len(Hrange), len(periods)], float)

activityT = {}
activityP = {}
windows = [7, 14, 30, 60, 180]
nbins = [4, 6, 8, 10, 13, 16, 20]
for w in windows:
    activityT[w] = np.zeros([len(ssoids), len(Hrange)], float)
for n in nbins:
    activityP[n] = np.zeros([len(ssoids), len(Hrange)], float)
    
# Run Metrics.
for i, sso in enumerate(ssoids):
    ssoObs = obs[np.where(obs[ssoID] == sso)]
    orbit = orbits[np.where(orbits[ssoID_orbit] == sso)]
    #nobsSsos[i] = nObsMetric(ssoObs, Hrange=Hrange)
    #discoveries[i] = DiscoveryMetric(ssoObs, Hrange=Hrange, window=15)
    #discoveriesSimple[i] = SimplerPythonDiscoveryMetric(ssoObs, Hrange=Hrange, window=15)
    #phaseGaps[i] = PhaseCoverageMetric(ssoObs, Hrange=Hrange, periods=periods)
    for w in windows:
        activityT[w][i] = ActivityOverTimeMetric(ssoObs, Hrange=Hrange, window=w)
    for n in nbins:
        activityP[n][i] = ActivityOverPeriodMetric(ssoObs, orbit, Hrange=Hrange, nBins=n)
#completeness = Completeness(discoveries, Hrange, sNum)

# Plot the min/mean/max of the fraction of activity detection opportunities, over all objects
meanFraction = np.zeros(len(windows), float)
minFraction = np.zeros(len(windows), float)
maxFraction = np.zeros(len(windows), float)

for i, w in enumerate(windows):
    meanFraction[i] = np.mean(activityT[w].swapaxes(0, 1)[0])
    minFraction[i] = np.min(activityT[w].swapaxes(0, 1)[0])
    maxFraction[i] = np.max(activityT[w].swapaxes(0, 1)[0])

plt.figure()
plt.plot(windows, meanFraction, 'r', label='Mean')
plt.plot(windows, minFraction, 'b--', label='Min')
plt.plot(windows, maxFraction, 'g--', label='Max')
plt.xlabel('Length of activity (days)')
plt.ylabel('Fraction of survey length when activity detectable')
plt.title('Activity detection metric (time): %s' %(outroot))
plt.grid()


# Plot the min/mean/max of the fraction of activity detection opportunities, over all objects
meanFraction = np.zeros(len(nbins), float)
minFraction = np.zeros(len(nbins), float)
maxFraction = np.zeros(len(nbins), float)

for i, n in enumerate(nbins):
    meanFraction[i] = np.mean(activityP[n].swapaxes(0, 1)[0])
    minFraction[i] = np.min(activityP[n].swapaxes(0, 1)[0])
    maxFraction[i] = np.max(activityP[n].swapaxes(0, 1)[0])

plt.figure()
x = 360/np.array(nbins, float)
plt.plot(x, meanFraction, 'r', label='Mean')
plt.plot(x, minFraction, 'b--', label='Min')
plt.plot(x, maxFraction, 'g--', label='Max')
plt.xlabel('Length of activity/orbit coverage')
plt.ylabel('Fraction of orbit when activity detectable')
plt.title('Activity detection metric (period) : %s' %(outroot))
plt.grid()


# to do: repeat at a range of H mags

def integrateH(summaryVals, Hrange, Hindex=0.3):
    # Set expected H distribution. 
    # dndh = differential size distribution (number in this bin)
    dndh = np.power(10., Hindex*(Hrange-Hrange.min()))
    # dn = cumulative size distribution (number in this bin and brighter)
    dn = np.cumsum(dndh)
    intVals = np.cumsum(summaryVals*dndh)/dn
    return intVals

# Defaults.
for alpha in [0.3, 0.4, 0.5]:
    compH = integrateH(completeness, Hrange, Hindex=alpha)
    fignum = metricVsH(compH, Hrange, None)
    plt.ylabel('Completeness <=H')
    plt.title('enigma_1189: %s - %s' %(outroot, 'Completeness'))
    #completeness at H<=22
    Hidx = np.where(Hrange==22.0)[0]
    comp22 = compH[Hidx]
    plt.axhline(comp22, color='r', linestyle=':')
    plt.axvline(22, color='r')
    plt.figtext(0.2, 0.2, 'Completeness H<=22 (alpha=%.1f): %.0f%s' %(alpha, comp22*100, '%'))
    print "Completeness at H<=22 with alpha=%.1f = %.1f%s"  %(alpha, comp22*100, '%')

# Look at the difference with 30 night windows
for alpha in [0.3, 0.4, 0.5]:
    compH = integrateH(completeness30, Hrange, Hindex=alpha)
    fignum = metricVsH(compH, Hrange, None)
    plt.ylabel('Completeness <=H')
    plt.title('enigma_1189: %s - %s' %(outroot, 'Completeness (30 night window)'))
    #completeness at H<=22
    Hidx = np.where(Hrange==22.0)[0]
    comp22 = compH[Hidx]
    plt.axhline(comp22, color='r', linestyle=':')
    plt.axvline(22, color='r')
    plt.figtext(0.2, 0.2, 'Completeness H<=22 (alpha=%.1f): %.0f%s' %(alpha, comp22*100, '%'))
    print "Completeness at H<=22 with alpha=%.1f = %.1f%s (30 night window)"  %(alpha, comp22*100, '%')

# Increased requirement of discovery chances (in case we lose something we could have discovered)
alpha = 0.4
for i, c in enumerate([completeness, completeness_2, completeness_3]):
    compH = integrateH(c, Hrange, Hindex=alpha)
    fignum = metricVsH(compH, Hrange, None)
    plt.ylabel('Completeness <=H')
    plt.title('enigma_1189: %s - %s' %(outroot, 'Completeness %d chances' %(i+1)))
    #completeness at H<=22
    Hidx = np.where(Hrange==22.0)[0]
    comp22 = compH[Hidx]
    plt.axhline(comp22, color='r', linestyle=':')
    plt.axvline(22, color='r')
    plt.figtext(0.2, 0.2, 'Completeness H<=22 (alpha=%.1f): %.0f%s' %(alpha, comp22*100, '%'))
    print "Completeness at H<=22 with alpha=%.1f = %.1f%s, require %d chances"  %(alpha, comp22*100, '%', i+1)

# what if we require three or four visits per night? 
# Set up to run metrics over wider range of H.
Hrange = np.arange(13, 27.2, 0.5)
sNum = float(len(ssoids))

discoveries_4 = np.zeros([len(ssoids), len(Hrange)], int)
discoveries30_4 = np.zeros([len(ssoids), len(Hrange)], int)
    
# Run Metrics.
for i, sso in enumerate(ssoids):
    orbit = orbits[np.where(orbits[ssoID_orbit] == sso)]
    discoveries_4[i] = DiscoveryMetric(ssoObs, Hrange=Hrange, tWindow=15, nObsPerNight=3, tNight=120*60.)
    discoveries30_4[i] = DiscoveryMetric(ssoObs, Hrange=Hrange, tWindow=30, nObsPerNight=3, tNight=120*60.)
completeness_4 = Completeness(discoveries_4, Hrange, sNum)
completeness30_4 = Completeness(discoveries30_4, Hrange, sNum)

# Integrate over H.
for alpha in [0.3, 0.4, 0.5]:
    compH = integrateH(completeness_4, Hrange, Hindex=alpha)
    fignum = metricVsH(compH, Hrange, None)
    plt.ylabel('Completeness <=H')
    plt.title('enigma_1189: %s - %s' %(outroot, 'Completeness'))
    #completeness at H<=22
    Hidx = np.where(Hrange==22.0)[0]
    comp22 = compH[Hidx]
    plt.axhline(comp22, color='r', linestyle=':')
    plt.axvline(22, color='r')
    plt.figtext(0.2, 0.2, 'Completeness H<=22 (alpha=%.1f): %.0f%s' %(alpha, comp22*100, '%'))
    print "Completeness at H<=22 with alpha=%.1f = %.1f%s"  %(alpha, comp22*100, '%')

def metricVs2dOrbits(x, y, xlabel, ylabel, 
                     metricVals, units, Hval, Hrange,
                     xbins, ybins):
    xvals = x
    yvals = y
    nxbins = len(xbins)
    nybins = len(ybins)
    xbinsize = np.mean(np.diff(xbins))
    ybinsize = np.mean(np.diff(ybins))
    xmin = np.min(xbins)
    ymin = np.min(ybins)
    # Set up to calculate summary values at each x/y binpoint. 
    summaryVals = np.zeros((nybins, nxbins), dtype='float')
    summaryNums = np.zeros((nybins, nxbins), dtype='int')
    Hidx = np.where(Hrange == Hval)[0]
    # Metrics are evaluated in the order of the orbits. 
    for i, (xi, yi) in enumerate(zip(x, y)):
        xidx = np.min([int((xi - xmin)/xbinsize), nxbins-1])
        yidx = np.min([int((yi - ymin)/ybinsize), nybins-1])
        summaryVals[yidx][xidx] += metricVals[i][Hidx]
        summaryNums[yidx][xidx] += 1
    summaryVals = np.where(summaryNums != 0, summaryVals / summaryNums, 0)
    # Create 2D x/y arrays, to match 2D counts array.
    xi, yi = np.meshgrid(xbins, ybins)
    # Plot. 
    plt.figure()
    levels = np.arange(summaryVals.min(), summaryVals.max(), (summaryVals.max() - summaryVals.min())/200.0)
    levels = np.arange(0, 30, 1)
    plt.contourf(xi, yi, summaryVals, levels, extend='max', zorder=0)
    #plt.plot(orbits[xlabel], orbits[ylabel], 'k.', markersize=2, zorder=3)
    cbar = plt.colorbar()
    cbar.set_label(units)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

mag = Hrange[0]
abins = np.arange(0, 4.0, 4/100.0)
ebins = np.arange(0, 1, 0.01)
ibins = np.arange(0, 100, 1)
a = orbits['q'] / (1-orbits['e'])
metricVs2dOrbits(a, orbits['e'], 'a', 'e', discoveries, 'N Discovery Opps', mag, Hrange, abins, ebins)
metricVs2dOrbits(a, orbits['i'], 'a', 'i', discoveries, 'N Discovery Opps', mag, Hrange, abins, ibins)

mag = Hrange[0]
a = orbits['q'] / (1-orbits['e'])
metricVs2dOrbits(a, orbits['e'], 'a', 'e', discoveries30,  'N Discovery Opps', mag, Hrange, abins, ebins)
metricVs2dOrbits(a, orbits['i'], 'a', 'i', discoveries30, 'N Discovery Opps', mag, Hrange, abins, ibins)

mag = 22
abins = np.arange(0, 4.0, 4/100.0)
qbins = np.arange(orbits['q'].min(), orbits['q'].max(), (orbits['q'].max() - orbits['q'].min())/100.0)
ebins = np.arange(0, 1, 0.01)
ibins = np.arange(0, 100, 1)
a = orbits['q'] / (1-orbits['e'])
metricVs2dOrbits(orbits['q'], orbits['e'], 'q', 'e', discoveries, 'N Discovery Opps', mag, Hrange, qbins, ebins)
plt.title('Number of discovery opportunities, H=%.1f' %(mag))
metricVs2dOrbits(orbits['q'], orbits['i'], 'q', 'i', discoveries, 'N Discovery Opps', mag, Hrange, qbins, ibins)
plt.title('Number of discovery opportunities, H=%.1f' %(mag))

a = orbits['q'] / (1-orbits['e'])
df = np.zeros([len(ssoids), len(Hrange)], float)
df = np.where(discoveries > 1, 1, 0)
metricVs2dOrbits(orbits['q'], orbits['e'], 'q', 'e', df, 'Fraction Discovered', mag, Hrange, qbins, ebins)
plt.title('Fraction discovered in this bin, at H=%.1f' %(mag))
metricVs2dOrbits(orbits['q'], orbits['i'], 'q', 'i', df, 'Fraction Discovered', mag, Hrange, qbins, ibins)
plt.title('Fraction discovered in this bin, at H=%.1f' %(mag))

from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

def metricVs2dOrbitsPoints(orbits, xlabel, ylabel, metricVals, Hval, Hrange, 
                           foregroundPoints=True, backgroundPoints=True):
    x = orbits[xlabel]
    y = orbits[ylabel]    
    Hidx = np.where(Hrange == Hval)[0]
    plt.figure()
    colors = np.swapaxes(metricVals, 0, 1)[Hidx][0]
    vmin = np.max(1, colors.min())
    if backgroundPoints:
        condition = np.where(colors == 0)
        plt.plot(x[condition], y[condition], 'r.', markersize=4, alpha=0.5, zorder=3)
    if foregroundPoints:
        plt.scatter(x, y, c=colors, vmin=vmin, s=15, alpha=0.8, zorder=0)
        cb = plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

metricVs2dOrbitsPoints(orbits, 'q', 'e', nobsSsos, mag, Hrange, foregroundPoints=False)
metricVs2dOrbitsPoints(orbits, 'q', 'i', nobsSsos, mag, Hrange, foregroundPoints=False)



