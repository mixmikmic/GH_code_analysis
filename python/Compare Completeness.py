import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import ProtoMakeObs as pmo

orbitfile = 'pha20141031.des'
outroot = orbitfile.replace('.des', '').replace('.txt', '')

orbits = pmo.readOrbits(orbitfile)

# Read observations files from the various runs.
obsfile = 'pha_withChip_allObs.txt' #enigma_1189/' + outroot + '_allObs.txt'
enigmaobs = pd.read_table(obsfile, sep='\s*', engine='python')
enigmaobs = enigmaobs.to_records()

obsfile = 'opsim4_152/' + outroot + '_allObs.txt'
opsim4152obs = pd.read_table(obsfile, sep='\s*', engine='python')
opsim4152obs = opsim4152obs.to_records()

obsfile = 'opsim3_61/' + outroot + '_allObs.txt'
opsim361obs = pd.read_table(obsfile, sep='\s*', engine='python')
opsim361obs = opsim361obs.to_records()

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

def integrateH(summaryVals, Hrange, Hindex=0.3):
    # Set expected H distribution. 
    # dndh = differential size distribution (number in this bin)
    dndh = np.power(10., Hindex*(Hrange-Hrange.min()))
    # dn = cumulative size distribution (number in this bin and brighter)
    dn = np.cumsum(dndh)
    intVals = np.cumsum(summaryVals*dndh)/dn
    return intVals

def metricVsH(metricVals, Hrange, npmethod=np.mean, label='Mean', fignum=None):
    fig = plt.figure(fignum)
    if npmethod is not None:
        vals = npmethod(metricVals, axis=0)
    else:
        vals = metricVals
    plt.plot(Hrange, vals, label=label)
    plt.xlabel('H (mag)')
    return fig.number

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

ssoID = '#objId' #!!ObjID'
ssoID_orbit = 'objId'
ssoids = orbits[ssoID_orbit]
if len(ssoids) != len(np.unique(orbits[ssoID_orbit])):
    print "Orbit id's are repeated!"

obs = enigmaobs
opsim = 'enigma_1189'

# Set up to run metrics over wider range of H.
Hrange = np.arange(13, 27.2, 0.5)
sNum = float(len(ssoids))

discoveries = np.zeros([len(ssoids), len(Hrange)], int)
discoveries30 = np.zeros([len(ssoids), len(Hrange)], int)
completeness = np.zeros([len(ssoids), len(Hrange)], int)

# Run Metrics.
for i, sso in enumerate(ssoids):
    ssoObs = obs[np.where(obs[ssoID] == sso)]
    orbit = orbits[np.where(orbits[ssoID_orbit] == sso)]
    discoveries[i] = DiscoveryMetric(ssoObs, Hrange=Hrange, tWindow=15)
    discoveries30[i] = DiscoveryMetric(ssoObs, Hrange=Hrange, tWindow=30)

completeness = Completeness(discoveries, Hrange, sNum)
completeness30 = Completeness(discoveries30, Hrange, sNum)

completeness_2 = Completeness(discoveries, Hrange, sNum, requiredChances=2)
completeness30_2 = Completeness(discoveries30, Hrange, sNum, requiredChances=2)

completeness_3 = Completeness(discoveries, Hrange, sNum, requiredChances=3)
completeness30_3 = Completeness(discoveries30, Hrange, sNum, requiredChances=3)

for i, c in enumerate([completeness, completeness_2, completeness_3]):
    for alpha in [0.3, 0.4, 0.5]:
        compH = integrateH(c, Hrange, Hindex=alpha)
        fignum = metricVsH(compH, Hrange, None)
        plt.ylabel('Completeness <=H')
        plt.title('%s: %s\n%s' %(opsim, outroot, 'Completeness (15 night window): %d chances' %(i+1)))
        #completeness at H<=22
        Hidx = np.where(Hrange==22.0)[0]
        comp22 = compH[Hidx]
        plt.axhline(comp22, color='r', linestyle=':')
        plt.axvline(22, color='r')
        plt.figtext(0.2, 0.2, 'Completeness H<=22 (alpha=%.1f): %.0f%s' %(alpha, comp22*100, '%'))
        print "Completeness at H<=22 with alpha=%.1f = %.1f%s (15 night window), requiring %d chances"  %(alpha, 
                                                                                            comp22*100, 
                                                                                            '%', i+1)

for i, c in enumerate([completeness30, completeness30_2, completeness30_3]):
    for alpha in [0.3, 0.4, 0.5]:
        compH = integrateH(c, Hrange, Hindex=alpha)
        fignum = metricVsH(compH, Hrange, None)
        plt.ylabel('Completeness <=H')
        plt.title('%s: %s\n%s' %(opsim, outroot, 'Completeness (30 night window): %d chances' %(i+1)))
        #completeness at H<=22
        Hidx = np.where(Hrange==22.0)[0]
        comp22 = compH[Hidx]
        plt.axhline(comp22, color='r', linestyle=':')
        plt.axvline(22, color='r')
        plt.figtext(0.2, 0.2, 'Completeness H<=22 (alpha=%.1f): %.0f%s' %(alpha, comp22*100, '%'))
        print "Completeness at H<=22 with alpha=%.1f = %.1f%s (30 night window), requiring %d chances"  %(alpha, 
                                                                                            comp22*100, 
                                                                                            '%', i+1)

obs = opsim4152obs
opsim = 'opsim4_152'

# Set up to run metrics over wider range of H.
Hrange = np.arange(13, 27.2, 0.5)
sNum = float(len(ssoids))

discoveries = np.zeros([len(ssoids), len(Hrange)], int)
discoveries30 = np.zeros([len(ssoids), len(Hrange)], int)
completeness = np.zeros([len(ssoids), len(Hrange)], int)

# Run Metrics.
for i, sso in enumerate(ssoids):
    ssoObs = obs[np.where(obs[ssoID] == sso)]
    orbit = orbits[np.where(orbits[ssoID_orbit] == sso)]
    discoveries[i] = DiscoveryMetric(ssoObs, Hrange=Hrange, window=15)
    discoveries30[i] = DiscoveryMetric(ssoObs, Hrange=Hrange, window=30)

completeness = Completeness(discoveries, Hrange, sNum)
completeness30 = Completeness(discoveries30, Hrange, sNum)

completeness_2 = Completeness(discoveries, Hrange, sNum, requiredChances=2)
completeness30_2 = Completeness(discoveries30, Hrange, sNum, requiredChances=2)

completeness_3 = Completeness(discoveries, Hrange, sNum, requiredChances=3)
completeness30_3 = Completeness(discoveries30, Hrange, sNum, requiredChances=3)

for i, c in enumerate([completeness, completeness_2, completeness_3]):
    for alpha in [0.3, 0.4, 0.5]:
        compH = integrateH(c, Hrange, Hindex=alpha)
        fignum = metricVsH(compH, Hrange, None)
        plt.ylabel('Completeness <=H')
        plt.title('%s: %s\n%s' %(opsim, outroot, 'Completeness (15 night window): %d chances' %(i+1)))
        #completeness at H<=22
        Hidx = np.where(Hrange==22.0)[0]
        comp22 = compH[Hidx]
        plt.axhline(comp22, color='r', linestyle=':')
        plt.axvline(22, color='r')
        plt.figtext(0.2, 0.2, 'Completeness H<=22 (alpha=%.1f): %.0f%s' %(alpha, comp22*100, '%'))
        print "Completeness at H<=22 with alpha=%.1f = %.1f%s (15 night window), requiring %d chances"  %(alpha, 
                                                                                            comp22*100, 
                                                                                            '%', i+1)

for i, c in enumerate([completeness30, completeness30_2, completeness30_3]):
    for alpha in [0.3, 0.4, 0.5]:
        compH = integrateH(c, Hrange, Hindex=alpha)
        fignum = metricVsH(compH, Hrange, None)
        plt.ylabel('Completeness <=H')
        plt.title('%s: %s\n%s' %(opsim, outroot, 'Completeness (30 night window): %d chances' %(i+1)))
        #completeness at H<=22
        Hidx = np.where(Hrange==22.0)[0]
        comp22 = compH[Hidx]
        plt.axhline(comp22, color='r', linestyle=':')
        plt.axvline(22, color='r')
        plt.figtext(0.2, 0.2, 'Completeness H<=22 (alpha=%.1f): %.0f%s' %(alpha, comp22*100, '%'))
        print "Completeness at H<=22 with alpha=%.1f = %.1f%s (30 night window), requiring %d chances"  %(alpha, 
                                                                                            comp22*100, 
                                                                                            '%', i+1)

condition = np.where(enigmaobs['dec'] < 0.046916) # exclude the NES
obs = enigmaobs[condition]
opsim = 'enigma_1189 no NES'

# Set up to run metrics over wider range of H.
Hrange = np.arange(13, 27.2, 0.5)
sNum = float(len(ssoids))

discoveries = np.zeros([len(ssoids), len(Hrange)], int)
discoveries30 = np.zeros([len(ssoids), len(Hrange)], int)
completeness = np.zeros([len(ssoids), len(Hrange)], int)

# Run Metrics.
for i, sso in enumerate(ssoids):
    ssoObs = obs[np.where(obs[ssoID] == sso)]
    orbit = orbits[np.where(orbits[ssoID_orbit] == sso)]
    discoveries[i] = DiscoveryMetric(ssoObs, Hrange=Hrange, window=15)
    discoveries30[i] = DiscoveryMetric(ssoObs, Hrange=Hrange, window=30)

completeness = Completeness(discoveries, Hrange, sNum)
completeness30 = Completeness(discoveries30, Hrange, sNum)

completeness_2 = Completeness(discoveries, Hrange, sNum, requiredChances=2)
completeness30_2 = Completeness(discoveries30, Hrange, sNum, requiredChances=2)

completeness_3 = Completeness(discoveries, Hrange, sNum, requiredChances=3)
completeness30_3 = Completeness(discoveries30, Hrange, sNum, requiredChances=3)

for i, c in enumerate([completeness, completeness_2, completeness_3]):
    for alpha in [0.3, 0.4, 0.5]:
        compH = integrateH(c, Hrange, Hindex=alpha)
        fignum = metricVsH(compH, Hrange, None)
        plt.ylabel('Completeness <=H')
        plt.title('%s: %s\n%s' %(opsim, outroot, 'Completeness (15 night window): %d chances' %(i+1)))
        #completeness at H<=22
        Hidx = np.where(Hrange==22.0)[0]
        comp22 = compH[Hidx]
        plt.axhline(comp22, color='r', linestyle=':')
        plt.axvline(22, color='r')
        plt.figtext(0.2, 0.2, 'Completeness H<=22 (alpha=%.1f): %.0f%s' %(alpha, comp22*100, '%'))
        print "Completeness at H<=22 with alpha=%.1f = %.1f%s (15 night window), requiring %d chances"  %(alpha, 
                                                                                            comp22*100, 
                                                                                            '%', i+1)

for i, c in enumerate([completeness30, completeness30_2, completeness30_3]):
    for alpha in [0.3, 0.4, 0.5]:
        compH = integrateH(c, Hrange, Hindex=alpha)
        fignum = metricVsH(compH, Hrange, None)
        plt.ylabel('Completeness <=H')
        plt.title('%s: %s\n%s' %(opsim, outroot, 'Completeness (30 night window): %d chances' %(i+1)))
        #completeness at H<=22
        Hidx = np.where(Hrange==22.0)[0]
        comp22 = compH[Hidx]
        plt.axhline(comp22, color='r', linestyle=':')
        plt.axvline(22, color='r')
        plt.figtext(0.2, 0.2, 'Completeness H<=22 (alpha=%.1f): %.0f%s' %(alpha, comp22*100, '%'))
        print "Completeness at H<=22 with alpha=%.1f = %.1f%s (30 night window), requiring %d chances"  %(alpha, 
                                                                                            comp22*100, 
                                                                                            '%', i+1)

obs = opsim361obs
opsim = 'opsim3_61'

# Set up to run metrics over wider range of H.
Hrange = np.arange(13, 27.2, 0.5)
sNum = float(len(ssoids))

discoveries = np.zeros([len(ssoids), len(Hrange)], int)
discoveries30 = np.zeros([len(ssoids), len(Hrange)], int)
completeness = np.zeros([len(ssoids), len(Hrange)], int)

# Run Metrics.
for i, sso in enumerate(ssoids):
    ssoObs = obs[np.where(obs[ssoID] == sso)]
    orbit = orbits[np.where(orbits[ssoID_orbit] == sso)]
    discoveries[i] = DiscoveryMetric(ssoObs, Hrange=Hrange, window=15)
    discoveries30[i] = DiscoveryMetric(ssoObs, Hrange=Hrange, window=30)

completeness = Completeness(discoveries, Hrange, sNum)
completeness30 = Completeness(discoveries30, Hrange, sNum)

completeness_2 = Completeness(discoveries, Hrange, sNum, requiredChances=2)
completeness30_2 = Completeness(discoveries30, Hrange, sNum, requiredChances=2)

completeness_3 = Completeness(discoveries, Hrange, sNum, requiredChances=3)
completeness30_3 = Completeness(discoveries30, Hrange, sNum, requiredChances=3)

for i, c in enumerate([completeness, completeness_2, completeness_3]):
    for alpha in [0.3, 0.4, 0.5]:
        compH = integrateH(c, Hrange, Hindex=alpha)
        fignum = metricVsH(compH, Hrange, None)
        plt.ylabel('Completeness <=H')
        plt.title('%s: %s\n%s' %(opsim, outroot, 'Completeness (15 night window): %d chances' %(i+1)))
        #completeness at H<=22
        Hidx = np.where(Hrange==22.0)[0]
        comp22 = compH[Hidx]
        plt.axhline(comp22, color='r', linestyle=':')
        plt.axvline(22, color='r')
        plt.figtext(0.2, 0.2, 'Completeness H<=22 (alpha=%.1f): %.0f%s' %(alpha, comp22*100, '%'))
        print "Completeness at H<=22 with alpha=%.1f = %.1f%s (15 night window), requiring %d chances"  %(alpha, 
                                                                                            comp22*100, 
                                                                                            '%', i+1)

for i, c in enumerate([completeness30, completeness30_2, completeness30_3]):
    for alpha in [0.3, 0.4, 0.5]:
        compH = integrateH(c, Hrange, Hindex=alpha)
        fignum = metricVsH(compH, Hrange, None)
        plt.ylabel('Completeness <=H')
        plt.title('%s: %s\n%s' %(opsim, outroot, 'Completeness (30 night window): %d chances' %(i+1)))
        #completeness at H<=22
        Hidx = np.where(Hrange==22.0)[0]
        comp22 = compH[Hidx]
        plt.axhline(comp22, color='r', linestyle=':')
        plt.axvline(22, color='r')
        plt.figtext(0.2, 0.2, 'Completeness H<=22 (alpha=%.1f): %.0f%s' %(alpha, comp22*100, '%'))
        print "Completeness at H<=22 with alpha=%.1f = %.1f%s (30 night window), requiring %d chances"  %(alpha, 
                                                                                            comp22*100, 
                                                                                            '%', i+1)

# Test enigma_1189 with a requirement for 3 or 4 visits per night!


obs = enigmaobs
opsim = 'enigma_1189'

# Set up to run metrics over wider range of H.
Hrange = np.arange(13, 27.2, 0.5)
sNum = float(len(ssoids))

discoveries = np.zeros([len(ssoids), len(Hrange)], int)
discoveries30 = np.zeros([len(ssoids), len(Hrange)], int)
completeness = np.zeros([len(ssoids), len(Hrange)], int)

# Run Metrics.
for i, sso in enumerate(ssoids):
    ssoObs = obs[np.where(obs[ssoID] == sso)]
    orbit = orbits[np.where(orbits[ssoID_orbit] == sso)]
    discoveries[i] = DiscoveryMetric(ssoObs, Hrange=Hrange, window=15, nObsPerNight=3)
    discoveries30[i] = DiscoveryMetric(ssoObs, Hrange=Hrange, window=30, nObsPerNight=3)

completeness = Completeness(discoveries, Hrange, sNum)
completeness30 = Completeness(discoveries30, Hrange, sNum)

for i, c in enumerate([completeness]):
    for alpha in [0.3, 0.4, 0.5]:
        compH = integrateH(c, Hrange, Hindex=alpha)
        fignum = metricVsH(compH, Hrange, None)
        plt.ylabel('Completeness <=H')
        plt.title('%s: %s\n%s' %(opsim, outroot, 'Completeness (15 night window): %d chances' %(i+1)))
        #completeness at H<=22
        Hidx = np.where(Hrange==22.0)[0]
        comp22 = compH[Hidx]
        plt.axhline(comp22, color='r', linestyle=':')
        plt.axvline(22, color='r')
        plt.figtext(0.2, 0.2, 'Completeness H<=22 (alpha=%.1f): %.0f%s' %(alpha, comp22*100, '%'))
        print "Completeness at H<=22 with alpha=%.1f = %.1f%s (15 night window), requiring %d chances"  %(alpha, 
                                                                                            comp22*100, 
                                                                                            '%', i+1)

for i, c in enumerate([completeness30]):
    for alpha in [0.3, 0.4, 0.5]:
        compH = integrateH(c, Hrange, Hindex=alpha)
        fignum = metricVsH(compH, Hrange, None)
        plt.ylabel('Completeness <=H')
        plt.title('%s: %s\n%s' %(opsim, outroot, 'Completeness (30 night window): %d chances' %(i+1)))
        #completeness at H<=22
        Hidx = np.where(Hrange==22.0)[0]
        comp22 = compH[Hidx]
        plt.axhline(comp22, color='r', linestyle=':')
        plt.axvline(22, color='r')
        plt.figtext(0.2, 0.2, 'Completeness H<=22 (alpha=%.1f): %.0f%s' %(alpha, comp22*100, '%'))
        print "Completeness at H<=22 with alpha=%.1f = %.1f%s (30 night window), requiring %d chances"  %(alpha, 
                                                                                            comp22*100, 
                                                                                            '%', i+1)

