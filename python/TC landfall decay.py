get_ipython().magic('matplotlib inline')

from __future__ import print_function

import sys
import numpy as np
import pandas as pd

from scipy.optimize import leastsq
import scipy.stats as stats
from scipy.stats import logistic as slogistic
from scipy.stats import norm as snorm
import io

from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.basemap import Basemap

import statsmodels.api as sm

from Utilities.loadData import loadTrackFile
from Utilities.config import ConfigParser
from Utilities.track import Track
from Utilities.metutils import convert

from Evaluate.interpolateTracks import parseTracks

import seaborn
seaborn.set_context("poster")
seaborn.set_style("whitegrid")

configstr = """
[DataProcess]
InputFile=C:/WorkSpace/data/TC/Allstorms.ibtracs_wmo.v03r06.csv
Source=IBTRACS
StartSeason=1981
FilterSeasons=True

[Region]
; Domain for windfield and hazard calculation
gridLimit={'xMin':110.,'xMax':155.,'yMin':-30.0,'yMax':-5.0}
;gridLimit={'xMin':250.,'xMax':340.,'yMin':10.0,'yMax':40.0}
gridSpace={'x':1.0,'y':1.0}
gridInc={'x':1.0,'y':0.5}

[Input]
landmask = C:/WorkSpace/tcrm/input/landmask.nc
mslpfile = C:/WorkSpace/data/MSLP/slp.day.ltm.nc
datasets = IBTRACS,LTMSLP

[IBTRACS]
; Input data file settings
url = ftp://eclipse.ncdc.noaa.gov/pub/ibtracs/v03r06/wmo/csv/Allstorms.ibtracs_wmo.v03r06.csv.gz
path = C:/WorkSpace/data/TC/
filename = Allstorms.ibtracs_wmo.v03r06.csv
columns = tcserialno,season,num,skip,skip,skip,date,skip,lat,lon,skip,pressure
fielddelimiter = ,
numberofheadinglines = 3
pressureunits = hPa
lengthunits = km
dateformat = %Y-%m-%d %H:%M:%S
speedunits = kph

[LTMSLP]
; MSLP climatology file settings
URL = ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.derived/surface/slp.day.1981-2010.ltm.nc
path = C:/WorkSpace/data/MSLP
filename = slp.day.ltm.nc
"""

config = ConfigParser()
config.readfp(io.BytesIO(configstr))

trackFile = config.get('DataProcess', 'InputFile')
source = config.get('DataProcess', 'Source')

print("Track file: {0}".format(trackFile))
print("Track format: {0}".format(source))
tracks = loadTrackFile(configstr, trackFile, source)
#tracks = parseTracks(configstr, trackFile, source, 1.)
print("There are {0:d} tracks in the input dataset".format(len(tracks)))

domain = config.geteval('Region', 'gridLimit')
mapkwargs = dict(llcrnrlon=domain['xMin'],
                 llcrnrlat=domain['yMin'],
                 urcrnrlon=domain['xMax'],
                 urcrnrlat=domain['yMax'],
                 resolution='c',
                 projection='cyl')
m = Basemap(**mapkwargs)

def processTrack(track, m):
    onland = np.zeros(len(track.Longitude))
    dp0 = 0
    v0 = 0
    for i, (lon, lat) in enumerate(zip(track.Longitude, 
                                      track.Latitude)):
        if m.is_land(lon, lat):
            onland[i] = 1

    dp = []
    dt = []
    v = []
    flag = 0
    for i in range(1, len(onland)):
        if (onland[i]==1) & (onland[i-1]==0) & (track.CentralPressure[i-1] > 0.0):
            # New landfall (with central pressure prior to landfall):
            t0 = track.TimeElapsed[i-1]
            dp0 = track.EnvPressure[i-1] - track.CentralPressure[i-1]
            v0 = track.Speed[i-1]
            lat0 = track.Latitude[i-1]
            flag = 1
        
        if (flag==1) & (track.CentralPressure[i] > 0.0):
            # Storm is on land and has a valid central pressure record:
            dp.append(track.EnvPressure[i] - track.CentralPressure[i])
            dt.append(track.TimeElapsed[i] - t0)
            v.append(track.Speed[i])
            flag = onland[i]

            if flag == 0:
                return dp0, dp, dt, v0, np.mean(np.array(v)), lat0
    if len(dp) > 1:
        return dp0, dp, dt, v0, np.mean(np.array(v)), lat0
    else:
        return None, None, None, None, None, None
        

def residuals(params, dp, dt, dp0):
    yfit = dp0 * np.exp(-(np.array(params) * dt))
    return dp - yfit

def minimise(dp, dt, dp0, alpha=0., beta=1.):
    plsq = leastsq(residuals, [alpha], args=(dp, dt, dp0))
    return plsq[0]

def plotEvent(ax, dt, dp, dp0, params):
    ax.scatter(dt, dp/dp0, marker='s')
    xt = ax.get_xticks()
    ax.set_xlabel("Time after landfall (hours)")
    ax.set_xticks(range(0,12*int(1+max(xt)/12.+1),12))
    ax.set_ylabel(r"$\frac{\Delta p_c(t)}{\Delta p_0}$ (hPa)")
    xm = np.linspace(0, 12*int(max(dt)/12.), 100)    
    ax.set_xlim((0,12*int(1+max(xt)/12.)))

    legtext = r"$\Delta p_c = \Delta p_0 \exp{(%f t)}$"%(-params[0])
    ym = np.exp(-params[0]*xm)
    ax.plot(xm, ym, label=legtext)
    (ymin, ymax) = ax.get_ylim()
    l = ax.legend(loc=0)
    ax.set_ylim((0,ymax))

landfall_pressure = []
landfall_speed = []
landfall_lat = []
decayrate = []
lftracks = []
nevents=0
fig1, (ax1, ax2) = plt.subplots(2,1,sharex=True)

for n, track in enumerate(tracks):
    if track.inRegion(domain):
        # Process the track to get the pressure at landfall and the decay rate thereafter:
        dp0, dp, dt, v0, v, lat0 = processTrack(track, m)
        if (dp0 is not None)  and (min(dp) > 0) and (dp0 >= max(dp)):
            nevents += 1
            lftracks.append(track)
            p = minimise(dp, dt, dp0, 0., 1.)
            ax1.plot(dt, dp)
            ax2.plot(dt, (dp/dp0))
            #fig0, ax = plt.subplots(1,1,sharex=True)
            #plotEvent(ax, dt, dp, dp0, p)
            #ax.set_title("Storm {0} ({1}) ".format(n, track.Year[0]))
            #plt.savefig("{0:03d}.png".format(n))

            decayrate.append(p[0])
            landfall_pressure.append(dp0)
            landfall_speed.append(v0)
            landfall_lat.append(lat0)
    
ax2.set_xlabel("Time after landfall (hours)")
xt = ax2.get_xticks()
ax2.set_xticks(range(0,12*int(1+max(xt)/12.),12))
ax1.set_ylabel("$\Delta p_c(t) $ (hPa)")
ax1.set_title("Pressure deficit decay rates ({0} events)".format(nevents))
ax2.set_ylabel(r"$\frac{\Delta p_c(t)}{\Delta p_0}$")

df = pd.DataFrame({'alpha':decayrate,
                   'dp0':landfall_pressure,
                   'v0':landfall_speed,
                   'lat0':landfall_lat})
jp = seaborn.jointplot('dp0','alpha',df, kind='reg', size=10,xlim=(0,140))
jp.set_axis_labels(r"${\Delta p_0}$", r"$\hat{\alpha}$")

X = sm.add_constant(landfall_pressure)
y = np.array(decayrate)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
print(results.params)
print(results.rsquared)
print(stats.mstats.normaltest(results.resid))
print(stats.kstest(results.resid, 'norm'))

rp = seaborn.residplot('dp0','alpha',df,lowess=False)
rp.set_xlabel(r"${\Delta p_0}$")
rp.set_xlim((0,120))
rp.set_ylabel(r"$\varepsilon$")
rp.set_title(r"Model residuals: $\sigma^2 = ${0:.4f}".format(np.std(results.resid)))

fig, (ax0, ax1) = plt.subplots(1,2, figsize=(14,6))

bins = np.arange(-0.1, 0.21, 0.01)
ax = seaborn.distplot(results.resid,bins=bins, ax=ax0, 
                      kde_kws={'label':'Residuals','linestyle':'--'}, 
                      norm_hist=True)

fp = stats.nct.fit(results.resid)
print("Fit parameters for the non-central Student's T distribution:")
print(fp)

x = np.linspace(-0.1, 0.2, 1000)



ax.set_ylabel("Count")
ax.set_xlabel(r"$\varepsilon$")
pp = sm.ProbPlot(results.resid, stats.nct, fit=True)
pp.qqplot('Non-central T', 'Residuals', line='45', ax=ax1, color='gray',alpha=0.5)
fig.tight_layout()
print(pp.fit_params)
ppfit = pp.fit_params

ax.plot(x, stats.nct.pdf(x,fp[0], fp[1], loc=fp[2], scale=fp[3]), label='Non-central T')
ax.legend(loc=0)

fig, (ax0, ax1) = plt.subplots(1,2, figsize=(14,6))

bins = np.arange(-0.1, 0.21, 0.01)
ax = seaborn.distplot(results.resid, bins=bins, ax=ax0, kde_kws={'label':'Residuals','linestyle':'--'})
n, b = np.histogram(results.resid, bins=bins, density=True)
fpnorm = stats.norm.fit(results.resid)#, floc=np.median(results.resid),scale=np.std(results.resid))
print("Fit parameters for the normal distribution:")
print(fpnorm)

x = np.linspace(-0.1, 0.2, 1000)
ax.plot(x, stats.norm.pdf(x, *fpnorm),label='Normal')
ax.legend(loc=0)


ax.set_ylabel("Count")
ax.set_xlabel(r"$\varepsilon$")
ppnorm = sm.ProbPlot(results.resid, stats.norm, fit=True)
ppnorm.qqplot('Normal', 'Residuals', line='45', ax=ax1, color='gray',alpha=0.5)
fig.tight_layout()

fig, (ax0, ax1) = plt.subplots(1,2, figsize=(14,6))

bins = np.arange(-0.1, 0.21, 0.01)
ax = seaborn.distplot(results.resid, bins=bins, ax=ax0, kde_kws={'label':'Residuals','linestyle':'--'})
n, b = np.histogram(results.resid, bins=bins, density=True)
fplog = stats.logistic.fit(results.resid)#, loc=np.median(results.resid), scale=np.std(results.resid))
print("Fit parameters for the logistic distribution:")
print(fplog)

x = np.linspace(-0.1, 0.2, 1000)
ax.plot(x, stats.logistic.pdf(x, *fplog),label='Logistic')
ax.legend(loc=0)


ax.set_ylabel("Count")
ax.set_xlabel(r"$\varepsilon$")
pplog = sm.ProbPlot(results.resid, stats.logistic, fit=True)
pplog.qqplot('Logistic', 'Residuals', line='45', ax=ax1, color='gray',alpha=0.5)
fig.tight_layout()

print(stats.kstest(results.resid, 'nct', args=pp.fit_params))
print(stats.kstest(results.resid, 'norm', args=ppnorm.fit_params))
print(stats.kstest(results.resid, 'logistic', args=pplog.fit_params))

r = (results.rsquared)
pr = results.params
p0 = np.random.choice(landfall_pressure, 178, replace=True)

pp = (pr[0] + pr[1]*p0) + stats.nct.rvs(fp[0], fp[1], loc=fp[2],
                                        scale=fp[3], size=178)

fig, ax1 = plt.subplots(1, 1)

seaborn.regplot('dp0', 'alpha', df, scatter_kws={'s':25}, label="Observed", ax=ax1)
ax1.set_xlim((0, 100))
ax1.set_ylim((-0.05, 0.20))
ax1.set_xlabel(r"${\Delta p_0}$ (hPa)")
ax1.set_ylabel(r"$\hat{\alpha}$")
ax1.axhline(0.0, color='k')

ax1.scatter(p0, pp, s=25, color='red', marker='s', 
            alpha=0.5, label="Modelled")
ax1.legend(loc=0)

seaborn.interactplot("dp0", 'v0', 'alpha', df, cmap=get_cmap("YlOrRd"))

X = np.column_stack((landfall_pressure, landfall_speed))
X = sm.add_constant(X)
y = np.array(decayrate)
var2model = sm.OLS(y, X)
var2results = var2model.fit()
print(var2results.summary())
print('Parameters: ', var2results.params)
print('R-squared: ', var2results.rsquared)
print('P-value: ', var2results.pvalues)

fp2d = stats.nct.fit(var2results.resid, loc=np.mean(var2results.resid),
                     scale=np.std(var2results.resid))
pr = var2results.params
p0 = np.random.choice(landfall_pressure, 55, replace=True)
v0 = np.random.choice(landfall_speed, 55, replace=True)

rpoints = pr[0] + pr[1]*p0 + pr[2]*v0 +             stats.nct.rvs(fp2d[0], fp2d[1], 
                          loc=fp2d[2], scale=fp2d[3],
                          size=55)

dp = np.arange(0, 101, 1)
vv = np.arange(0, 40.1, 0.5)

xx, yy = np.meshgrid(dp, vv)
pp = pr[0] + pr[1]*xx + pr[2]*yy
fig, ax1 = plt.subplots(1, 1)
CS = ax1.contourf(xx, yy, pp, 10,extend='both')
plt.colorbar(CS, extend='both', label='Decay rate (hPa/hr)')
ax1.set_xlim((0, 100))
ax1.set_ylim((0, 20))
ax1.set_xlabel(r"${\Delta p_0}$ (hPa)")
ax1.set_ylabel(r"$v_0$ ($\mathrm{m} \mathrm{s}^{-1}$)")
pal1 = seaborn.light_palette("Red", as_cmap=True)
pal2 = seaborn.light_palette("Blue", as_cmap=True)
ax1.scatter(landfall_pressure, landfall_speed, s=50, marker='s', 
            c=decayrate, cmap=pal1, label="Observed")
ax1.scatter(p0, v0, s=50, marker='o', c=rpoints, cmap=pal2,  
            label="Modelled")
ax1.legend(loc=0, frameon=True)
print(fp2d)
print(rpoints.min(), rpoints.max())

fig1, (ax0, ax1) = plt.subplots(1,2, figsize=(14,6))
bins = np.arange(-0.1, 0.21, 0.01)
ax = seaborn.distplot(var2results.resid, bins=bins, 
                      ax=ax0, kde_kws={'label':'Residuals','linestyle':'--'})
#n, b = np.histogram(var2results.resid, bins=bins, density=False)
fp2d = stats.nct.fit(var2results.resid)#, 
                     #loc=np.median(var2results.resid), 
                     #scale=np.std(var2results.resid))
print("Fit parameters for the non-central Student's T distribution:")
print(fp2d)
x = np.linspace(-0.1, 0.2, 1000)

ax.plot(x, stats.nct.pdf(x, fp2d[0], fp2d[1], 
                         loc=fp2d[2], scale=fp2d[3]), 
        label='Non-central T')
ax.legend(loc=0)


ax.set_ylabel("Count")
ax.set_xlabel(r"$\varepsilon$")
ax.set_ylim(0,25)
ax.set_xlim(-0.1, 0.2)
var2pp = sm.ProbPlot(var2results.resid, stats.nct, fit=True)
var2pp.qqplot('Non-central T', 'Residuals', 
              line='45', color='gray', alpha=0.5, ax=ax1)
print(var2pp.fit_params)

var2ppnorm = sm.ProbPlot(var2results.resid, stats.norm, fit=True)
var2pplog = sm.ProbPlot(var2results.resid, stats.logistic, fit=True)

print(stats.kstest(var2results.resid, 'nct', args=var2pp.fit_params))
print(stats.kstest(var2results.resid, 'norm', args=var2ppnorm.fit_params))
print(stats.kstest(var2results.resid, 'logistic', args=var2pplog.fit_params))

ax = seaborn.interactplot("dp0", 'lat0', 'alpha', df, cmap=get_cmap("YlOrRd"),)
ax.set_xlabel(r'$\Delta p_0$')
ax.set_ylabel('Landfall latitude')

X = np.column_stack((landfall_pressure, landfall_lat))
X = sm.add_constant(X)
y = np.array(decayrate)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
print('Parameters: ', results.params)
print('P-value: ', results.pvalues)
print('R-squared: ', results.rsquared)

from PlotInterface.tracks import saveTrackMap
from IPython.display import Image

startSeason = config.get("DataProcess", "StartSeason")
xx = np.arange(domain['xMin'], domain['xMax'] + 0.1, 0.1)
yy = np.arange(domain['yMin'], domain['yMax'] + 0.1, 0.1)

[xgrid, ygrid] = np.meshgrid(xx,yy)
title = "Landfalling TCs - {0} - 2013".format(startSeason)
mapkwargs = dict(llcrnrlon=domain['xMin']-10,
                 llcrnrlat=domain['yMin'],
                 urcrnrlon=domain['xMax'],
                 urcrnrlat=domain['yMax'],
                 resolution='f',
                 projection='merc')

saveTrackMap(lftracks, xgrid, ygrid, title, mapkwargs, "tracks.png")
Image("tracks.png")

