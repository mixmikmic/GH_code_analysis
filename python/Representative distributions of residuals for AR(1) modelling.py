get_ipython().magic('matplotlib inline')
from __future__ import print_function, division

import io
import os
import sys
from os.path import join as pjoin

import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm


from StatInterface.GenerateDistributions import GenerateDistributions
from Utilities.stats import maxCellNum
from Utilities.config import ConfigParser
from Utilities.metutils import convert
from Utilities.files import flLoadFile

sns.set_context("poster")
sns.set_style("ticks")

fig, (ax1) = plt.subplots(1, 1,sharey=True)
l = stats.logistic.rvs(size=1000)
n = stats.norm.rvs(size=1000)
lx = np.linspace(stats.logistic.ppf(0.005),
                 stats.logistic.ppf(0.995), 100)
ax1.plot(lx, stats.logistic.pdf(lx), 'r-', lw=2, label="Logistic")


nx = np.linspace(stats.norm.ppf(0.005),
                 stats.norm.ppf(0.995), 100)
ax1.plot(nx, stats.norm.pdf(nx), 'k-', lw=2, label="Normal")

ax1.set_xlim((-6, 6))
ax1.legend(loc=2)
ax1.grid()

configstr = """
[DataProcess]
InputFile=C:/WorkSpace/data/TC/Allstorms.ibtracs_wmo.v03r06.csv
Source=IBTRACS
StartSeason=1961
FilterSeasons=True

[Region]
; Domain for windfield and hazard calculation
gridLimit={'xMin':100.,'xMax':120.,'yMin':-20.0,'yMax':-5.0}
gridSpace={'x':1.0,'y':1.0}
gridInc={'x':1.0,'y':0.5}

[TrackGenerator]
NumSimulations=5000
YearsPerSimulation=10
SeasonSeed=68876543
TrackSeed=334825
TimeStep=1.0

[Input]
landmask = C:/WorkSpace/tcrm/input/landmask.nc
mslpfile = C:/WorkSpace/data/MSLP/slp.day.ltm.nc
datasets = IBTRACS,LTMSLP

[Output]
Path=C:/WorkSpace/data/TC/aus

[Hazard]
Years=2,5,10,20,25,50,100,200,250,500,1000,2000,2500,5000
MinimumRecords=10
CalculateCI=False

[Logging]
LogFile=C:/WorkSpace/data/TC/aus/log/aus.log
LogLevel=INFO
Verbose=False

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
filename = slp.day.ltm.nc"""

config = ConfigParser()
config.readfp(io.BytesIO(configstr))

outputPath = config.get('Output', 'Path')
processPath = pjoin(outputPath, 'process')


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
pRateData = flLoadFile(pjoin(processPath, 'pressure_rate'))
pAllData = flLoadFile(pjoin(processPath, 'all_pressure'))
bRateData = flLoadFile(pjoin(processPath, 'bearing_rate'))
bAllData = flLoadFile(pjoin(processPath, 'all_bearing'))
sRateData = flLoadFile(pjoin(processPath, 'speed_rate'))
sAllData = flLoadFile(pjoin(processPath, 'all_speed'))

d = pRateData.compress(pRateData < sys.maxint)
m = np.average(d)
sd = np.std(d)
nd = (d-m)/sd


ppn = sm.ProbPlot(nd, stats.norm)
ppl = sm.ProbPlot(nd, stats.logistic)
ppn.qqplot(xlabel="Model", ylabel="Observations", ax=ax1, line='45')
ppl.qqplot(xlabel="Model", ylabel="Observations", ax=ax2, line='45')

ax1.set_title("Normal distribution")
ax2.set_title("Logistic distribution")

ax2.set_xlim((-15, 15))
ax1.set_xlim((-15, 15))
fig.tight_layout()

d = bRateData.compress(bRateData < sys.maxint)
m = np.average(d)
sd = np.std(d)
nd = (d-m)/sd

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ppn = sm.ProbPlot(nd, stats.norm)
ppl = sm.ProbPlot(nd, stats.logistic)
ppn.qqplot(xlabel="Model", ylabel="Observations", ax=ax1, line='45')
ppl.qqplot(xlabel="Model", ylabel="Observations", ax=ax2, line='45')

ax1.set_title("Normal distribution")
ax2.set_title("Logistic distribution")

ax2.set_aspect('equal')
ax1.set_aspect('equal')
fig.tight_layout()

d = sRateData.compress(sRateData < sys.maxint)
m = np.average(d)
sd = np.std(d)
nd = (d-m)/sd

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ppn = sm.ProbPlot(nd, stats.norm)
ppl = sm.ProbPlot(nd, stats.logistic)
ppn.qqplot(xlabel="Model", ylabel="Observations", ax=ax1, line='45')
ppl.qqplot(xlabel="Model", ylabel="Observations", ax=ax2, line='45')

ax1.set_title("Normal distribution")
ax2.set_title("Logistic distribution")

ax2.set_aspect('equal')
ax1.set_aspect('equal')
fig.tight_layout()

gridLimit = config.geteval('Region', 'gridLimit')
gridSpace = config.geteval('Region', 'gridSpace')
gridInc = config.geteval('Region', 'gridInc')
kdeType = 'Gaussian'

ncells = maxCellNum(gridLimit, gridSpace)
rows = int(np.ceil(np.sqrt(ncells)))
cols = int(np.ceil(ncells / rows))
fig, axes = plt.subplots(cols, rows, sharex=True, sharey=True)
rvals = np.zeros(ncells)

gd = GenerateDistributions(io.BytesIO(configstr), gridLimit, gridSpace, gridInc, kdeType)
for n in xrange(0, ncells+1):
    gd.allDistributions()
    gd.extractParameter(n)
    d = gd.parameter - np.mean(gd.parameter)
    (osm, osr), (m, b, r)= stats.probplot(d, dist='norm', plot=axes[n])
    rvals[n] = r
    

rvals.reshape(cols, rows)
plt.pcolor(rvals)

