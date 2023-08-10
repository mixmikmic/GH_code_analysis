get_ipython().magic('matplotlib inline')

from __future__ import print_function, division

import io
import os
import sys
import numpy as np
import numpy.ma as ma
from os.path import join as pjoin

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from scipy.stats import pearsonr, betai


from Utilities.config import ConfigParser
from Utilities.loadData import loadTrackFile
from Utilities.config import ConfigParser
from Utilities.track import Track
from Utilities.metutils import convert

import seaborn as sns
sns.set_context("notebook")
sns.set_style("whitegrid")

configstr = """
[DataProcess]
InputFile=C:/WorkSpace/data/TC/Allstorms.ibtracs_wmo.v03r06.csv
Source=IBTRACS
StartSeason=1961
FilterSeasons=True

[Region]
; Domain for windfield and hazard calculation
gridLimit={'xMin':90.,'xMax':180.,'yMin':-30.0,'yMax':-5.0}
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

trackFile = config.get('DataProcess', 'InputFile')
source = config.get('DataProcess', 'Source')

print("Track file: {0}".format(trackFile))
print("Track format: {0}".format(source))
alltracks = loadTrackFile(configstr, trackFile, source)
#tracks = parseTracks(configstr, trackFile, source, 1.)

gridLimit=config.geteval('Region', 'gridLimit')

tracks = [track for track in alltracks if track.inRegion(gridLimit)]

print("There are {0:d} tracks in the input dataset".format(len(tracks)))

def getIndices(elapsedTime, delta=6, maxtime=120):
    filter1 = (np.mod(elapsedTime, delta)==0)
    filter2 = (elapsedTime<=maxtime)
    idx = np.nonzero(filter1 & filter2)
    return idx

def corrcoef(matrix):
    rows, cols = matrix.shape[0], matrix.shape[1]
    r = np.ones(shape=(rows, rows))
    p = np.ones(shape=(rows, rows))
    for i in range(rows):
        for j in range(i+1, rows):
            r_, p_ = pearsonr(matrix[i], matrix[j])
            r[i, j] = r[j, i] = r_
            p[i, j] = p[j, i] = p_
    return r, p

ntracks = len(tracks)
ndays = 7
nhours = ndays * 24.
interval = 6
steps = range(interval, int(nhours + 1), interval)
prsteps = range(0, int(nhours + 1), interval)
alldp = ma.zeros((nhours/interval, ntracks))
allprs = ma.zeros((nhours/interval + 1, ntracks))
allspd = ma.zeros((nhours/interval + 1, ntracks))
allbear = ma.zeros((nhours/interval + 1, ntracks))
for i, track in enumerate(tracks):
    idx = getIndices(track.TimeElapsed, delta=interval, maxtime=nhours)
    tstep = track.TimeElapsed[idx]
    tstepidx = [steps.index(int(t)) for t in tstep if int(t)!=0]
    prtstepidx = [prsteps.index(int(t)) for t in tstep]


    dt = np.diff(track.TimeElapsed[idx]) 
    dp = np.diff(track.CentralPressure[idx])
    prs = track.CentralPressure[idx]
    spd = track.Speed[idx]
    bear = track.Bearing[idx]
    pmask = (track.CentralPressure[idx[0][1:]]!=0)
    prs = ma.array(prs, mask=(track.CentralPressure[idx[0]]!=0))
    spd = ma.array(spd)
    bear = ma.array(bear)
    dpdt = dp/interval
    dpmask = (np.abs(dpdt) > 10)

    dpdt = ma.array(dp/interval, mask=(pmask & dpmask))

    if len(dpdt)!=len(tstepidx):
        pass
    else:
        alldp[tstepidx, i] = dpdt
    allprs[prtstepidx, i] = prs
    allspd[prtstepidx, i] = spd
    allbear[prtstepidx, i] = bear
    
alldp[np.where(np.abs(alldp) > 10)] = ma.masked

fig, ax = plt.subplots(1, 1)
cm = ax.pcolor(alldp, cmap='RdBu', vmin=-5, vmax=5)
plt.colorbar(cm)

ax.set_yticklabels(range(0,int(nhours)+1, 24))

corr, dpp = corrcoef(alldp)
cov = np.cov(alldp)
pcor, pp = corrcoef(allprs)
pcov = np.cov(allprs)
scor, sp = corrcoef(allspd)
scov = np.cov(allspd)
bcor, bp = corrcoef(allbear)
bcov = np.cov(allbear)

fig = plt.figure(1, (18, 6))
grid = ImageGrid(fig, 111, nrows_ncols=(1,3), axes_pad=0.2,
                 add_all=True,cbar_location="top", cbar_mode="each",
                 cbar_size="5%", cbar_pad="2%")
im0 = grid[0].pcolor(np.arange(0, nhours, interval), np.arange(0, nhours, interval), cov, cmap='gray_r',)
grid[0].cax.colorbar(im0)
grid[0].set_xticks(np.arange(24, nhours + 1, 24))
grid[0].set_yticks(np.arange(24, nhours + 1, 24))
grid[0].autoscale()

im1 = grid[1].pcolor(np.arange(0, nhours, interval), np.arange(0, nhours, interval), corr, cmap='seismic', vmin=-1, vmax=1)
grid[1].cax.colorbar(im1)
grid[1].set_xticks(np.arange(24, nhours + 1, 24))
grid[1].autoscale()

im2 = grid[2].pcolor(np.arange(0, nhours, interval), np.arange(0, nhours, interval), dpp, cmap='gray_r', vmax=0.05)
grid[2].cax.colorbar(im2)
grid[2].set_xticks(np.arange(24, nhours + 1, 24))
grid[2].autoscale()

fig = plt.figure(1, (18, 6))
grid = ImageGrid(fig, 111, nrows_ncols=(1,3), axes_pad=0.2,
                 add_all=True,cbar_location="top", cbar_mode="each",
                 cbar_size="5%", cbar_pad="2%")
im0 = grid[0].pcolor(np.arange(0, nhours + 1, interval), 
                     np.arange(0, nhours + 1, interval), 
                     pcov[1:, 1:], cmap='gray_r',)
grid[0].cax.colorbar(im0)
grid[0].set_xticks(np.arange(24, nhours + 1, 24))
grid[0].set_yticks(np.arange(24, nhours + 1, 24))
grid[0].autoscale()
im1 = grid[1].pcolor(np.arange(0, nhours + 1, interval), 
                     np.arange(0, nhours + 1, interval), 
                     pcor[1:, 1:], cmap='seismic', vmin=-1, vmax=1)
grid[1].cax.colorbar(im1)
grid[1].set_xticks(np.arange(24, nhours + 1, 24))
grid[1].autoscale()

im2 = grid[2].pcolor(np.arange(0, nhours, interval), 
                     np.arange(0, nhours, interval), 
                     pp[1:, 1:], cmap='gray_r', vmax=0.05)
grid[2].cax.colorbar(im2)
grid[2].set_xticks(np.arange(24, nhours + 1, 24))
grid[2].autoscale()

fig = plt.figure(1, (18, 6))
grid = ImageGrid(fig, 111, nrows_ncols=(1,3), axes_pad=0.2,
                 add_all=True,cbar_location="top", cbar_mode="each",
                 cbar_size="5%", cbar_pad="2%")
im0 = grid[0].pcolor(np.arange(0, nhours+1, interval), 
                     np.arange(0, nhours+1, interval), 
                     scov[1:, 1:], cmap='gray_r',)
grid[0].cax.colorbar(im0)
grid[0].set_xticks(np.arange(24, nhours + 1, 24))
grid[0].set_yticks(np.arange(24, nhours + 1, 24))
grid[0].autoscale()
im1 = grid[1].pcolor(np.arange(0, nhours + 1, interval), 
                     np.arange(0, nhours + 1, interval), 
                     scor[1:, 1:], cmap='seismic', vmin=-1, vmax=1)
grid[1].cax.colorbar(im1)
grid[1].set_xticks(np.arange(24, nhours + 1, 24))
grid[1].autoscale()

im2 = grid[2].pcolor(np.arange(0, nhours+1, interval), 
                     np.arange(0, nhours+1, interval), 
                     sp[1:, 1:], cmap='gray_r', vmax=0.05)
grid[2].cax.colorbar(im2)
grid[2].set_xticks(np.arange(24, nhours + 1, 24))
grid[2].autoscale()

fig = plt.figure(1, (18, 6))
grid = ImageGrid(fig, 111, nrows_ncols=(1,3), axes_pad=0.2,
                 add_all=True,cbar_location="top", cbar_mode="each",
                 cbar_size="5%", cbar_pad="2%")
im0 = grid[0].pcolor(np.arange(0, nhours + 1, interval), 
                     np.arange(0, nhours + 1, interval), bcov[1:, 1:], cmap='gray_r',)
grid[0].cax.colorbar(im0)
grid[0].set_xticks(np.arange(24, nhours + 1, 24))
grid[0].set_yticks(np.arange(24, nhours + 1, 24))
grid[0].autoscale()
im1 = grid[1].pcolor(np.arange(0, nhours + 1, interval), 
                     np.arange(0, nhours + 1, interval), 
                     bcor[1:, 1:], cmap='seismic', vmin=-1, vmax=1)
grid[1].cax.colorbar(im1)
grid[1].set_xticks(np.arange(24, nhours + 1, 24))
grid[1].autoscale()

im2 = grid[2].pcolor(np.arange(0, nhours + 1, interval), 
                     np.arange(0, nhours + 1, interval), 
                     bp[1:, 1:], cmap='gray_r', vmax=0.05)
grid[2].cax.colorbar(im2)
grid[2].set_xticks(np.arange(24, nhours + 1, 24))
grid[2].autoscale()



