get_ipython().magic('matplotlib inline')

from __future__ import print_function, division

import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
import seaborn as sns
import logging as log

import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.dates import HourLocator, DayLocator, DateFormatter

from datetime import datetime
from glob import glob
import pytz

import Utilities.metutils as metutils

sns.set_style("ticks")
sns.set_context("poster")

LTZ = pytz.timezone('Australia/Brisbane')
UTC = pytz.utc
def parseTime(yr, month, day, hour, minute):
    #LTZ = pytz.timezone(tz)
    #dt = LTZ.localize()
    dt =    datetime(int(yr), int(month), int(day), int(hour), int(minute))
    return dt

def parseMetarTime(dtstr, tmstr):
    dt = datetime.strptime("{0} {1}".format(dtstr, tmstr), "%Y%m%d %H%M")
    return UTC.localize(dt)

HALFHOURLY_DTYPE = [('hm', 'S2'), ('stnId', 'i'), ('dtLocalYear', 'i'), ('dtLocalMonth', 'i'), ('dtLocalDay', 'i'),
                    ('dtLocalHour', 'i'), ('dtLocalMinute', 'i'), ('dtStdYear', 'i'), ('dtStdMonth', 'i'), 
                    ('dtStdDay', 'i'), ('dtStdHour', 'i'), ('dtStdMinute', 'i'), ('WindSpeed', 'f8'), ('WindSpeedQual', 'S1'),
                    ('WindDir', 'f8'), ('WindDirQual', 'S1'), ('WindGust', 'f8'), ('WindGustQual', 'S1'), ('AWSFlag', 'S1'),
                    ('end', 'S1')]
HALFHOURLY_NAMES = [fields[0] for fields in HALFHOURLY_DTYPE]
CONVERTER = {'WindSpeed': lambda s: metutils.convert(float(s or 0), 'kmh', 'mps'),
             'WindGust': lambda s: metutils.convert(float(s or 0), 'kmh', 'mps')}

METAR_DTYPE = [('stnWMO', 'i'), ('stnCode', '|S4'), ('dtUTCDate', '|S8'), ('dtUTCTime', '|S4'),
               ('stnLat', 'f8'), ('stnLon', 'f8'), ('WindDir', 'f8'), ('WindSpeed', 'f8'),
               ('TempDry', 'f8'), ('DewPoint', 'f8'), ('MSLP', 'f8'), ('RF24hr', 'f8'), 
               ('RF10min', 'f8'), ('vis', 'f8'), ('Avis', 'f8'), ('WindGust', 'f8')]
METAR_NAMES = [fields[0] for fields in METAR_DTYPE]


TCRM_DTYPE = [('stnWMO', 'i'), ('dtUTCDatetime', 'S16'), ('stnLon', 'f8'), ('stnLat', 'f8'),
              ('tcWindGust', 'f8'), ('tcUwind', 'f8'), ('tcVwind', 'f8'), ('tcWindDir', 'f8'),
              ('tcMSLPressure', 'f8')]
TCRM_NAMES = [fields[0] for fields in TCRM_DTYPE]
TCRM_CONVERTER = {'dtUTCDatetime': lambda s: UTC.localize(datetime.strptime(s, "%Y-%m-%d %H:%M"))}

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

obs_path = "N:/climate_change/CHARS/B_Wind/data/raw/obs/halfhourly/"
metar_path = "N:/climate_change/CHARS/B_Wind/data/derived/obs/metar/"
tcrm_path = "N:/climate_change/CHARS/B_Wind/data/derived/tc/events/{0}/process/timeseries/"

obsbasename = "HM01X_Data_{0:06d}_999999997960860.txt"
metarbasename = "{0}-*.csv"
tcrmbase = "ts.{0:d}.csv"
stnId = 200840
stnWMO = 94283
stnName = "Cooktown AMO"
tcName = "Ita"
tcId = "bsh232014"

obsfname = pjoin(obs_path, obsbasename.format(int(stnId)))
if os.path.exists(obsfname):
    df = pd.read_csv(obsfname, skipinitialspace=True, skiprows=1, names=HALFHOURLY_NAMES, 
                     parse_dates={'dtLocalDatetime':['dtLocalYear', 'dtLocalMonth', 
                                                      'dtLocalDay', 'dtLocalHour', 'dtLocalMinute']}, 
                     date_parser=parseTime, index_col=False, converters=CONVERTER)
obsdata = df.to_records()

tcrmfname = pjoin(tcrm_path.format(tcId), tcrmbase.format(int(stnWMO)))

if os.path.exists(tcrmfname):
    df = pd.read_csv(tcrmfname, skipinitialspace=True, skiprows=1, names=TCRM_NAMES,
                     index_col=False, converters=TCRM_CONVERTER)
tcrmdata = df.to_records()

dtint = np.sort(intersect(tcrmdata['dtUTCDatetime'], obsdata['dtLocalDatetime']))
obsidx = [list(obsdata['dtLocalDatetime']).index(dt) for dt in dtint]
tcrmidx = [list(tcrmdata['dtUTCDatetime']).index(dt) for dt in dtint]

tcrmdt = tcrmdata['dtUTCDatetime'][tcrmidx]
tcrmws = tcrmdata['tcWindGust'][tcrmidx]
tcrmdir = tcrmdata['tcWindDir'][tcrmidx]

obsdt = obsdata['dtLocalDatetime'][obsidx]
obsws = obsdata['WindGust'][obsidx]
obsdir = obsdata['WindDir'][obsidx]

dayLocator = DayLocator()
hourLocator = HourLocator(interval=12)
dateFormat = DateFormatter('\n%Y-%m-%d')
hourFormat = DateFormatter('%H:%MZ')
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.scatter(obsdt, obsws * 1.093, label="Observations", color='r')
ax1.plot(tcrmdt, tcrmws, label="Simulation")
ax1.set_xlim((tcrmdt[0], tcrmdt[-1]))
#ax1.set_xlabel("Date")


ax1.set_title("{0} observations for TC {1}".format(stnName, tcName))
ax1.set_ylabel("Wind gust (m/s)")
ax1.set_ylim((0,80))
ax1.legend(loc=2)

ax2.scatter(obsdt, obsdir, label="Observations", color='r')
ax2.plot(tcrmdt, tcrmdir, label="Simulation")
ax2.set_xlim((tcrmdt[0], tcrmdt[-1]))
ax2.set_ylim((0, 360))
ax2.set_yticks(np.arange(0, 361, 45))
ax2.set_ylabel("Wind direction")
ax2.set_xlabel("Date")

ax2.xaxis.set_minor_locator(hourLocator)
ax2.xaxis.set_major_locator(dayLocator)
ax2.xaxis.set_major_formatter(dateFormat)
ax2.xaxis.set_minor_formatter(hourFormat)

fig, ax = plt.subplots(1, 2)
ax[0].scatter(obsws.compress(obsws!=0)* 1.093, tcrmws.compress(obsws!=0))
ax[0].set_xlabel("Observations")
ax[0].set_ylabel("Model")
ax[0].plot([0,60], [0,60], linestyle='--', color='k')
ax[0].set_ylim((0, 60))
ax[0].set_xlim((0, 60))
ax[0].set_aspect(1)

ax[1].scatter(obsdir.compress(obsws!=0), tcrmdir.compress(obsws!=0))
ax[1].set_xlabel("Observations")
ax[1].set_ylabel("Model")
ax[1].plot([0,360], [0,360], linestyle='--', color='k')
ax[1].set_ylim((0, 360))
ax[1].set_xlim((0, 360))
ax[1].set_aspect(1)
fig.set_tight_layout(2)

n = len(tcrmws.compress(obsws!=0))
rmse = np.linalg.norm(tcrmws.compress(obsws!=0) - obsws.compress(obsws!=0)* 1.093) / np.sqrt(n)
print(rmse)

mae = np.sum(np.absolute((tcrmws.compress(obsws!=0) - obsws.compress(obsws!=0)* 1.093))) / n
print(mae)

bias = np.mean(tcrmws.compress(obsws!=0) - obsws.compress(obsws!=0)* 1.093)
print(bias)

scatter = rmse/np.mean(obsws.compress(obsws!=0)* 1.093)
print(scatter)

n = len(tcrmws.compress(obsws>10))
rmse = np.linalg.norm(tcrmws.compress(obsws>10) - obsws.compress(obsws>10)* 1.093) / np.sqrt(n)
print(rmse)

mae = np.sum(np.absolute((tcrmws.compress(obsws>10) - obsws.compress(obsws>10)* 1.093))) / n
print(mae)

bias = np.mean(tcrmws.compress(obsws>10) - obsws.compress(obsws>10)* 1.093)
print(bias)

scatter = rmse/np.mean(obsws.compress(obsws>10)* 1.093)
print(scatter)

metarfname = pjoin(metar_path, metarbasename.format(stnWMO))
from glob import glob
metarfiles = glob(metarfname)
print(metarfiles)
if os.path.exists(metarfiles[0]):
    df = pd.read_csv(metarfiles[0], skipinitialspace=True, skiprows=1, names=METAR_NAMES, 
                     parse_dates={'dtUTCDatetime':['dtUTCDate', 'dtUTCTime']}, 
                     date_parser=parseMetarTime, index_col=False)
metardata = df.to_records()

dtint = np.sort(intersect(tcrmdata['dtUTCDatetime'], metardata['dtUTCDatetime']))
metaridx = [list(metardata['dtUTCDatetime']).index(dt) for dt in dtint]
tcrmidx = [list(tcrmdata['dtUTCDatetime']).index(dt) for dt in dtint]

tcrmdt = tcrmdata['dtUTCDatetime'][tcrmidx]
tcrmws = tcrmdata['tcWindGust'][tcrmidx]
tcrmdir = tcrmdata['tcWindDir'][tcrmidx]

metardt = metardata['dtUTCDatetime'][metaridx]
metarws = metardata['WindGust'][metaridx]
metardir = metardata['WindDir'][metaridx]

dayLocator = DayLocator()
hourLocator = HourLocator(interval=12)
dateFormat = DateFormatter('\n%Y-%m-%d')
hourFormat = DateFormatter('%H:%MZ')
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.scatter(metardt, metarws * 1.093, label="Observations", c='r')
ax1.plot(tcrmdt, tcrmws, label="Simulation")
ax1.set_xlim((tcrmdt[0], tcrmdt[-1]))
#ax1.set_xlabel("Date")


ax1.set_title("{0} observations for TC {1}".format(stnName, tcName))
ax1.set_ylabel("Wind gust (m/s)")
ax1.set_ylim((0,80))
ax1.legend(loc=2)

ax2.scatter(metardt, metardir, label="Observations", c='r')
ax2.plot(tcrmdt, tcrmdir, label="Simulation")
ax2.set_xlim((tcrmdt[0], tcrmdt[-1]))
ax2.set_ylim((0, 360))
ax2.set_yticks(np.arange(0, 361, 45))
ax2.set_ylabel("Wind direction")
ax2.set_xlabel("Date")

ax2.xaxis.set_minor_locator(hourLocator)
ax2.xaxis.set_major_locator(dayLocator)
ax2.xaxis.set_major_formatter(dateFormat)
ax2.xaxis.set_minor_formatter(hourFormat)

fig, ax = plt.subplots(1, 2)
ax[0].scatter(metarws.compress(metarws!=0)* 1.093, tcrmws.compress(metarws!=0), 
              c=metarws.compress(metarws!=0)* 1.093, cmap=cm.get_cmap('hot'))
ax[0].set_xlabel("Observations")
ax[0].set_ylabel("Model")
ax[0].plot([0,60], [0,60], linestyle='--', color='k')
ax[0].set_ylim((0, 60))
ax[0].set_xlim((0, 60))
ax[0].set_aspect(1)
ax[1].scatter(metardir.compress(metarws!=0), tcrmdir.compress(metarws!=0), 
              c=metarws.compress(metarws!=0)* 1.093, cmap=cm.get_cmap('hot'))
ax[1].set_xlabel("Observations")
ax[1].set_ylabel("Model")
ax[1].plot([0,360], [0,360], linestyle='--', color='k')
ax[1].set_ylim((0, 360))
ax[1].set_xlim((0, 360))
ax[1].set_aspect(1)
fig.set_tight_layout(2)

n = len(tcrmws.compress(metarws!=0))
rmse = np.linalg.norm(tcrmws.compress(metarws!=0) - metarws.compress(metarws!=0)* 1.093) / np.sqrt(n)
print(rmse)

mae = np.sum(np.absolute((tcrmws.compress(metarws!=0) - metarws.compress(metarws!=0)* 1.093))) / n
print(mae)

bias = np.mean(tcrmws.compress(metarws!=0) - metarws.compress(metarws!=0)* 1.093)
print(bias)

scatter = rmse/np.mean(metarws.compress(metarws!=0)* 1.093)
print(scatter)

n = len(tcrmws.compress(metarws>10))
rmse = np.linalg.norm(tcrmws.compress(metarws>10) - metarws.compress(metarws>10)* 1.093) / np.sqrt(n)
print(rmse)

mae = np.sum(np.absolute((tcrmws.compress(metarws>10) - metarws.compress(metarws>10)* 1.093))) / n
print(mae)

bias = np.mean(tcrmws.compress(metarws>10) - metarws.compress(metarws>10)* 1.093)
print(bias)

scatter = rmse/np.mean(metarws.compress(metarws>10)* 1.093)
print(scatter)

eventfile = "C:/WorkSpace/obs/tcevents.csv"
eventdf = pd.read_csv(eventfile, index_col=False)
print(len(eventdf))

for rec in eventdf.itertuples():
    idx, tcName, tcId, stnName, stnTZ, stnType, stnId, stnWMO = rec
    print(tcName, tcId, stnName, stnTZ)

