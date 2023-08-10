import sys
sys.path.append("C:/Users/ahaberlie/Documents/GitHub/MCS/")

import pickle
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from mcs.utils.mapping_help import get_NOWrad_conus_lon_lat
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from string import ascii_lowercase

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 30, 30

from_proj = ccrs.PlateCarree()
to_proj = ccrs.AlbersEqualArea(central_longitude=-100.0000, central_latitude=38.0000)

lons, lats = get_NOWrad_conus_lon_lat()

lons, lats = np.meshgrid(lons, lats)

splot = 1

year = 2015

track_occur_dir = "../data/track_data/occurrence/"

for crsr in [6, 12, 24, 48]:
    for ssr in [48, 96, 192]:
        
        ax = plt.subplot(4, 3, splot, projection=to_proj)
        ax.set_extent([-105, -75, 25, 48])
        shapename = 'admin_1_states_provinces_lakes_shp'
        states_shp = shpreader.natural_earth(resolution='50m',
                                             category='cultural', name=shapename)

        for state, info in zip(shpreader.Reader(states_shp).geometries(), 
                               shpreader.Reader(states_shp).records()):
            if info.attributes['admin'] == 'United States of America':

                ax.add_geometries([state], ccrs.PlateCarree(),
                                  facecolor='None', edgecolor='k')
        
        pdict = {'p0.00':None, 'p0.50': None, 'p0.90': None, 'p0.95': None}
        
        for p, ps in zip([0.0, 0.5, 0.95], ['0.00', '0.50', '0.95']):
            
            prom = pickle.load(open(track_occur_dir + str(year) + "/" + str(crsr).zfill(2) + "_"                                     + str(ssr) + "_p" + str(int(100*p)) + "_rematched_mjjas.pkl", "rb"))

            pdict["p" + ps] = prom

        cmap = plt.cm.Greys
        classes = list(range(0, 225, 40))
        norm = BoundaryNorm(classes, ncolors=cmap.N, clip=True)

        #Since each count is a 15 minute window, divide by 4 to get number of hours
        m50_ = gaussian_filter(pdict['p0.50']/4, 30)

        mmp = ax.pcolormesh(lons, lats, m50_, transform=from_proj, norm=norm, cmap=cmap)
        
        plt.colorbar(mmp, ax=ax, shrink=0.4, pad=0.01)

        m0_ = gaussian_filter(pdict['p0.00']/4, 30)
        m95_ = gaussian_filter(pdict['p0.95']/4, 30)

        l1 = ax.contour(lons, lats, m0_, levels=[40], colors=['k',], 
                          transform=from_proj, linestyles='dashed', linewidths=1)

        l3 = ax.contour(lons, lats, m95_, levels=[40], colors=['k',], 
                          transform=from_proj, linewidths=2)
        
        ax.set_title(ascii_lowercase[splot-1] + ".")
        
        splot += 1
plt.tight_layout()

import sys
sys.path.append("C:/Users/ahaberlie/Documents/GitHub/MCS/")

import pickle
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from mcs.utils.mapping_help import get_NOWrad_conus_lon_lat
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from string import ascii_lowercase

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 30, 30

from_proj = ccrs.PlateCarree()
to_proj = ccrs.AlbersEqualArea(central_longitude=-100.0000, central_latitude=38.0000)

lons, lats = get_NOWrad_conus_lon_lat()

lons, lats = np.meshgrid(lons, lats)

splot = 1

year = 2016

track_occur_dir = "../data/track_data/occurrence/"

for crsr in [6, 12, 24, 48]:
    for ssr in [48, 96, 192]:
        
        ax = plt.subplot(4, 3, splot, projection=to_proj)
        ax.set_extent([-105, -75, 25, 48])
        shapename = 'admin_1_states_provinces_lakes_shp'
        states_shp = shpreader.natural_earth(resolution='50m',
                                             category='cultural', name=shapename)

        for state, info in zip(shpreader.Reader(states_shp).geometries(), 
                               shpreader.Reader(states_shp).records()):
            if info.attributes['admin'] == 'United States of America':

                ax.add_geometries([state], ccrs.PlateCarree(),
                                  facecolor='None', edgecolor='k')
        
        pdict = {'p0.00':None, 'p0.50': None, 'p0.90': None, 'p0.95': None}
        
        for p, ps in zip([0.0, 0.5, 0.95], ['0.00', '0.50', '0.95']):
            
            prom = pickle.load(open(track_occur_dir + str(year) + "/" + str(crsr).zfill(2) + "_"                                     + str(ssr) + "_p" + str(int(100*p)) + "_rematched_mjjas.pkl", "rb"))

            pdict["p" + ps] = prom

        cmap = plt.cm.Greys
        classes = list(range(0, 225, 40))
        norm = BoundaryNorm(classes, ncolors=cmap.N, clip=True)

        #Since each count is a 15 minute window, divide by 4 to get number of hours
        m50_ = gaussian_filter(pdict['p0.50']/4, 30)

        mmp = ax.pcolormesh(lons, lats, m50_, transform=from_proj, norm=norm, cmap=cmap)
        
        plt.colorbar(mmp, ax=ax, shrink=0.4, pad=0.01)

        m0_ = gaussian_filter(pdict['p0.00']/4, 30)
        m95_ = gaussian_filter(pdict['p0.95']/4, 30)

        l1 = ax.contour(lons, lats, m0_, levels=[40], colors=['k',], 
                          transform=from_proj, linestyles='dashed', linewidths=1)

        l3 = ax.contour(lons, lats, m95_, levels=[40], colors=['k',], 
                          transform=from_proj, linewidths=2)
        
        ax.set_title(ascii_lowercase[splot-1] + ".")
        
        splot += 1
plt.tight_layout()

import sys
sys.path.append("C:/Users/ahaberlie/Documents/GitHub/MCS_Tracking/")

import cartopy
import cartopy.crs as ccrs
from scipy.misc import imread
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from mcs.geography.mapping_help import *
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

plt.style.use('seaborn-poster')

import cartopy.io.shapereader as shpreader

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = 20, 20

from_proj = ccrs.PlateCarree()
to_proj = ccrs.AlbersEqualArea(central_longitude=-100.0000, central_latitude=38.0000)

cmap = plt.cm.Greys
classes = list(range(0, 35, 5))
norm = BoundaryNorm(classes, ncolors=cmap.N, clip=True)

crsr = 24
ssr = 96

pecan_data = "../data/track_data/occurrence/PECAN/"

pref = "100_" + str(crsr) + "_" + str(ssr) + "_"

for splot, (pr, prob) in enumerate(zip(['0.0', '0.5', '0.9', '0.95'], 
                                       ['p0', 'p50', 'p90', 'p95'])):

    ax = plt.subplot(2, 2, splot+1, projection=to_proj)
    ax.set_extent([-105, -75, 25, 48])
    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shpreader.natural_earth(resolution='50m',
                                         category='cultural', name=shapename)

    for state, info in zip(shpreader.Reader(states_shp).geometries(), 
                           shpreader.Reader(states_shp).records()):
        if info.attributes['admin'] == 'United States of America':

            ax.add_geometries([state], ccrs.PlateCarree(),
                              facecolor='None', edgecolor='k')

    prom = pickle.load(open(pecan_data + pref + pr + "_pecannight_climo.pkl", "rb"))

    m50_ = gaussian_filter(prom/4, 5)

    mmp = ax.pcolormesh(lons, lats, m50_, transform=from_proj, norm=norm, cmap=cmap)

    plt.colorbar(mmp, ax=ax, shrink=.8, pad=0.01, orientation='horizontal')

import sys
sys.path.append("C:/Users/ahaberlie/Documents/GitHub/MCS_Tracking/")

import pandas as pd

from mcs.format.formatting import to_datetime

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 8, 15


file_location = "../data/slice_data/2015/"

dfs = []
for pref in ['6_48', '12_48', '24_48', '48_48', 
             '6_96', '12_96', '24_96', '48_96',
             '6_192', '12_192', '24_192', '48_192']:

    for i in range(5, 10):

        df = pd.DataFrame.from_csv(file_location + "100_" + pref + "/" + str(i) + "file_info.csv")
        
        df['CRSR'] = pref.split("_")[0]
        
        df['SSR'] = pref.split("_")[1]
        
        dfs.append(df)
           
file_info = pd.concat(dfs, ignore_index=True)

file_info = file_info.apply(pd.to_numeric, errors='ignore')

file_info['datetime'] = pd.to_datetime(file_info.datetime)
file_info = file_info.set_index('datetime')

import matplotlib.dates as mdates
import datetime
import pickle
from pandas.tseries import converter as pdtc
pdtc.register()
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 30, 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['xtick.major.pad']='8'
plt.rcParams['ytick.major.pad']='8'

def duration(x):

    return (x[-1] - x[0]).total_seconds() / 3600

def set_durations(df):
    grouped = df.groupby('storm_num')
    df['duration'] = grouped['storm_num'].transform(lambda x: duration(x.index))
    return df

def get_track_data(start, end, crsr, ssr, p):
    
    year = start.year
    fn = "../data/track_data/rematched/" + str(year) + "/" + str(year) +"_" 
    fn += str(crsr).zfill(2) + "_" + str(ssr).zfill(3) + "_p" + str(int(p*100)).zfill(2) + ".pkl"
    df = pickle.load(open(fn, 'rb'))
    df = df[df.major_axis_length >= 100]
    df['datetime'] = pd.to_datetime(df.datetime)
    df = df.set_index('datetime')
    df = df[(df.index >= start) & (df.index <= end)]
    return df

crsr = 24
ssr = 96
p = 0

for splot, date in enumerate([1, 8, 15, 22]):
    
    ax = plt.subplot(4, 1, splot+1)
    
    stime = datetime.datetime(2015, 6, date, 0, 0)
    etime = datetime.datetime(2015, 6, date+7, 0, 0)

    fi = file_info[(file_info.CRSR==crsr) &                    (file_info.SSR==ssr) &                    (file_info.index >= stime) &                    (file_info.index <= etime)]

    df_0_15 = fi.resample('15Min').sum()

    plt.plot(df_0_15.index.values, 4*df_0_15.conv_area.values, 'k:', label='All Convection')

    df_0 = get_track_data(stime, etime, crsr, ssr, p=0.0)
    df_0 = set_durations(df_0)
    df_0 = df_0[df_0.duration >= 3]

    df_0_15 = df_0.resample('15Min').sum()

    ax.plot(df_0_15.index.values, df_0_15.convection_area.values, '-', 
             color='grey', label="MCS Swath (" + r'$P_{MCS}$' + " ≥ 0.00)")


    df_95 = get_track_data(stime, etime, crsr, ssr, p=0.95)
    df_95 = set_durations(df_95)
    df_95 = df_95[df_95.duration >= 3]

    df_95_15 = df_95.resample('15Min').sum()

    ax.plot(df_95_15.index.values, df_95_15.convection_area.values, 'k-', 
              label="MCS Swath (" + r'$P_{MCS}$' + " ≥ 0.95)")


    s = stime + datetime.timedelta(hours=2)
    e = stime + datetime.timedelta(hours=11)

    while s < etime:

        ax.fill_between([s, e, e, s, s], [0, 0, 200000, 200000, 0], color='grey', alpha=0.2)

        s += datetime.timedelta(hours=24)
        e += datetime.timedelta(hours=24)

    ax.set_ylim(0, 120000)
    ax.set_xlim(stime, etime)
    ax.set_ylabel("Area (" + r'$km^2$' + ")", fontsize=15)
    
    if splot == 0:
        ax.legend(prop={'size': 20})
    
plt.tight_layout()

import matplotlib.dates as mdates
import datetime
import pickle
from pandas.tseries import converter as pdtc
pdtc.register()
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 30, 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['xtick.major.pad']='8'
plt.rcParams['ytick.major.pad']='8'

crsr = 24
ssr = 96
p = 0

for splot, date in enumerate([1, 8, 15, 22]):
    
    ax = plt.subplot(4, 1, splot+1)
    
    stime = datetime.datetime(2015, 7, date, 0, 0)
    etime = datetime.datetime(2015, 7, date+7, 0, 0)

    fi = file_info[(file_info.CRSR==crsr) &                    (file_info.SSR==ssr) &                    (file_info.index >= stime) &                    (file_info.index <= etime)]

    df_0_15 = fi.resample('15Min').sum()

    plt.plot(df_0_15.index.values, 4*df_0_15.conv_area.values, 'k:', label='All Convection')

    df_0 = get_track_data(stime, etime, crsr, ssr, p=0.0)
    df_0 = set_durations(df_0)
    df_0 = df_0[df_0.duration >= 3]

    df_0_15 = df_0.resample('15Min').sum()

    ax.plot(df_0_15.index.values, df_0_15.convection_area.values, '-', 
             color='grey', label="MCS Swath (" + r'$P_{MCS}$' + " ≥ 0.00)")


    df_95 = get_track_data(stime, etime, crsr, ssr, p=0.95)
    df_95 = set_durations(df_95)
    df_95 = df_95[df_95.duration >= 3]

    df_95_15 = df_95.resample('15Min').sum()

    ax.plot(df_95_15.index.values, df_95_15.convection_area.values, 'k-', 
              label="MCS Swath (" + r'$P_{MCS}$' + " ≥ 0.95)")


    s = stime + datetime.timedelta(hours=2)
    e = stime + datetime.timedelta(hours=11)

    while s < etime:

        ax.fill_between([s, e, e, s, s], [0, 0, 200000, 200000, 0], color='grey', alpha=0.2)

        s += datetime.timedelta(hours=24)
        e += datetime.timedelta(hours=24)

    ax.set_ylim(0, 120000)
    ax.set_xlim(stime, etime)
    ax.set_ylabel("Area (" + r'$km^2$' + ")", fontsize=15)
    
    if splot == 0:
        ax.legend(prop={'size': 20})
    
plt.tight_layout()

import matplotlib.dates as mdates
import datetime
import pickle
from pandas.tseries import converter as pdtc
pdtc.register()
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 30, 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['xtick.major.pad']='8'
plt.rcParams['ytick.major.pad']='8'

crsr = 24
ssr = 96
p = 0

for splot, date in enumerate([1, 8, 15, 22]):
    
    ax = plt.subplot(4, 1, splot+1)
    
    stime = datetime.datetime(2015, 8, date, 0, 0)
    etime = datetime.datetime(2015, 8, date+7, 0, 0)

    fi = file_info[(file_info.CRSR==crsr) &                    (file_info.SSR==ssr) &                    (file_info.index >= stime) &                    (file_info.index <= etime)]

    df_0_15 = fi.resample('15Min').sum()

    plt.plot(df_0_15.index.values, 4*df_0_15.conv_area.values, 'k:', label='All Convection')

    df_0 = get_track_data(stime, etime, crsr, ssr, p=0.0)
    df_0 = set_durations(df_0)
    df_0 = df_0[df_0.duration >= 3]

    df_0_15 = df_0.resample('15Min').sum()

    ax.plot(df_0_15.index.values, df_0_15.convection_area.values, '-', 
             color='grey', label="MCS Swath (" + r'$P_{MCS}$' + " ≥ 0.00)")


    df_95 = get_track_data(stime, etime, crsr, ssr, p=0.95)
    df_95 = set_durations(df_95)
    df_95 = df_95[df_95.duration >= 3]

    df_95_15 = df_95.resample('15Min').sum()

    ax.plot(df_95_15.index.values, df_95_15.convection_area.values, 'k-', 
              label="MCS Swath (" + r'$P_{MCS}$' + " ≥ 0.95)")


    s = stime + datetime.timedelta(hours=2)
    e = stime + datetime.timedelta(hours=11)

    while s < etime:

        ax.fill_between([s, e, e, s, s], [0, 0, 200000, 200000, 0], color='grey', alpha=0.2)

        s += datetime.timedelta(hours=24)
        e += datetime.timedelta(hours=24)

    ax.set_ylim(0, 120000)
    ax.set_xlim(stime, etime)
    ax.set_ylabel("Area (" + r'$km^2$' + ")", fontsize=15)
    
    if splot == 0:
        ax.legend(prop={'size': 20})
    
plt.tight_layout()

