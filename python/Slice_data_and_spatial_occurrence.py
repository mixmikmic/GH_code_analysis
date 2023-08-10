import sys
sys.path.append("C:/Users/ahaberlie/Documents/GitHub/MCS/")

import pandas as pd

df = pd.DataFrame.from_csv("../data/slice_data/labeled_slices_020618.csv")

df = df[df.major_axis_length >= 100]

df

import numpy as np

table9 = {'Year':[],
          'CRSR':[],
          'SSR':[],
          'Count_0.00':[],
          'Area_0.00':[],
          'Count_0.50':[],
          'Area_0.50':[],
          'Count_0.90':[],
          'Area_0.90':[],
          'Count_0.95':[],
          'Area_0.95':[]}


for year in [2015, 2016]:
    for crsr in [6, 12, 24, 48]:
        for ssr in [48, 96, 192]:

            table9['Year'].append(year)
            table9['CRSR'].append(crsr)
            table9['SSR'].append(ssr)
            

            t_df = df[(df.CRSR==crsr) & (df.SSR==ssr) & (pd.to_datetime(df.datetime).dt.year==year)]

            for p, ps in zip([0.0, 0.5, 0.9, 0.95], ['0.00', '0.50', '0.90', '0.95']):

                t_df1 = t_df[t_df.mcs_proba >= p]

                table9['Count_' + ps].append(len(t_df1))
                table9['Area_' + ps].append(np.sum(t_df1.area)/10**9)

df1 = pd.DataFrame.from_dict(table9)

df1[['Year','CRSR','SSR','Count_0.00','Area_0.00',
     'Count_0.50','Area_0.50','Count_0.90',
     'Area_0.90','Count_0.95','Area_0.95']]

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
plt.rcParams['figure.figsize'] = 20, 20

from_proj = ccrs.PlateCarree()
to_proj = ccrs.AlbersEqualArea(central_longitude=-100.0000, central_latitude=38.0000)

lons, lats = get_NOWrad_conus_lon_lat()

lons, lats = np.meshgrid(lons, lats)

splot = 1

year = 2015

data_dir = "../data/slice_data/occurrence_maps/"

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
        
        for p, ps in zip([0.0, 0.5, 0.9, 0.95], ['0.00', '0.50', '0.90', '0.95']):
            
            fn = str(year) + "_" + str(crsr).zfill(2) + "_" + str(ssr) + "_p" + str(100*p) + ".pkl"
            prom = pickle.load(open( data_dir + str(year) + "/" + fn, "rb"))

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
plt.rcParams['figure.figsize'] = 20, 20

from_proj = ccrs.PlateCarree()
to_proj = ccrs.AlbersEqualArea(central_longitude=-100.0000, central_latitude=38.0000)

lons, lats = get_NOWrad_conus_lon_lat()

lons, lats = np.meshgrid(lons, lats)

splot = 1

year = 2016

data_dir = "../data/slice_data/occurrence_maps/"

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
        
        for p, ps in zip([0.0, 0.5, 0.9, 0.95], ['0.00', '0.50', '0.90', '0.95']):
            
            fn = str(year) + "_" + str(crsr).zfill(2) + "_" + str(ssr) + "_p" + str(100*p) + ".pkl"
            prom = pickle.load(open( data_dir + str(year) + "/" + fn, "rb"))

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

