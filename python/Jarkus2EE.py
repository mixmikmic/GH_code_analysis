# !pip install git+https://github.com/openearth/jarkus.git
get_ipython().magic('matplotlib inline')

import numpy as np
from jarkus.transects import Transects
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import num2date

tr = Transects(url='http://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/jarkus/profiles/transect.nc')
ids = tr.get_data('id')
idx = np.nonzero(np.logical_or(np.logical_and(ids>=3e6,ids<5e6),
               np.logical_and(ids>=16e6,ids<18e6)))[0]
yearx = np.arange(2015,2018)
#idx

get_ipython().run_cell_magic('time', '', "lat = []\nlon = []\ndf_tot = pd.DataFrame()\n\nfor ix in range(len(yearx)):\n        tr.set_filter(alongshore=idx, year=yearx[ix])\n        z = np.squeeze(tr.get_data('altitude'))[idx,:]\n        lat = tr.get_data('lat')[idx,:]\n        lon = tr.get_data('lon')[idx,:]\n        t = tr.get_data('time')\n        # get times as datetime\n        t_topo = [num2date(tt,tr.ds.variables['time_topo'].units) \n                  if ~np.isnan(tt) \n                  else np.nan \n                  for tt in tr.get_data('time_topo')[0][idx]]\n        t_bathy = [num2date(tt,tr.ds.variables['time_bathy'].units) \n                  if ~np.isnan(tt) \n                  else np.nan \n                  for tt in tr.get_data('time_bathy')[0][idx]]\n        print('year '+str(yearx[ix]))\n        df_base = pd.DataFrame(data={'id': np.ravel(np.tile(ids[idx],(lat.shape[1],1)).T),\n                              'lat': np.ravel(np.squeeze(lat)),\n                              'lon': np.ravel(np.squeeze(lon))})\n        \n        df = pd.DataFrame(data={'z_'+str(yearx[ix]): np.ravel(z),\n                              'time_topo_'+str(yearx[ix]): np.ravel(np.tile(t_topo,(lat.shape[1],1)).T),\n                              'time_bathy_'+str(yearx[ix]): np.ravel(np.tile(t_bathy,(lat.shape[1],1)).T),\n                             })\n        df_tot = pd.concat([df_tot, df],axis=1,join='outer')\n        #\n# add to the base with lat, lon, id\ndf_tr = pd.concat([df_base, df_tot],axis=1,join='outer')")

# get rid of values where you have NaN's for these three columns:
df_tr.dropna(subset=['z_2015', 'z_2016', 'z_2017'], how='all')



