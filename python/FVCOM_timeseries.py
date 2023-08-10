# Plot time series data from FVCOM model from list of lon,lat locations
# (uses the nearest point, no interpolation)
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import datetime as dt
import pandas as pd
from StringIO import StringIO

# make dictionary of various model simulation endpoints
models={}
models['Massbay_forecast']='http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc'
models['GOM3_Forecast']='http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc'
models['Massbay_forecast_archive']='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/archives/necofs_mb'
models['GOM3_30_year_hindcast']='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3'

def start_stop(url,tvar):
    nc = netCDF4.Dataset(url)
    ncv = nc.variables
    time_var = ncv[tvar]
    first = netCDF4.num2date(time_var[0],time_var.units)
    last = netCDF4.num2date(time_var[-1],time_var.units)

    print first.strftime('%Y-%b-%d %H:%M')
    print last.strftime('%Y-%b-%d %H:%M')

tvar = 'time'
for model,url in models.iteritems():
    print model

    try:
        start_stop(url,tvar)
    except:
        print '[problem accessing data]'

#model='Massbay_forecast_archive' 
model='Massbay_forecast'
#model='GOM3_Forecast'
#model='GOM3_30_year_hindcast'
url=models[model]
# Desired time for snapshot
# ....right now (or some number of hours from now) ...
start = dt.datetime.utcnow() - dt.timedelta(hours=72)
stop = dt.datetime.utcnow() + dt.timedelta(hours=72)

# ... or specific time (UTC)
#start = dt.datetime(2004,9,1,0,0,0)
#stop = dt.datetime(2004,11,1,0,0,0)

def dms2dd(d,m,s):
    return d+(m+s/60.)/60.
  

dms2dd(41,33,15.7)
dms2dd(42,51,17.40)

-dms2dd(70,30,20.2)
-dms2dd(70,18,42.0)

x = '''
Station, Lat, Lon
Falmouth Harbor,    41.541575, -70.608020
Sage Lot Pond, 41.554361, -70.505611
'''

x = '''
Station, Lat, Lon
Boston,             42.368186, -71.047984
Carolyn Seep Spot,    39.8083, -69.5917
Falmouth Harbor,  41.541575, -70.608020
'''

# Enter desired (Station, Lat, Lon) values here:
x = '''
Station, Lat, Lon
Boston,             42.368186, -71.047984
Scituate Harbor,    42.199447, -70.720090
Scituate Beach,     42.209973, -70.724523
Falmouth Harbor,    41.541575, -70.608020
Marion,             41.689008, -70.746576
Marshfield,         42.108480, -70.648691
Provincetown,       42.042745, -70.171180
Sandwich,           41.767990, -70.466219
Hampton Bay,        42.900103, -70.818510
Gloucester,         42.610253, -70.660570
'''

x = '''
Station, Lat, Lon
Buoy A,             42.52280, -70.56535
Buoy B,             43.18089, -70.42788
Nets,               42.85483, -70.3116
DITP,               42.347  , -70.960
'''

# Create a Pandas DataFrame
obs=pd.read_csv(StringIO(x.strip()), sep=",\s*",index_col='Station',engine='python')

obs

# find the indices of the points in (x,y) closest to the points in (xi,yi)
def nearxy(x,y,xi,yi):
    ind = np.ones(len(xi),dtype=int)
    for i in np.arange(len(xi)):
        dist = np.sqrt((x-xi[i])**2+(y-yi[i])**2)
        ind[i] = dist.argmin()
    return ind

nc=netCDF4.Dataset(url)

# open NECOFS remote OPeNDAP dataset 
ncv = nc.variables

# find closest NECOFS nodes to station locations
obs['0-Based Index'] = nearxy(ncv['lon'][:],ncv['lat'][:],obs['Lon'],obs['Lat'])
obs

ncv['lon'][0:10]

# get time values and convert to datetime objects
time_var = ncv['time']
istart = netCDF4.date2index(start,time_var,select='nearest')
istop = netCDF4.date2index(stop,time_var,select='nearest')
jd = netCDF4.num2date(time_var[istart:istop],time_var.units)

# get all time steps of water level from each station
# NOTE: this takes a while....
nsta=len(obs)
z = np.ones((len(jd),nsta))
layer = 0  # surface layer =0, bottom layer=-1
for i in range(nsta):
    z[:,i] = ncv['temp'][istart:istop,layer,obs['0-Based Index'][i]]
    

# make a DataFrame out of the interpolated time series at each location
zvals=pd.DataFrame(z*9./5.+32.,index=jd,columns=obs.index)

# list out a few values
zvals.head()

# plotting at DataFrame is easy!
ax=zvals.plot(figsize=(16,4),grid=True,
    title=('NECOFS Forecast Bottom Water Temperature from %s Grid' % model),legend=False);
# read units from dataset for ylabel
plt.ylabel(ncv['temp'].units)
# plotting the legend outside the axis is a bit tricky
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));

# make a new DataFrame of maximum water levels at all stations
b=pd.DataFrame(zvals.idxmax(),columns=['time of max water temp (UTC)'])
# create heading for new column containing max water level
zmax_heading='tmax (%s)' % ncv['temp'].units
# Add new column to DataFrame
b[zmax_heading]=zvals.max()

b





