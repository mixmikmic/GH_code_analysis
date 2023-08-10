get_ipython().magic('matplotlib inline')
from mpl_toolkits.basemap import Basemap  # import Basemap matplotlib toolkit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.basemap import cm as basemapcm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from netCDF4 import Dataset, num2date # netcdf4-python module
from datetime import datetime, timedelta

# Import widgets for interactive notebook
from ipywidgets import interact, fixed, Dropdown

def get_forecast_time(delay=6):
    now = datetime.utcnow()
    print "Current time: {0}".format(now)
    fcast_time = datetime.utcnow()
    if now.hour < delay:
        fcast_time = fcast_time - timedelta(1)
        fcast_time = fcast_time.replace(hour=delay+6, minute=0)
    elif (now.hour >=delay and now.hour < (delay+12)):
        fcast_time = fcast_time.replace(hour=0)
    elif now.hour >= (delay+12):
        fcast_time = fcast_time.replace(hour=12)
    print "Forecast time: {0}".format(fcast_time)
    return fcast_time

fcast_time = get_forecast_time()

fcast_date_string = fcast_time.strftime("gfs%Y%m%d/gfs_0p25_%Hz")
data_url = 'http://nomads.ncep.noaa.gov:80/dods/gfs_0p25/{0}'.format(fcast_date_string)
print "Dataset URL: " +  data_url

gfs_fcst = Dataset(data_url, mode='r')
print ""
print "Dataset description:"
print gfs_fcst # get some summary information about the dataset

time = gfs_fcst.variables['time']
print time
valid_dates = num2date(time[:], time.units).tolist()

timelist = [d.strftime('%Y-%m-%d %H:%M') for d in valid_dates]
print "Valid times:"
print timelist
levels = gfs_fcst.variables['lev']
print "Levels:"
print [level for level in levels[:]]

def plot_rain(timestamp):
    idx = timelist.index(timestamp)
    apcp = gfs_fcst.variables['apcpsfc'][idx,:,:]
    prmsl = gfs_fcst.variables['prmslmsl'][idx,:,:]
    hgt1000 = gfs_fcst.variables['hgtprs'][idx,0,:,:]
    hgt500 = gfs_fcst.variables['hgtprs'][idx,12,:,:]
    thk = (hgt500 - hgt1000)/10.

    lats = gfs_fcst.variables['lat'][:]; lons = gfs_fcst.variables['lon'][:]
    lons, lats = np.meshgrid(lons, lats)
    fig = plt.figure(figsize=(18,10))

    m = Basemap(projection='mill', llcrnrlon=130., llcrnrlat=-45, 
                urcrnrlon=160., urcrnrlat=-25.,resolution='i')
    
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    x,y = m(lons, lats) # convert lats/lons to map projection coordinates
    clevs = [0,.1,.25,.5,.75,1.0,1.5,2.0,3.0,4.0,5.0,7.0,10.0,
             15.0,20.0,25.0,30.0,40.0,50.0,60.0,75.0]
    

    cs = m.contourf(x,y, apcp, clevs, cmap=basemapcm.s3pcpn, extend='max')  # color-filled contours
    ct = m.contour(x, y, thk, np.arange(500, 600, 4), colors='0.75', linestyle='--')
    cp = m.contour(x, y, prmsl/100., np.arange(900, 1040, 2), colors='k')
    cb = m.colorbar(cs, extend='max')  # draw colorbar
    parallels = m.drawparallels(np.arange(-50, 0, 10), labels=[1,0,0,0])  # draw parallels, label on left
    meridians = m.drawmeridians(np.arange(80, 190, 10), labels=[0,0,0,1]) # label meridians on bottom
    
    fig.suptitle('Forecast apcp (mm) from %s for %s' % (valid_dates[0],valid_dates[idx]),fontweight='bold')

def plot_tsindex(timestamp):
    idx = timelist.index(timestamp)
    apcp = gfs_fcst.variables['apcpsfc'][idx,:,:]
    prmsl = gfs_fcst.variables['prmslmsl'][idx,:,:]
    no4lftx = gfs_fcst.variables['no4lftxsfc'][idx,:,:]
    cape = gfs_fcst.variables['capesfc'][idx,:,:]
    #thk = (hgt500 - hgt1000)/10.

    lats = gfs_fcst.variables['lat'][:]; lons = gfs_fcst.variables['lon'][:]
    lons, lats = np.meshgrid(lons, lats)
    fig = plt.figure(figsize=(18,10))

    m = Basemap(projection='mill', llcrnrlon=130., llcrnrlat=-45, 
                urcrnrlon=160., urcrnrlat=-25.,resolution='i')
    
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    x,y = m(lons, lats) # convert lats/lons to map projection coordinates
    clevs = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000,
             1500, 2000, 2500, 3000, 4000, 5000]

    cs = m.contourf(x,y, cape, clevs, cmap=basemapcm.s3pcpn, extend='max')  # color-filled contours
    clm = m.contour(x, y, no4lftx, np.arange(-20, 0, 2), colors='0.75', linestyle='--')
    clp = m.contour(x, y, no4lftx, np.arange(0, 20, 2), colors='0.75', linestyle='-')
    cp = m.contour(x, y, prmsl/100., np.arange(900, 1040, 2), colors='k')
    #cc = m.contour(x, y, cape, np.arange(0, 5000, 100), colors='b', linewidth=2)
    cb = m.colorbar(cs, extend='max')  # draw colorbar
    parallels = m.drawparallels(np.arange(-50, 0, 10), labels=[1,0,0,0])  # draw parallels, label on left
    meridians = m.drawmeridians(np.arange(80, 190, 10), labels=[0,0,0,1]) # label meridians on bottom
    
    fig.suptitle('Forecast CAPE (J/kg), LI (K) for %s' % (valid_dates[idx]),fontweight='bold')

def plot_wind(timestamp):
    idx = timelist.index(timestamp)
    uu = gfs_fcst.variables['ugrdsig995'][idx,:,:]
    vv = gfs_fcst.variables['vgrdsig995'][idx,:,:]
    wspd = np.sqrt(uu*uu + vv*vv)
    pmsl = gfs_fcst.variables['prmslmsl'][idx,:,:]
    hgt1000 = gfs_fcst.variables['hgtprs'][idx,0,:,:]
    hgt500 = gfs_fcst.variables['hgtprs'][idx,12,:,:]
    thk = (hgt500 - hgt1000)/10.


    lats = gfs_fcst.variables['lat'][:]; lons = gfs_fcst.variables['lon'][:]
    lons, lats = np.meshgrid(lons, lats)
    fig = plt.figure(figsize=(18,10))

    m = Basemap(projection='mill',llcrnrlon=130., llcrnrlat=-45, 
                urcrnrlon=160., urcrnrlat=-25.,resolution='i')
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    x,y = m(lons, lats) # convert lats/lons to map projection coordinates
    clevs = np.arange(5.,41.)

    cs = m.contourf(x,y,wspd,clevs,cmap=basemapcm.GMT_haxby_r, extend='both')  # color-filled contours
    ct = m.contour(x,y,thk, np.arange(500, 600, 4), colors='0.75', linestyle='--')
    cp = m.contour(x,y,pmsl/100.,np.arange(900, 1040, 2), colors='k')
    cb = m.colorbar(cs,extend='both')  # draw colorbar
    parallels = m.drawparallels(np.arange(-50,0,10),labels=[1,0,0,0])  # draw parallels, label on left
    meridians = m.drawmeridians(np.arange(80,190,10),labels=[0,0,0,1]) # label meridians on bottom
    
    fig.suptitle('Forecast wind speed (m/s) from %s for %s' % (valid_dates[0],valid_dates[idx]),fontweight='bold')
    

interact(plot_rain, timestamp=Dropdown(options=timelist, value=timelist[1]))

interact(plot_wind, timestamp=Dropdown(options=timelist, value=timelist[1]))

interact(plot_tsindex, timestamp=Dropdown(options=timelist, value=timelist[1]))



