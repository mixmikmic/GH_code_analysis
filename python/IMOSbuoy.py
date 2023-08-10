get_ipython().magic('matplotlib inline')
from pylab import *
import netCDF4
import datetime as dt

# Offshore Sydney buoy data
nc_data=netCDF4.Dataset('http://www.metoc.gov.au/thredds/dodsC/MHLWAVE/Sydney/IMOS_ANMN-NSW_W_20050215T020000Z_WAVESYD_FV01_END-20080312T210000Z.nc')

nc_data.variables.keys()

print nc_data.variables['HRMS']

start = dt.datetime(1985,1,1)
# Get desired time step  
time_var = nc_data.variables['TIME']
itime = netCDF4.date2index(start,time_var,select='nearest')
dtime = netCDF4.num2date(time_var[itime],time_var.units)
daystr = dtime.strftime('%Y-%b-%d %H:%M')
print 'buoy record start time:',daystr

end = dt.datetime(2018,1,1)
# Get desired time step  
time_var = nc_data.variables['TIME']
itime2 = netCDF4.date2index(end,time_var,select='nearest')
dtime2 = netCDF4.num2date(time_var[itime2],time_var.units)
dayend = dtime2.strftime('%Y-%b-%d %H:%M')
print 'buoy record end time:',dayend

loni = nc_data.variables['LONGITUDE'][:]
lati = nc_data.variables['LATITUDE'][:]
print loni,lati
names=[]
names.append('Offshore Sydney Buoy')

times = nc_data.variables['TIME']
jd_data = netCDF4.num2date(times[:],times.units).flatten()
hm_data = nc_data.variables['HMEAN'][:].flatten()
#hs_data = ma.masked_where(hs_data > 98., hs_data)

# make the time series plot, with nicely formatted labels
MyDateFormatter = DateFormatter('%Y-%b-%d')
fig = plt.figure(figsize=(8,6), dpi=80) 
ax1 = fig.add_subplot(111)

ax1.plot(jd_data[itime:itime2],hm_data[itime:itime2]) 
ax1.xaxis.set_major_locator(WeekdayLocator(byweekday=MO,interval=12))
ax1.xaxis.set_major_formatter(MyDateFormatter)
ax1.grid(True)
setp(gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('IMOS Offshore Sydney Buoy Data')
ax1.set_ylabel('meters')
ax1.legend(names,loc='upper right')

#OCEAN FORECAST MODEL WW3
nc_data=netCDF4.Dataset('http://134.178.63.198/thredds/dodsC/paccsapwaves_gridded/ww3.glob_24m.199801.nc')

nc_data.variables.keys()

print nc_data.variables['t']

start = dt.datetime(1980,1,1)
# Get desired time step  
time_var = nc_data.variables['time']
itime = netCDF4.date2index(start,time_var,select='nearest')
dtime = netCDF4.num2date(time_var[itime],time_var.units)
daystart = dtime.strftime('%Y-%b-%d %H:%M')
print 'WW3 dataset start time:',daystart

end = dt.datetime(2018,1,1)
# Get desired time step  
time_var = nc_data.variables['time']
itime2 = netCDF4.date2index(end,time_var,select='nearest')
dtime2 = netCDF4.num2date(time_var[itime2],time_var.units)
dayend = dtime2.strftime('%Y-%b-%d %H:%M')
print 'WW3 dataset end time:', dayend

maroubra_lon=[151.25750340000002]
maroubra_lat=[-33.947314899999]

# Function to find index to nearest point
def near(array,value):
    idx=(abs(array-value)).argmin()
    return idx

lon_model=nc_data.variables['longitude'][:].flatten()
lat_model=nc_data.variables['latitude'][:].flatten()
ix = near(lon_model, maroubra_lon)
iy = near(lat_model, maroubra_lat)
print ix, iy

times = nc_data.variables['time']
jd_model = netCDF4.num2date(times[:],times.units)
hs_model = nc_data.variables['hs'][:,ix,iy]
istart_model = netCDF4.date2index(start,times,select='nearest')
istop_model = netCDF4.date2index(end,times,select='nearest')

# make the time series plot, with nicely formatted labels
MyDateFormatter = DateFormatter('%Y-%b-%d')
fig = plt.figure(figsize=(5,5), dpi=80) 
ax1 = fig.add_subplot(111)

names=[]
names.append('Maroubra')

#ax1.plot(jd_data[istart:istop],hs_data[istart:istop]) 
ax1.plot(jd_model[istart_model:istop_model],hs_model[istart_model:istop_model])
#ax1.plot(jd_mod2[istart_mod2:istop_mod2],hs_mod2[istart_mod2:istop_mod2])
ax1.xaxis.set_major_locator(WeekdayLocator(byweekday=MO))
ax1.xaxis.set_major_formatter(MyDateFormatter)
ax1.grid(True)
setp(gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('WAVEWATCH III model')
ax1.set_ylabel('meters')
ax1.legend(names,loc='upper right')

