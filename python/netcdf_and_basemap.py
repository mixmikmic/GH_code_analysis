import numpy as np
from scipy.io import netcdf
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

f = netcdf.netcdf_file('./data/r2_reanalysis.nc', 'r')

for var in f.variables:
    print var

u = f.variables['u']
v = f.variables['v']
t = f.variables['t']

longitude = f.variables['lon'][:]
latitude = f.variables['lat'][:]
level = f.variables['level'][:]
time = f.variables['time']

print u.shape, longitude.shape, latitude.shape, level.shape, time.shape
print time.units

print latitude
print longitude
print level

def unpack(var):
    return var[:] * var.scale_factor + var.add_offset 

u = unpack(u)
v = unpack(v)
t = unpack(t)

lons, lats = np.meshgrid(longitude, latitude)
print lons.shape, lats.shape

from mpl_toolkits.basemap import Basemap

fig = plt.figure(figsize=(16,35))
m = Basemap(llcrnrlon=lons.min(), llcrnrlat=lats.min(), urcrnrlon=lons.max(), urcrnrlat=lats.max())

m.drawcoastlines()
m.drawcountries()

skip = 2

cs = m.contourf(lons, lats, t[0, 0, :, :])
qv = m.quiver(lons[::skip, ::skip], lats[::skip, ::skip], u[0, 0, ::skip, ::skip], v[0, 0, ::skip, ::skip])

fig = plt.figure(figsize=(20,20))

ax1 = fig.add_subplot(121)
ax1.set_title('Orthographic Projection')

m = Basemap(projection='ortho',lat_0=-15,lon_0=300, ax=ax1)

m.drawcoastlines()
m.drawcountries()

skip = 2

x, y = m(lons, lats)

cs = m.contourf(x, y, t[0, 0, :, :])
qv = m.quiver(x[::skip, ::skip], y[::skip, ::skip], u[0, 0, ::skip, ::skip], v[0, 0, ::skip, ::skip])

###############################################################################################

ax2 = fig.add_subplot(122)
ax2.set_title('Polar Azimuthal Equidistant Projection')

m = Basemap(projection='spaeqd',boundinglat=-10,lon_0=270, ax=ax2)

m.drawcoastlines()
m.drawcountries()

skip = 2

x, y = m(lons, lats)

cs = m.contourf(x, y, t[0, 0, :, :])
qv = m.quiver(x[::skip, ::skip], y[::skip, ::skip], u[0, 0, ::skip, ::skip], v[0, 0, ::skip, ::skip])

fig = plt.figure(figsize=(16,35))
m = Basemap(llcrnrlon=-180, llcrnrlat=lats.min(), urcrnrlon=180, urcrnrlat=lats.max())

m.drawcoastlines()
m.drawcountries()

skip = 2

cs = m.contourf(lons, lats, t[0, 0, :, :])
qv = m.quiver(lons[::skip, ::skip], lats[::skip, ::skip], u[0, 0, ::skip, ::skip], v[0, 0, ::skip, ::skip])

def flip_grid(var, lons):
    fltr = lons >= 180
    # fltr =  [False False False ... True  True  True]
    newlons = np.concatenate(((lons - 360)[fltr], lons[~fltr]), axis=-1)
    # newlons = [-180 -177.5 -175 ... -5 -2.5 ] concatenated with [0 2.5 5 ... 175 177.5]
    # newlons = [-180 -177.5 -175 ... 175 177.5 180]
    if var.ndim == 2:
        newvar = np.concatenate((var[:, fltr], var[:, ~fltr]), axis=-1)
    elif var.ndim == 3:
        newvar = np.concatenate((var[:, :, fltr], var[:, :, ~fltr]), axis=-1)
    elif var.ndim == 4:
        newvar = np.concatenate((var[:, :, :, fltr], var[:, :, :, ~fltr]), axis=-1)        
        
    return newvar, newlons

u, newlon = flip_grid(u, longitude)
v, newlon = flip_grid(v, longitude)
t, newlon = flip_grid(t, longitude)

print newlon

lons, lats = np.meshgrid(newlon, latitude)

fig = plt.figure(figsize=(16,35))
m = Basemap(llcrnrlon=lons.min(), llcrnrlat=lats.min(), urcrnrlon=lons.max(), urcrnrlat=lats.max())

m.drawcoastlines()
m.drawcountries()

skip = 2

cs = m.contourf(lons, lats, t[0, 0, :, :])
qv = m.quiver(lons[::skip, ::skip], lats[::skip, ::skip], u[0, 0, ::skip, ::skip], v[0, 0, ::skip, ::skip])

def find_nearest(x, y, gridx, gridy):

    distance = (gridx - x)**2 + (gridy - y)**2
    idx = np.where(distance == distance.min())
    # idx = (array([45]), array([0]))
    
    return [idx[0][0], idx[1][0]]

idx = find_nearest(-46.616667, -23.650000, lons, lats)

print lons[idx[0], idx[1]], lats[idx[0], idx[1]]

fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(111)

newlats, levs = np.meshgrid(lats[:, idx[1]], level)

clevs = np.linspace(-50, 50, 8)
cs = ax.contourf(newlats, levs, u[0, :, :, idx[1]], clevs, cmap='PiYG', extend='both')
cbar = plt.colorbar(cs)
ax.invert_yaxis()

print time[:]

print time.units

from datetime import datetime, timedelta

def get_dates(time):
    splited_date = time.units.split()
    # splited_date = ['hours', 'since', '1800-1-1', '00:00:00']
    start_date_string = ' '.join(splited_date[-2:])
    # start_date_string = 1800-1-1 00:00:00
    # convert string into datetime object
    start_date = datetime.strptime(start_date_string, '%Y-%m-%d %H:%M:%S')
    
    dates = [start_date + timedelta(hours=i) for i in time[:]]
    return dates

dates = get_dates(time)
print dates

u10 = f.variables['u10']
v10 = f.variables['v10']
prate = f.variables['prate']
tmax = f.variables['tmax']
tmin = f.variables['tmin']

print u10.shape, prate.units

u10 = unpack(u10)
v10 = unpack(v10)
prate = unpack(prate)
tmax = unpack(tmax)
tmin = unpack(tmin)

u10, newlon = flip_grid(u10, longitude)
v10, newlon = flip_grid(v10, longitude)
prate, newlon = flip_grid(prate, longitude)
tmax, newlon = flip_grid(tmax, longitude)
tmin, newlon = flip_grid(tmin, longitude)

lons, lats = np.meshgrid(newlon, latitude)

idx = find_nearest(-46.248889, -22.880833, lons, lats)

import pandas as pd

tx2C = tmax - 273.15 # K to oC
tm2C = tmin - 273.15 # K to oC
acum = prate * 3600. * 24. # mm/s mm/day

data = {'u10': u10[:, 0, idx[0], idx[1]], 'v10': v10[:, 0, idx[0], idx[1]], 'prec': acum[:, idx[0], idx[1]],         'tmax': tx2C[:, 0, idx[0], idx[1]], 'tmin': tm2C[:, 0, idx[0], idx[1]]}

df = pd.DataFrame(data, index=dates)

print df

df.to_csv('./data/modeldata.csv', sep=';')

