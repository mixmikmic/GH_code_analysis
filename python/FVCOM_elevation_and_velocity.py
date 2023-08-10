from pylab import *
import matplotlib.tri as Tri
import netCDF4
import datetime as dt

# DAP Data URL
# MassBay GRID
#url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc'
# GOM3 GRID
#url='http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc'
# GOM3 Monthly mean
url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3/mean'
# Open DAP
nc = netCDF4.Dataset(url).variables
nc.keys()

# take a look at the "metadata" for the variable "u"
print nc['u']

shape(nc['temp'])

shape(nc['nv'])

# Desired time for snapshot
# ....right now (or some number of hours from now) ...
#start = dt.datetime.utcnow() + dt.timedelta(hours=6)
# ... or specific time (UTC)
start = dt.datetime(1998,4,15,0,0,0)

# Get desired time step  
time_var = nc['time']
itime = netCDF4.date2index(start,time_var,select='nearest')

# Get lon,lat coordinates for nodes (depth)
lat = nc['lat'][:]
lon = nc['lon'][:]
# Get lon,lat coordinates for cell centers (depth)
latc = nc['latc'][:]
lonc = nc['lonc'][:]
# Get Connectivity array
nv = nc['nv'][:].T - 1 

dtime = netCDF4.num2date(time_var[itime],time_var.units)
daystr = dtime.strftime('%Y-%b-%d %H:%M')
print daystr

tri = Tri.Triangulation(lon,lat, triangles=nv)

# get current at layer [0 = surface, -1 = bottom]
ilayer = 0
u = nc['u'][itime, ilayer, :]
v = nc['v'][itime, ilayer, :]
# Get water level
h = nc['zeta'][itime,:]  # water level 

#woods hole
levels=arange(-0.3,0.3,0.01)   # water level contours to plot
ax = [-70.7, -70.6, 41.48, 41.55]
maxvel = 1.0
subsample = 2

#boston harbor
levels=arange(-0.3,0.3,0.01)   # water level contours to plot
ax= [-70.97, -70.82, 42.25, 42.35] # 
maxvel = 0.5
subsample = 3

# whole gulf
levels=arange(-0.3,0.3,0.01)   # water level contours to plot
ax= [-74.5, -70, 39.0, 41.0] # 
maxvel = 0.2
subsample = 10

# whole gom3
levels=arange(-0.5,0.5,0.02)   # water level contours to plot
ax= [-80, -55, 34.0, 48.0] # 
maxvel = 1.0
subsample = 20

# find velocity points in bounding box
ind = argwhere((lonc >= ax[0]) & (lonc <= ax[1]) & (latc >= ax[2]) & (latc <= ax[3]))

np.random.shuffle(ind)
Nvec = int(len(ind) / subsample)
idv = ind[:Nvec]

# tricontourf plot of water depth with vectors on top
figure(figsize=(18,10))
subplot(111,aspect=(1.0/cos(mean(lat)*pi/180.0)))
#tricontourf(tri, h-h.mean(),levels=levels,shading='faceted',cmap=plt.cm.gist_earth)
tricontourf(tri, h-h.mean(),levels=levels,shading='faceted')
#axis(ax)
gca().patch.set_facecolor('0.5')
cbar=colorbar()
cbar.set_label('Water Level (m)', rotation=-90)
#Q = quiver(lonc[idv],latc[idv],u[idv],v[idv],scale=20)
#maxstr='%3.1f m/s' % maxvel
#qk = quiverkey(Q,0.92,0.08,maxvel,maxstr,labelpos='W')
title('NECOFS Velocity, Layer %d, %s UTC' % (ilayer, daystr));


# turn the triangles into a PolyCollection
verts = concatenate((tri.x[tri.triangles][..., None],
      tri.y[tri.triangles][..., None]), axis=2)
collection = PolyCollection(verts)
collection.set_edgecolor('none')

# set the magnitude of the polycollection to the speed
collection.set_array(-h)
collection.norm.vmin=-300
collection.norm.vmax=0

fig=figure(figsize=(12,12))
ax=fig.add_subplot(111)
m.drawmapboundary(fill_color='0.3')
#m.drawcoastlines()
#m.fillcontinents()
# add the speed as colored triangles 
ax.add_collection(collection) # add polygons to axes on basemap instance
title('FVCOM Bathymetry')

