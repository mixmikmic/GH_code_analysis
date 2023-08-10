# get set up
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)

import nc_particles

# open the file:
r = nc_particles.Reader('boston_trajectory.nc')

# see what's in there:
print r.variables

# what are the attributes of some of those variables?
print r.get_attributes('depth')
print r.get_attributes('status_codes')

# what times have we got?
print r.times
print"There are {} time steps".format(len(r.times))

# get the positions of the particles at last time step

positions = r.get_timestep(24)

# plot them on a basemap
from mpl_toolkits.basemap import Basemap
m = Basemap(projection='merc',
            llcrnrlat=42.3,
            urcrnrlat=42.5,
            llcrnrlon=-71.1,
            urcrnrlon=-70.7,
            lat_ts=42,
            resolution='h', #NoteL not good resolution, but 'f' takes way too long!
           )
_ = m.drawcoastlines()
_ = m.fillcontinents(color='coral',lake_color='aqua')
m.plot(positions['longitude'], positions['latitude'], 'o', latlon=True)

# plot the path of a particular particle
trajectory = r.get_individual_trajectory(15)

m = Basemap(projection='merc',
            llcrnrlat=42.3,
            urcrnrlat=42.5,
            llcrnrlon=-71.1,
            urcrnrlon=-70.7,
            lat_ts=42,
            resolution='h',# 'f' tales way too long!
           )
_ = m.drawcoastlines()
_ = m.fillcontinents(color='coral',lake_color='aqua')
m.plot(trajectory['longitude'], trajectory['latitude'], 'o-', latlon=True)

