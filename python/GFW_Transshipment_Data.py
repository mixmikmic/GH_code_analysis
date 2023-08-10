get_ipython().magic('matplotlib inline')
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from matplotlib import colors,colorbar
import pandas as pd
import math

# First, let's map potential transshipments

df = pd.DataFrame.from_csv('GFW_transshipment_data_20170222/potential_transshipments_20170222.csv')

df.head()

min_lat = -85
max_lat = 85
min_lon = -180
max_lon = 180

firstlon, lastlat, lastlon, firstlat = min_lon,min_lat,max_lon,max_lat

one_over_cellsize = 4
cellsize = .25

numlats = int((max_lat-min_lat)*one_over_cellsize)
numlons = int((max_lon-min_lon)*one_over_cellsize)

def get_area(lat):
    '''This function converts square degrees to square kilometers. 
    It is not exact, but it is close enough.'''
    lat_degree = 69 # miles
    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0        
    # phi = 90 - latitude
    phi = (math.fabs(lat)+cellsize/2.)*degrees_to_radians #plus half a cell size to get the middle
    lon_degree = math.cos(phi)*lat_degree 
    return  lat_degree*lon_degree* 2.59 # square miles to square km

grid = np.zeros(shape=(numlats,numlons))

# There are more efficient ways to do this
for index, row in df.iterrows():
    lat = int(math.floor(row['latitude']*one_over_cellsize))
    lon = int(math.floor(row['longitude']*one_over_cellsize))
    lat_index = lat-min_lat*one_over_cellsize 
    lon_index = lon-min_lon*one_over_cellsize
    area = get_area(lat*cellsize)*cellsize*cellsize # lat*cellsize is the latitude, 
    try:
        grid[lat_index][lon_index] += float(row['duration_hrs'])/area #/770
    except:
        pass

plt.rcParams["figure.figsize"] = [10,7]

title = "Potential Transshipments, 2012-2016"

fig = plt.figure()

fig_min_value = 10
fig_max_value = 1000
x = np.linspace(firstlon, lastlon, -(firstlon-lastlon)*one_over_cellsize+1)
y = np.linspace(lastlat, firstlat, (firstlat-lastlat)*one_over_cellsize+1)
x, y = np.meshgrid(x, y)
lat_boxes = np.linspace(lastlat,firstlat,num=numlats,endpoint=False)
lon_boxes = np.linspace(firstlon,lastlon,num=numlons,endpoint=False)

m = Basemap(projection='robin',lon_0=0,resolution='c')
m.drawmapboundary(fill_color='#111111')
cont_color = '#454848'
m.fillcontinents(cont_color,lake_color=cont_color )

converted_x, converted_y = m(x, y)
norm = colors.LogNorm(vmin=fig_min_value, vmax=fig_max_value)
m.pcolormesh(converted_x, converted_y, grid*1000, norm=norm, vmin=fig_min_value,
             vmax=fig_max_value, cmap = plt.get_cmap('viridis'))
plt.title(title, color = "#000000", fontsize=18)

# legend
ax = fig.add_axes([0.25, 0.16, 0.5, 0.02]) 
norm = colors.LogNorm(vmin=fig_min_value, vmax=fig_max_value)
lvls = np.logspace(np.log10(fig_min_value),np.log10(fig_max_value),num=3)
cb = colorbar.ColorbarBase(ax,norm = norm, orientation='horizontal',ticks=lvls, 
                           cmap = plt.get_cmap('viridis')) 
cb.ax.set_xticklabels([int(i) for i in lvls], fontsize=10, color = "#000000")
cb.set_label('Hours per 1000 $\mathregular{km^{2}}$ Spent by Reefers in Transhipment Behavoir',
             labelpad=-40, y=0.45, color = "#000000")
plt.savefig('potential_transshipments.png',bbox_inches='tight',dpi=300,transparent=True,pad_inches=0)
plt.show()

# How many Likely Transshipments?
df = pd.DataFrame.from_csv('GFW_transshipment_data_20170222/likely_transshipments_20170222.csv')

df.head()

grid = np.zeros(shape=(numlats,numlons))

# There are more efficient ways to do this
for index, row in df.iterrows():
    lat = int(math.floor(row['latitude']*one_over_cellsize))
    lon = int(math.floor(row['longitude']*one_over_cellsize))
    lat_index = lat-min_lat*one_over_cellsize 
    lon_index = lon-min_lon*one_over_cellsize
    area = get_area(lat*cellsize)*cellsize*cellsize # lat*cellsize is the latitude, 
    try:
        grid[lat_index][lon_index] += 1./area #/770
    except:
        pass

plt.rcParams["figure.figsize"] = [10,7]

title = "Likely Transshipments, 2012-2016"

fig = plt.figure()

fig_min_value = 1
fig_max_value = 10
x = np.linspace(firstlon, lastlon, -(firstlon-lastlon)*one_over_cellsize+1)
y = np.linspace(lastlat, firstlat, (firstlat-lastlat)*one_over_cellsize+1)
x, y = np.meshgrid(x, y)
lat_boxes = np.linspace(lastlat,firstlat,num=numlats,endpoint=False)
lon_boxes = np.linspace(firstlon,lastlon,num=numlons,endpoint=False)

m = Basemap(projection='robin',lon_0=0,resolution='c')
m.drawmapboundary(fill_color='#111111')
cont_color = '#454848'
m.fillcontinents(cont_color,lake_color=cont_color )
converted_x, converted_y = m(x, y)
norm = colors.LogNorm(vmin=fig_min_value, vmax=fig_max_value)
m.pcolormesh(converted_x, converted_y, grid*1000, norm=norm, vmin=fig_min_value,
             vmax=fig_max_value, cmap = plt.get_cmap('viridis'))
plt.title(title, color = "#000000", fontsize=18)

# legend
ax = fig.add_axes([0.25, 0.16, 0.5, 0.02]) 
norm = colors.LogNorm(vmin=fig_min_value, vmax=fig_max_value)
lvls = np.logspace(np.log10(fig_min_value),np.log10(fig_max_value),num=3)
cb = colorbar.ColorbarBase(ax,norm = norm, orientation='horizontal',ticks=lvls, 
                           cmap = plt.get_cmap('viridis')) 
cb.ax.set_xticklabels([int(i) for i in lvls], fontsize=10, color = "#000000")
cb.set_label('Likely Transshipemnts per 1000 $\mathregular{km^{2}}$',
             labelpad=-40, y=0.45, color = "#000000")
plt.savefig('likely_transshipments.png',bbox_inches='tight',dpi=300,transparent=True,pad_inches=0)
plt.show()



