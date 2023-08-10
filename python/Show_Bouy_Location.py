IDs= [41002, 41040, 41041, 41043, 41044, 41046, 41047, 41048, 41049, 42059]

def get_bouy_param(ID):
    values = {
     41002 : {'c':'b' , 'loc': 'Atlantic', 'line_style': ':', 'lat' : 31.760 , 'lon': -74.840},
     41040 : {'c':'gray' , 'loc': 'Atlantic', 'line_style': '-', 'lat' : 14.516 , 'lon': -53.024},
     41041 : {'c':'c' , 'loc': 'Atlantic', 'line_style': '-', 'lat' : 14.329 , 'lon': -46.082},
     41043 : {'c':'y' , 'loc': 'Carribean', 'line_style': '--', 'lat' : 21.132 , 'lon': -64.856},
     41044 : {'c':'lightseagreen' , 'loc': 'Atlantic', 'line_style': '--', 'lat' : 21.575 , 'lon': -58.625},
     41046 : {'c':'black' , 'loc': 'Carribean', 'line_style': '--', 'lat' : 23.866 , 'lon': -68.481},
     41047 : {'c':'pink' , 'loc': 'Carribean', 'line_style': '--', 'lat' : 27.517  , 'lon': -71.483},
     41048 : {'c':'cyan' , 'loc': 'Mid. Atlantic', 'line_style': ':', 'lat' : 31.860 , 'lon': -69.590},
     41049 : {'c':'navy' , 'loc': 'Mid. Atlantic', 'line_style': ':', 'lat' : 27.537 , 'lon': -62.945},
     42059 : {'c':'purple' , 'loc': 'Carribean', 'line_style': '--', 'lat' : 15.252 , 'lon': -67.510},
    }
    return values[ID]

from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import numpy as np
import matplotlib.pyplot as plt



# create the figure and axes instances.
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.9,0.9])
# setup of basemap ('lcc' = lambert conformal conic).
# use major and minor sphere radii from WGS84 ellipsoid.
m = Basemap(projection='aeqd',
              lon_0 = -60,
              lat_0 = 20,
              width = 5000000,
              height = 5000000)

# draw coastlines and political boundaries.
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='g',lake_color='aqua')
m.drawcountries(color='coral')

parallels = np.arange(0.,80,20.)
m.drawparallels(parallels,labels=[1,0,0,1])
meridians = np.arange(10.,360.,30.)
m.drawmeridians(meridians,labels=[1,0,0,1])

ax.set_title('Bouy Location')

for ID in IDs:

    x, y = m(get_bouy_param(ID)['lon'], get_bouy_param(ID)['lat'])
    m.scatter(x, y, marker='D',color=get_bouy_param(ID)['c'])
    
    plt.text(x, y, ID ,fontsize=8,fontweight='bold',
                    ha='left',va='bottom',color='k')
                

plt.show()

