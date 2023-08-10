from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib import pylab
from JSAnimation.IPython_display import display_animation

get_ipython().magic('matplotlib inline')
pylab.rcParams['figure.figsize'] = (10, 6)

#Center the map over the United States by specifying lat/lon values
def make_map(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat):
    m = Basemap(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat)
    m.fillcontinents(color = 'lightgreen', lake_color='aqua')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.drawmapboundary(fill_color='aqua')
    return m

my_map = make_map(llcrnrlon = -130, llcrnrlat = 15, 
                  urcrnrlon = -60, urcrnrlat = 60)

my_map

#Global Variables
lons = []
lats = []
iterr = 0

# re-initialize map
my_map = make_map(llcrnrlon = -130, llcrnrlat = 15, 
                  urcrnrlon = -60, urcrnrlat = 60)

# initialize point object
x,y = my_map(0, 0)
point = my_map.plot(x, y, 'ro', markersize=5)[0]

#Create a list of all the US capital coordinates
states = pd.read_csv('../data/us-state-capitals.csv')

#Create the initialization and animation functions
def init():
    point.set_data([], [])
    return point,

def animate(i):
    global lons
    global lats
    global iterr

    lats.append(states['latitude'][iterr])
    lons.append(states['longitude'][iterr])
    iterr += 1

    x, y = my_map(lons, lats)
    point.set_data(x, y)
    return point,

#Create the final animation
anim = animation.FuncAnimation(plt.gcf(), animate, init_func=init,
                               frames=50, interval=100, blit=True)
display_animation(anim)



