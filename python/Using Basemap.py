import pandas as pd #Python's data manipulation library
import matplotlib.pyplot as plt #Python's graphing library
from mpl_toolkits.basemap import Basemap #The geomapping library built ontop of matplotlib

geoRainfallDF = pd.read_pickle("../saved_dataframes/geoRainfallDF")

lats = geoRainfallDF["latitude"].values
lons = geoRainfallDF["longitude"].values

m = Basemap(projection='lcc', #this is the Lambert Conformal Conic projection
            resolution="c",
            width=100000., # we're specifying a width of 100,000 meters
            height=100000., # we're specifying a width of 100,000 meters
            lat_0=43.761539, # Toronto's lat coord
            lon_0=-79.411079) # Toronto's lon coord

m.fillcontinents(color='lightgray',zorder=0)
x, y = m(lats, lons)

m.scatter(x, y,0.05,color='r')
m.drawcoastlines()
m.drawstates() # not really nessesary but it's good to know that you can use this
m.drawcountries() # not really nessesary but it's good to know that you can use this
plt.title("Rainfall in Toronto")
plt.show()
plt.rcParams['savefig.dpi'] = 500

