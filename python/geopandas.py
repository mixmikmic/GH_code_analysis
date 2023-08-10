from IPython.display import display, Image # Displays things nicely
import pandas as pd # Key tool 
import matplotlib.pyplot as plt # Helps plot
import numpy as np # Numerical operations
import os

from census import Census # This is new...
from us import states

import fiona # Needed for geopandas to run
import geopandas as gpd # this is the main geopandas 
from shapely.geometry import Point, Polygon # also needed

##########################
# Then this stuff below allows us to make a nice inset


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

cwd = os.getcwd()

regions_shape = cwd + "\\shape_files\\NYC\\ZIP_CODE_040114.shx"

regions_shape

nyc_map = gpd.read_file(regions_shape)

type(nyc_map)

nyc_map.head(5)

nyc_map.dtypes

fig, ax = plt.subplots(figsize = (10,8))

# First create the map for the urban share

nyc_map.plot(ax = ax, # So the geopandas has a built in plot feature, we just pass our "ax to it
             edgecolor='tab:grey', # Tell it the edge color
             alpha = 0.55) # Transparent

plt.show()

fig, ax = plt.subplots(figsize = (10,8))

# First create the map for the urban share

nyc_map.plot(ax = ax, edgecolor='tab:grey',
             column='POPULATION', # THIS IS NEW, it says color it based on this column
             cmap='OrRd', # This is the color map scheme https://matplotlib.org/examples/color/colormaps_reference.html
             alpha = 0.75)

plt.show()

nyc_map.ZIPCODE = nyc_map.ZIPCODE.astype(int) # we want these to look like numbers

nyc_zips = nyc_map.ZIPCODE.tolist() # Create a list

nyc_zips = "".join(str(nyc_zips)) # turn the list into a string and join everything

nyc_zips = nyc_zips[1:-1] # Take the brakets off

my_api_key = '34e40301bda77077e24c859c6c6c0b721ad73fc7'
# This is my api_key

c = Census(my_api_key)
# This will create an object c which has methods associated with it.

code = ("NAME","B19013_001E", "B01001_001E") 
# median houshold income and population
    
zip_nyc = pd.DataFrame(c.acs5.get(code, 
                                         {'for': 'zip code tabulation area:' + nyc_zips }, year=2015))

# Then noteice what 'zip code tabulation area:' + nyc_zips does, it creats a big long string
# for which we can pass to the api and grab the data for each location

zip_nyc.info()

zip_nyc.rename(columns={code[1]:"Income", code[2]: "Population"}, inplace=True)

zip_nyc['Income'] = np.log(zip_nyc["Income"].astype(float)) 

# Im going to work with log income as this will show the differences better.

zip_nyc['Population'] = zip_nyc["Population"].astype(float)

zip_nyc['ZIPCODE'] = zip_nyc['zip code tabulation area'].astype(int)

zip_nyc.info()

nyc_map.info()

nyc_map  = nyc_map.merge(zip_nyc, on='ZIPCODE', how = "left")

fig, ax = plt.subplots(figsize = (10,8))

# First create the map for the urban share

nyc_map.plot(ax = ax, edgecolor='tab:grey', column='Income', cmap='OrRd', alpha = 0.95)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

#axins.plot()

plt.show()

fig, ax = plt.subplots(figsize = (10,8))

# First create the map for the urban share

nyc_map.plot(ax = ax, edgecolor='tab:grey', column='Income', cmap='OrRd', alpha = 0.95)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

##########################################################################################################
# This is the new part. It creats an "ax wihin an ax". We generate this by the following commands.

axins = zoomed_inset_axes(ax, # The original ax
                          4, # zoom level
                          loc=2, # location
                          borderpad=2)  # space around it relative to figure

nyc_map.plot(ax = axins, column='Income', cmap='OrRd')

# Then create the map in the "insice ax" or axins. Note, that you do not
# need to keep the colering or the income, you could have the inset 
# be population or what ever else.

# then the stuff below picks the box for the inset to cover. I
# I kind of just eyballed this untill I zoomed into what I wanted

# Note the "axins" object really just works like the ax

x1, x2, y1, y2 = 975000, 987000, 190000, 210000
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

axins.set_title("Downtown NYC")
# Make a title.

mark_inset(ax, axins, loc1=3, loc2=1, fc="none", alpha = 0.5)
# This then creates the lines that marks where the inset comes from

# Make it look nice
axins.spines["right"].set_visible(False)
axins.spines["top"].set_visible(False)
axins.spines["left"].set_visible(False)
axins.spines["bottom"].set_visible(False)

#axins.Tick.remove()

axins.get_xaxis().set_visible(False)
axins.get_yaxis().set_visible(False)



plt.show()

cwd = os.getcwd()

regions_shape = cwd + "\\shape_files\\UScounties\\cb_2017_us_county_500k.shx"

us_map = gpd.read_file(regions_shape)

type(us_map)

us_map.head()

us_map.set_index("STATEFP", inplace = True)

us_map.drop(["02","03","15","43","14","79","78","72","69","60","66"], inplace = True)

us_map.index.unique()

fig, ax = plt.subplots(figsize = (13,8))

# First create the map for the urban share

us_map.plot(ax = ax, edgecolor='tab:grey', alpha = 0.5)

plt.show()



