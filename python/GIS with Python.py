# Create a polygon from 3 points

from shapely.geometry import Polygon, Point

polygon = Polygon([(45.55, -73.62), (45.54, -73.62), (45.55, -73.60)])
polygon

# Show three attributes of the polygon

print("Area: " + str( polygon.area ))
print("Length: " + str( polygon.length ))
print("Bounding box: " + str( polygon.bounds ))

# The polygon's centroid in WKT

polygon.centroid.wkt

# Create a point. Is it inside the polygon?

point = Point(45.545, -73.613)
polygon.contains(point)

# Make another point, create a buffer on both.
# Show the difference between the two buffers

area1 = point.buffer(0.3)
area2 = Point(45.6, -73.7).buffer(0.3)
area1.difference(area2)

# Show where they intersect

print(area2.intersection(area1).area)
area2.intersection(area1)

# Let's look at MultiPoints

from shapely.geometry import MultiPoint

p = [(45.65, -73.63), (45.70, -73.60), (45.35, -73.55), (45.75, -73.42)]

points = MultiPoint(p)
points

# Find the smallest polygon that contains all the points

points.convex_hull

# Do the points fall within the polygon we created earlier?

points.within(polygon)

# Open a shapefile and check its CRS/projection

import fiona

shapefile = '../../Map files/Montreal/Addresses/adresse/ADRESSE.shp'

f = fiona.open(shapefile)
f.crs

# Read information about attributes

f.schema

# Inspect one record

f[0]

# Fiona can do very basic geometry, like checking the bounds

f.bounds

# Example of a geocoding script using Montreal's address point data.
# It checks if an address matches a fiona feature by iterating through all features.
# Not the most efficient script, but I code like a journalist.

import time

# Function to check if street number is even to get coordinates
# on the right side of the street
def is_even(n):
    if n % 2 == 0: return True
    else: return False

start_time = time.time()

address = '4416 rue Garnier'.split(' ')

even_number = is_even(int(address[0]))

# Iterate thorugh fiona features and print the matching address and coordinates
for item in f.items():
    if ( even_number == is_even(item[1]['properties']['ADDR_DE']) and
         int(address[0]) >= item[1]['properties']['ADDR_DE'] and
         int(address[0]) <= item[1]['properties']['ADDR_A'] and
         address[1] == item[1]['properties']['GENERIQUE'] and
         address[2] == item[1]['properties']['SPECIFIQUE'] ):
        print(item[1]['properties']['ADD_COMPL'], item[1]['geometry']['coordinates'])
        break
        
print ("Lookup time: " + str(time.time() - start_time) + " seconds")



import geopandas

# Open a shapefile of Montreal's boroughs and municipalities
# Notice the geometry column, encoded in WKT

shapefile = shapefile = '../Map files/Montreal/Boroughs/LIMADMIN.shp'

boroughs = geopandas.read_file(shapefile)
boroughs.head()

# Quickly see what it looks like

get_ipython().magic('matplotlib inline')

boroughs.plot(figsize=(15,9))

# Make a choropleth map based on the perimeter column

boroughs.plot(column='PERIM', 
           scheme='quantiles', 
           k=5, 
           cmap='YlGn', 
           figsize=(13,7))

# Load Montreal fire dept. intervention CSV with lat/lon columns
# We'll use Shapely to create geometry column

import pandas as pd

# Load data into a pandas dataframe
fire = pd.read_csv('Fire/donneesouvertes-interventions-sim-2016.csv')

# Convert datetime column to something pandas understands
fire['CREATION_DATE_TIME'] = pd.to_datetime(fire['CREATION_DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

# Filter the last month of 2016
fire = fire[fire.CREATION_DATE_TIME.dt.month > 11]

# Use Shapely's Point to create a geometry column
fire['geometry'] = fire.apply(lambda row: Point(row["LONGITUDE"], 
                                                row["LATITUDE"]), 
                              axis=1)

# Load into geopandas
geofire = geopandas.GeoDataFrame(fire)
geofire.plot(figsize=(13, 7))

# Do a spatial join (sjoin) on boroughs polygons and fire intervention points.
# Both datasets must have the same CRS (projection) for it to work.
# In this case, I know they do, so I assign the CRS from one to the other.
# The result is a geodataframe of all interventions with the attributes of the boroughs they are in

geofire.crs = boroughs.crs
per_borough = geopandas.tools.sjoin(geofire, 
                                    boroughs, 
                                    how='left', 
                                    op='within')
per_borough.head()

# Now we count the number of interventions per borough

# First, group the geodatagrame by borough name (NOM),
# count the number of records in each,
# keep only relevant columns.
# Column with point-in-polygon count is named NUM_y.

borough_count = per_borough.groupby('NOM').count()[['NUM', 'CASERNE']].reset_index()

# Merge (join) with the boroughs shapefile and plot it
boroughs_with_fire = boroughs.merge(borough_count, on='NOM')
boroughs_with_fire.plot(column='NUM_y', 
           scheme='quantiles', 
           k=5, 
           cmap='OrRd', 
           figsize=(13,7))



# Draw a basic world map in the Robinson projection

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
get_ipython().magic('matplotlib inline')

plt.figure(figsize=(12,6))
m = Basemap(projection='robin', lon_0=0, resolution='l')
m.drawcountries(linewidth=0.25)
m.drawcoastlines(linewidth=0.25)

# This long code is a function that creates a map in PNG format for every timestamp
# in scraped car2go API data. It loads several layers of shapefiles and iterates through
# a pandas dataframe grouped by time.

import pandas as pd
from IPython.display import clear_output

def make_map_slides(from_date, to_date):
    '''
    Create a map for every time stamp between two dates. 
    '''
    
    days = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
    
    font = {'family': 'Arial',
        'color':  '#323F44',
        }
    
    # Map bounds zoomed in on Montreal
    coords = [-73.773869, 45.428756, -73.477991, 45.619950 ]
    w, h = coords[2] - coords[0], coords[3] - coords[1]
    extra = 0.01

    # df is a pandas dataframe with car2go API data at 5-minute intervals.
    # Group the data by timestamp to iterate through each one
    grouped = (df.set_index('time')
               .loc[from_date:to_date]
               .reset_index()
               .groupby('time') )
    
    n = 0
    for name, group in grouped:
        lats = group.lat.tolist()
        lons = group.lon.tolist()   
        
        # Initialize basemap
        m = Basemap(
            projection='tmerc',
            lon_0= -73.6,
            lat_0= 45.5,
            ellps = 'WGS84',
            llcrnrlon = coords[0] - extra * w,
            llcrnrlat = coords[1] - extra + 0.01 * h,
            urcrnrlon = coords[2] + extra * w,
            urcrnrlat = coords[3] + extra + 0.01 * h,
            lat_ts = 0,
            resolution = 'i',
            suppress_ticks = True)

        plt.figure(figsize=[14, 14])

        # Boroughs shapefile
        boroughs = "/Users/usr/Documents/Map files/Montreal/boroughs_and_munis/boroughs_and_munis-NAD83"
        m.readshapefile(
            boroughs,
            'montreal',
            color='firebrick',
            zorder=2,
            linewidth=2)

        # Main road arteries shapefile
        arteries = "/Users/usr/Documents/Map files/Montreal/geobasemtlshp/GEOBASE_MTL_arteries"
        m.readshapefile(
            arteries,
            'arteries',
            color='grey',
            zorder=1)
    
        # Plot the latitude and longitude points for each timestamp
        x, y = m(lons, lats)
        m.plot(x, y, 'bo', markersize=3)

        # Text of timestamp
        weekday = name.dayofweek
        day = days[weekday]
        time = name.strftime('%Y-%m-%d %H:%M')

        # Location of timestamp text on figure
        tx, ty = m(-73.755, 45.56)
        wx, wy = m(-73.755, 45.565)
        plt.text(tx, ty, time, fontdict=font, fontsize=16)
        plt.text(wx, wy, day, fontdict=font, fontsize=18, fontweight='bold')
        
        # Save figure as a PNG with name format 'frame0001.png'
        plt.savefig('animate/en/frame{}.png'.format(str(n).zfill(4)), bbox_inches='tight')
        
        plt.close()
        clear_output()
        print("Creating file {}".format(n))
        n += 1

# Run the mapmaker function between two timestamps.
# Output below is the map of the first timestamp.

make_map_slides('2016-07-28 00:04:55','2016-07-30 23:56:01')

from folium import Map, CircleMarker, Popup

# Create blank map using the CartoDB Positron basemap
mymap = Map( location=[45.5, -73.6], 
            zoom_start=12, 
            tiles="cartodbpositron")

mymap 

# Note: Folium maps don't show up on GitHub. Download this notebook and try it on your machine.

# Add a circle marker
CircleMarker( location = [45.5, -73.6], 
            radius = 200, 
            popup = Popup('I\'m a marker!'),
            fill_color = 'maroon',
            fill_opacity = 0.7, 
            color=None
        ).add_to(mymap)

mymap

# Create a vincent chart with fake data. Add it to a new marker. 
# Vega is a folium class that turns vincent charts into popup content

import pandas as pd
import numpy as np
from folium import Vega
from vincent import Bar

# Make fake data with numpy
data = pd.DataFrame(np.random.randint(0,100,size=(14, 2)), 
                    columns=list('AB'))

# This makes a vincent chart
chart = ( Bar(data, width=200,  height=100)
        .colors(range_=['#404040'])
        .axis_titles(x='A', y='B')
        )
chart.x_axis_properties(color='#9E9E9E', title_size=11)
chart.y_axis_properties(color='#9E9E9E', title_size=11)

# This turns that chart into a JSON item that folium can use in a popup
vega = Vega(chart.to_json(),
             width = vega.width+50, 
             height = vega.height+50 )

# Create a circle marker with the vega chart as the popup content
CircleMarker(
        location = [45.53, -73.64], 
        popup = Popup(max_width=250).add_child(vega), 
        radius = 250, 
        fill_color = 'blue', 
        fill_opacity = 0.8, 
        color = None
    ).add_to(mymap)

# Ta-da!

mymap

# Save the folium map

mymap.save('map.html')



from pysal.esda.mapclassify import Natural_Breaks, Quantiles, Equal_Interval
import numpy as np

# Create fake data in exponential distribution and break it into 6 classes

data = np.random.exponential(3000, 200)

print("Natural breaks:")
print(Natural_Breaks( data, initial=200, k = 6).bins.tolist() )
print("")
      
print("Quantiles:")
print(Quantiles( data, k = 6).bins.tolist() )
print("")

print("Equal interval:")
print(Equal_Interval( data, k = 6).bins.tolist() )
print("")



