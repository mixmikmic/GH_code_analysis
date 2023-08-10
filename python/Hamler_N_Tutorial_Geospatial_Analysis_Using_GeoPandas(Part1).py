from IPython.display import Image
Image(filename="data\\raster_vector.jpg") 

get_ipython().run_line_magic('matplotlib', 'inline')

from __future__ import (absolute_import, division, print_function)
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
plt.style.use('bmh')

import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
from geopandas.tools import sjoin

# Import of geometric objects for their creation and visual presentation
from shapely.geometry import Point, LineString, Polygon

# Creating geometric objects with values
point = Point(1,1)
line = LineString([(0,0),(5,9),(10,8)])
poly = line.buffer(1)

# Does the polygon contain the point?
poly.contains(point)

# Assigning points to a GeoSeries:
gs = GeoSeries([Point(-10, 15), Point(-5, 16), Point(10, 2)])
gs

# Points can now be plotted on a chart:
gs.plot(marker='*', color='red', markersize=100, figsize=(4, 4))

# One can also assign a polygon to the GeoSeries:
gs1 = GeoSeries(poly)
gs1.plot(color = 'blue', markersize = 100, figsize = (4,4))
plt.xlim([-5, 15])
plt.ylim([-5, 15]);

# Assigning the line to the GeoSeries:
gs2 = GeoSeries(line)
gs2.plot(color = 'blue', markersize = 100, figsize = (4,4))
plt.xlim([0, 15])
plt.ylim([0, 15]);

# Getting the GeoDataFrame and assigning to variable:
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.head(2)

# Plot of NaturalEarth Dataset:
world.plot(figsize=(10,20))

# Plot by GDP per capita:
world = world[(world.pop_est>0)]

world['gdp_per_cap'] = world.gdp_md_est / world.pop_est
world.plot(column='gdp_per_cap', cmap = 'pink', figsize = (10,20));

germany = gpd.read_file("data\DEU_adm0.shp")
germany

germany.geom_type.head(2)

germany.plot(cmap='plasma', figsize=(4,6));

districts = gpd.read_file("data\DEU_adm2.shp")
districts.head()

# Accessing the geometry column only is similar to how it is done in Pandas:
districts.geometry.head()

# Type of file: 
type(districts)

# Data type of geometry column:
type(districts.geometry)

# Accessing a specific value's data type within the GeoSeries:
type(districts.geometry[0])

# To get the area of the different districts:
districts.geometry.area.head()

# Assigning coordinates of point of the geographic location for Rothenburg ob der Tauber (coordinates obtained from Google)
# Variable_Name = Point(location.longitude, location.latitude)
RBG = Point(10.1867, 49.33802)
districts.contains(RBG).head()

# Obtaining specific information on the geometry object by using .contains(Point)
districts[districts.contains(RBG)]

# Obtain distance from each district to point by using .geometry.distance(var)
districts.geometry.distance(RBG).head()

districts.plot(cmap='viridis', figsize=(4,6))

# Naturalearth dataset provides us with global data which we will have to restrict to the coordinates for Germany 
pop = gpd.read_file("data\\ne_10m_populated_places.shp")
pop.head(2)

# As with the population dataset, the road map data provides us with global data
rd = gpd.read_file("data\\ne_10m_roads.shp")
rd.head(1)

# Grouping by the different continents included in the file:
rd.groupby('continent').count()

# Extracting only European roadways as would be done in Pandas:
EU = rd.loc[rd['continent'] == "Europe"]
EU.head(2)

ax = EU.plot(linewidth = 0.5, color = 'w', figsize = (4,6))
districts.plot(ax=ax, cmap = "viridis", markersize = 5)
ax.set(xlim=(5.883,15.88), ylim=(47.12, 55.84));

ax = EU.plot(linewidth = 0.5, color = 'grey', figsize = (8,10))
districts.plot(ax=ax, cmap = "viridis", markersize = 5)
pop.plot(ax=ax, linewidth = 0.25, color = 'r', markersize = 15, figsize = (8,10))
ax.set(xlim=(5.883,15.88), ylim=(47.12, 55.84));

# Instead of using param of 'color=' we will use 'column=COLUMN_NAME' to allow us to color code specific values of the datapoints
# legend = True will display the legend on the picture
ax =districts.plot( cmap='pink',  markersize = 5, edgecolor = 'grey', figsize = (10,10)) 
rd.plot(ax=ax,linewidth = 1, column = "type", cmap = 'Greys')
pop.plot(ax=ax,linewidth = 0.5, column = 'FEATURECLA', cmap= 'cool', markersize = 30, legend = True)
ax.set(xlim=(5.883,15.88), ylim=(47.12, 55.84));

