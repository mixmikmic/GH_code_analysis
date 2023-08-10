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

from shapely.geometry import Point, LineString, Polygon

# ROAD MAP DATASET 
rd = gpd.read_file("data\\ne_10m_roads.shp")

rd.crs

rd.geometry.head(3)

# Converting from EPSG:4326 (lon/lat) to EPGS:32631 (units in meters)
rd2 = rd.to_crs(epsg=32631)

rd2.geometry.head(3)

# Importing the data
ATL = gpd.read_file("data\Housing_and_Transportation_Affordability_Type_8_Family.shp")

ATL.head(2)

# CRS type
ATL.crs

# Showing map with colorscheme based on column depicting: Median household income
ATL.plot(cmap = 'coolwarm', column = "Median_hou", figsize = (10,10), legend = True)

# Different colorschemes may emphasize data differences better: cmap = COLORSCHEME
ATL.plot(column = "Median_hou", cmap = 'cubehelix', figsize = (10,10), legend = True)

# Annual Transit Trips column
ATL.plot(cmap = 'coolwarm', column = "Type_8__An", figsize = (10,10), legend = True)

# Annual Vehicle miles traveled column
ATL.plot(cmap = 'coolwarm', column = "Tp8_Annual", figsize = (10,10), legend = True)

# Percent of housing costs of household income column 
ATL.plot(cmap = 'coolwarm', column = "Tp8_Housin", figsize = (10,10), legend = True)

# Showing only geometry column data contained in the dataset
ATL.geometry.head()

ax = rd.plot(linewidth = 1.5, column ="type", cmap = 'prism', figsize = (10,10), legend = True)
ATL.plot(ax=ax, column = "Median_hou", cmap = 'coolwarm', markersize = 5, legend = True)
ax.set(xlim=(-85.35,-83.5,), ylim=(33, 34.6))

# Importing Excel file using Pandas
apd = pd.read_excel("data\\COBRADATA2016.xlsx")
apd.head(2)

# Appending the two columns into one geometry column
geometry = [Point(xy) for xy in zip(apd.x, apd.y)]
# Dropping the x and y columns from the dataframe
apd = apd.drop(['x', 'y','apt_office_prefix', 'apt_office_num', 'dispo_code'], axis=1)
# Initializing the lat / long coordinates as WGS84
crs = {'init': 'epsg:4326'}
# Saving the GeoDataFrame in new variable 
gdf = gpd.GeoDataFrame(apd, crs=crs, geometry=geometry)

# Since the crime data does not encompass the Atlanta metro and surrounding areas, will 'zoom in' for a closer look:
ax = rd.plot(linewidth = 1.5, column ="type", cmap = 'prism', figsize = (10,10))
ATL.plot(ax=ax, column = "Median_hou", cmap = 'coolwarm', markersize = 5)
# Add crime dataset plot the base maps
# Notice: column = "COLUMN_NAME" will color code the different values in the specified column
gdf.plot(ax=ax, marker = "*", column = "crimes", markersize = 5, cmap='hot', legend = True)
# Adjust ax.set(xlim=(lat,lat), ylim=(lon,lon)) to the desired coordinates
ax.set(xlim=(-84.56,-84.25,), ylim=(33.6, 33.88))

ax = rd.plot(linewidth = 1.5, column ="type", cmap = 'prism', figsize = (10,10))
ATL.plot(ax=ax, column = "Median_hou", cmap = 'coolwarm', markersize = 5, legend = True)
gdf.plot(ax=ax, marker = "o", column = "crimes", markersize = 30, cmap='plasma', legend = True)
# Adjust coordinates to zoom in on a specific area
ax.set(xlim=(-84.465,-84.40,), ylim=(33.825, 33.89))

ax = rd.plot(linewidth = 1.5, column ="type", cmap = 'prism', figsize = (10,10))
ATL.plot(ax=ax, column = "Median_hou", cmap = 'coolwarm', markersize = 5, legend = True)
gdf.plot(ax=ax, marker = "o", column = "crimes", markersize = 30, cmap='seismic', legend = True)
ax.set(xlim=(-84.5,-84.425,), ylim=(33.74, 33.81))

# Creating layer for homicides committed, this extraction of specific data from a column using GeoPandas is similar to Pandas
gdf_h = gdf.loc[gdf['crimes'] == "HOMICIDE" ]
gdf_h.head(2)

gdf_r = gdf.loc[gdf['crimes'] == "RAPE" ]
gdf_r.head(2)

gdf_a = gdf.loc[gdf['crimes'] == "AGG ASSAULT" ]
gdf_a.head(2)

# Adding layers to higher income neighborhoods
ax = rd.plot(linewidth = 1.5, column ="type", cmap = 'prism', figsize = (10,10))
ATL.plot(ax=ax, column = "Median_hou", cmap = 'coolwarm', markersize = 5)
gdf_h.plot(ax=ax, marker = "o", column = "crimes", markersize = 30, color='r')
gdf_r.plot(ax=ax, marker = "o", column = "crimes", markersize = 30, color='k')
gdf_a.plot(ax=ax, marker = "o", column = "crimes", markersize = 30, color='b')
ax.set(xlim=(-84.465,-84.40,), ylim=(33.825, 33.89))

# Adding the same layers to the moderate income neighborhood
ax = rd.plot(linewidth = 1.5, column ="type", cmap = 'prism', figsize = (10,10))
ATL.plot(ax=ax, column = "Median_hou", cmap = 'coolwarm', markersize = 5)
gdf_h.plot(ax=ax, marker = "o", column = "crimes", markersize = 30, color='r')
gdf_r.plot(ax=ax, marker = "o", column = "crimes", markersize = 30, color='k')
gdf_a.plot(ax=ax, marker = "o", column = "crimes", markersize = 30, color='b')
ax.set(xlim=(-84.5,-84.425,), ylim=(33.74, 33.81))

ax = rd.plot(linewidth = 1.5, column ="type", cmap = 'prism', figsize = (10,10))
ATL.plot(ax=ax, column = "Median_hou", cmap = 'coolwarm', markersize = 5)
gdf_h.plot(ax=ax, marker = "o", column = "crimes", markersize = 5, color='r') # Homicides - red
gdf_r.plot(ax=ax, marker = "o", column = "crimes", markersize = 5, color='k') # Rapes - black
gdf_a.plot(ax=ax, marker = "o", column = "crimes", markersize = 5, color='b') # Aggravated Assaults - blue 
ax.set(xlim=(-84.56,-84.25,), ylim=(33.6, 33.88))

# Create new variable holding the central points of the geometry column
cents = ATL.centroid

cents.head()

