#Importing necessary packages

import geopandas as gpd
import pandas as pd
import matplotlib.pylab as plt
import fiona
import osmnx as ox
import networkx as nx
from geopy import Nominatim

get_ipython().run_line_magic('matplotlib', 'inline')

santa_monica = ox.graph_from_place('Santa monica, Los Angeles County, California', network_type='drive')
santa_monica_projected = ox.project_graph(santa_monica)
fig, ax = ox.plot_graph(Boulder_projected)

ox.save_graph_shapefile(santa_monica_projected, filename='santa_monica')

streets_sm = gpd.read_file('data/santa_monica/edges/edges.shp')
streets_sm.head()

streets_sm.length = streets_sm.length.astype('float')

dict_v = {'length': 'sum', 'highway': 'first', 'oneway': 'first'}
table = streets_sm.groupby('name').agg(dict_v).reset_index()
table.head()

table.sort_values(by='length', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10,10))
streets_sm.plot(ax=ax, alpha=0.5)
streets_sm[streets_sm.name == 'Ocean Park Boulevard'].plot(ax=ax, color='black')
ax.set_title('Longest Drive in Santa Monica', fontsize=15)
ax.set_axis_off()

geolocator = Nominatim()
location = geolocator.geocode("Ocean Park Boulevard, Santa monica, Los Angeles County, California")
print(location.address)
print(location.raw)

table[table.length > 100].sort_values(by='length', ascending=True).head(10)

fig, ax = plt.subplots(figsize=(10,10))
streets_sm.plot(ax=ax, alpha=0.5)
streets_sm[streets_sm.name == 'Pico Place'].plot(ax=ax, color='black')
ax.set_title('Shortest Drive in Santa Monica', fontsize=15)
ax.set_axis_off()

location = geolocator.geocode("Pico Place, Santa monica, Los Angeles County, California")
print(location.address)
print(location.raw)

streets_sm.name.unique()

streets_sm.highway.value_counts()

help(ox.graph_from_place)

