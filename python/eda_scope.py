# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import nbinteract as nbi

sns.set()
sns.set_context('talk')
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8

# HIDDEN
calls = pd.read_csv('data/calls.csv', parse_dates=['EVENTDTTM'], infer_datetime_format=True)
stops = pd.read_csv('data/stops.csv', parse_dates=[1], infer_datetime_format=True)

calls

# Shows earliest and latest dates in calls
calls['EVENTDTTM'].dt.date.sort_values()

calls['EVENTDTTM'].dt.date.max() - calls['EVENTDTTM'].dt.date.min()

import folium # Use the Folium Javascript Map Library
import folium.plugins

SF_COORDINATES = (37.87, -122.28)
sf_map = folium.Map(location=SF_COORDINATES, zoom_start=13)
locs = calls[['Latitude', 'Longitude']].astype('float').dropna().as_matrix()
heatmap = folium.plugins.HeatMap(locs.tolist(), radius = 10)
sf_map.add_child(heatmap)

stops

stops['Call Date/Time'].dt.date.sort_values()

SF_COORDINATES = (37.87, -122.28)
sf_map = folium.Map(location=SF_COORDINATES, zoom_start=13)
locs = stops[['Location - Latitude', 'Location - Longitude']].astype('float').dropna().as_matrix()
heatmap = folium.plugins.HeatMap(locs.tolist(), radius = 10)
sf_map.add_child(heatmap)

