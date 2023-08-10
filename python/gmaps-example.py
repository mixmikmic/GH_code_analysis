import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')

ais = pd.read_csv('../bigdata/ais-observations-1-week-20180315.csv', parse_dates = ['timestamp'])
ais = ais.sort_values(by=['timestamp'])

import gmaps
import gmaps.datasets
gmaps.configure(api_key="...")

df = ais[0:500000]
geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
df = df.drop(['lon', 'lat'], axis=1)
crs = {'init': 'epsg:4326'}
gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

raahe_port = gpd.read_file('../data/raahe_poly.shp')
raahe_port.plot()

raahe_ais = gpd.sjoin(gdf, raahe_port, how='inner', op='intersects')

fig = gmaps.figure()
fig.add_layer(gmaps.heatmap_layer(raahe_ais['geometry'].apply(lambda p: [p.y, p.x])))
fig



