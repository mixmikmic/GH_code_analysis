get_ipython().run_line_magic('matplotlib', 'inline')

from __future__ import (absolute_import, division, print_function)
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
# The two statemens below are used mainly to set up a plotting
# default style that's better than the default from Matplotlib 1.x
# Matplotlib 2.0 supposedly has better default styles.
import seaborn as sns
plt.style.use('bmh')

from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame

data_pth = "../data"

mpl.__version__, pd.__version__, gpd.__version__

from shapely.wkt import loads

GeoSeries([loads('POINT(1 2)'), loads('POINT(1.5 2.5)'), loads('POINT(2 3)')])

gs = GeoSeries([Point(-120, 45), Point(-121.2, 46), Point(-122.9, 47.5)])
gs

type(gs), len(gs)

gs.crs = {'init': 'epsg:4326'}

gs.plot(marker='*', color='red', markersize=100, figsize=(4, 4))
plt.xlim([-123, -119.8])
plt.ylim([44.8, 47.7]);

data = {'name': ['a', 'b', 'c'],
        'lat': [45, 46, 47.5],
        'lon': [-120, -121.2, -122.9]}

geometry = [Point(xy) for xy in zip(data['lon'], data['lat'])]
geometry

gs = GeoSeries(geometry, index=data['name'])
gs

df = pd.DataFrame(data)
df

geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
gdf = GeoDataFrame(df, geometry=geometry)

gdf.plot(marker='*', color='green', markersize=50, figsize=(3, 3));

oceans = gpd.read_file(os.path.join(data_pth, "oceans.shp"))

oceans.head()

oceans.crs

oceans.plot(cmap='Set2', figsize=(10, 10));

oceans.geom_type

# Beware that these area calculations are in degrees, which is fairly useless
oceans.geometry.area

oceans.geometry.bounds

oceans.envelope.plot(cmap='Set2', figsize=(8, 8), alpha=0.7, edgecolor='black');

oceans[oceans['Oceans'] == 'North Pacific Ocean'].plot(figsize=(8, 8));
plt.ylim([-100, 100]);

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.head(2)

world.crs

world.plot(figsize=(8, 8));

world.plot(ax=oceans.plot(cmap='Set2', figsize=(10, 10)), facecolor='gray');

f, ax = plt.subplots(1, figsize=(12, 6))
ax.set_title('Countries and Ocean Basins')
# Other nice categorical color maps (cmap) include 'Set2' and 'Set3'
oceans.plot(ax=ax, cmap='Paired')
world.plot(ax=ax, facecolor='lightgray', edgecolor='gray')
ax.set_ylim([-90, 90])
ax.set_axis_off()
plt.axis('equal');

import json
import psycopg2

with open(os.path.join(data_pth, "db.json")) as f:
    db_conn_dict = json.load(f)

conn = psycopg2.connect(**db_conn_dict)

db_conn_dict['user'] = '*****'
db_conn_dict['password'] = '*****'
db_conn_dict

seas = gpd.read_postgis("select * from world_seas", conn, 
                        geom_col='geom', crs={'init': 'epsg:4326'}, 
                        coerce_float=False)

conn.close()

seas.head()

# The geopandas plot method doesn't currently support the matplotlib legend location parameter,
# so we can't control the legend location w/o using additional matplotlib machinery
seas.plot(column='oceans', categorical=True, legend=True, figsize=(14, 6));

seas_na_arealt1000 = seas[(seas['oceans'] == 'North Atlantic Ocean') 
                          & (seas.geometry.area < 1000)]

seas_na_arealt1000.plot(ax=world.plot(facecolor='lightgray', figsize=(8, 8)), 
                        cmap='Paired', edgecolor='black')

# Use the bounds geometry attribute to set a nice
# geographical extent for the plot, based on the filtered GeoDataFrame
bounds = seas_na_arealt1000.geometry.bounds

plt.xlim([bounds.minx.min()-5, bounds.maxx.max()+5])
plt.ylim([bounds.miny.min()-5, bounds.maxy.max()+5]);

seas_na_arealt1000.to_file(os.path.join(data_pth, "seas_na_arealt1000.shp"))

import requests
import geojson

wfs_url = "http://data.nanoos.org/geoserver/ows"
params = dict(service='WFS', version='1.0.0', request='GetFeature',
              typeName='oa:goaoninv', outputFormat='json')

r = requests.get(wfs_url, params=params)
wfs_geo = geojson.loads(r.content)

print(type(wfs_geo))
print(wfs_geo.keys())
print(len(wfs_geo.__geo_interface__['features']))

wfs_gdf = GeoDataFrame.from_features(wfs_geo)

wfs_gdf.plot(ax=world.plot(cmap='Set3', figsize=(10, 6)),
             marker='o', color='red', markersize=15);

wfs_gdf.iloc[-1]

