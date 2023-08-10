get_ipython().run_line_magic('matplotlib', 'inline')
from shapely.geometry import Point, Polygon
import geopandas as gpd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
import rtree

mpl.__version__, pd.__version__, gpd.__version__

data_path = "../big_data_leave"

file = "/Harvey_FEMA_HCAD_Damage_reduced.geojson"
filePath = data_path+file
df_r = gpd.read_file(filePath)

print(type(df_r))

df_r.head()

shapeFileWatersheds = "/HCFCD_TSARP_WATERSHEDS/"
filePathWatersheds = data_path+shapeFileWatersheds
df_ws = gpd.read_file(filePathWatersheds)

df_ws

df_ws[geometry]

df_ws.plot()

type(df_ws)

df_ws.crs

df_ws = df_ws.to_crs({'init': 'epsg:4326'})

df_ws.crs

df_r.crs



properties_in_watershed = gpd.sjoin(df_r, df_ws, how="inner", op='intersects')
properties_in_watershed.head()

properties_in_watershed.plot()

type(properties_in_watershed)

properties_in_watershed

properties_in_watershed.describe()

reduced_ws_out = data_path+"/Harvey_FEMA_HCAD_Damage_reduced_ws.geojson"

properties_in_watershed.to_file(reduced_ws_out, driver='GeoJSON')



ws_out = data_path+"/HarrisCounty_Watersheds_poly.geojson"

df_ws.to_file(ws_out, driver='GeoJSON')





fig, ax = plt.subplots(1, figsize=(3.5,7))
base = df_ws(ax=ax, color='gray')
properties_in_watershed.plot(ax=base, marker="o", mfc="yellow", markersize=5, markeredgecolor="black", alpha=0.5)
_ = ax.axis('off')
ax.set_title("Buildings Likely Damaged in Harvey vs Watersheds Harris County")

df_ws.plot()



