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

file_r_ws = "/Harvey_FEMA_HCAD_Damage_reduced_ws.geojson"
filePath_r_ws = data_path+file_r_ws
df_r_ws = gpd.read_file(filePath_r_ws)

print(type(df_r_ws))

df_r_ws.head()

















c.crs





watersheds = df_r_ws.WTSHNAME.unique()

df_r_ws_WHITEOAKBAYOU = df_r_ws[df_r_ws['WTSHNAME'].str.contains("WHITE OAK BAYOU")]

df_r_ws_WHITEOAKBAYOU.plot()

print(watersheds)

watersheds[3]  

for i in watersheds:
    p = i.replace(" ","_")
    print(p)

print(watersheds)

#### This function takes in a geojson, an array of strings, and a column header and makes excepts. 
#### Each is a new geojson whose rows for the given column header match one of the strings in the array of strings. 
#### If there are 5 strings in the string array and all exist in the geojson at that column header, then 5 geojsons should be produced. 
def makegeojsonAttributeExcerpts(geojson,watershedsArray,colHeader):
    data_path = "../big_data_leave"
    for i in watershedsArray:
        watershedFileName = i.replace(" ","_")
        watershedFileName = watershedFileName.replace("&","_")
        geojson_holder = geojson[geojson[colHeader].str.contains(i)]
        ws_out = data_path+"/HarrisCnty_FEMAdamHCADwatershed_"+watershedFileName+".geojson"
        geojson_holder.to_file(ws_out, driver='GeoJSON')

### NOTE: If you call this more than once and try to write to the same filenames, you will see an error below. 
makegeojsonAttributeExcerpts(df_r_ws,watersheds,"WTSHNAME")











