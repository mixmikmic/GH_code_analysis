import pandas as pd
import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import shapely.geometry

#%% Import data into pandas
df = pd.read_csv(filepath_or_buffer='nsidc-0497/RonneFM_fractures.txt', sep=' ', header=None)
headers = ['Longitude', 'Latitude', 'GridX', 'GridY', 'SetNumber', 'GroupNumber']
df = df.rename(columns={i:v for i,v in enumerate(headers)})
df.head()

#%% Get data into geopandas Geoseries format
sets = df.GroupNumber.unique()  #get list of group numbers e.g. [1,2,3,...,1169]
series = []
for s in sets:
    linecoords = [(x, y) for x, y in zip(df[df['GroupNumber']==s].GridX, df[df['GroupNumber']==s].GridY)]
    try:
        assert(len(linecoords) != 1)  #make sure that we have more than one point
    except AssertionError:
        print('Skipping group {0} as only one point! Cannot build line'.format(s))
        continue
    linestring = shapely.geometry.LineString(coordinates=linecoords)
    series.append(linestring)
assert(len(series) <= len(sets))

#%% Create GeoDataFrame out of GeoSeries
gs = gpd.GeoSeries(series)
gs.crs = {'init': 'epsg:3031'}
gdf = gpd.GeoDataFrame(pd.DataFrame(sets, columns=['index']), geometry=gs)
gdf.head()

#%% Have a look at our Ronne ice shelf fractures using matplotlib
gdf.plot(figsize=(10,10))

#%% Export GeoDataFrame to GeoJson format!
assert('GeoJSON' in fiona.supported_drivers)
if not os.path.exists('model_data'):
    os.makedirs('model_data')
filename = 'model_data/RonneFM_fractures.geojson'
try:
    gdf.to_file(filename=filename, driver='GeoJSON')
except:
    pass
finally:
    print('Exported data to:', filename)



