import fiona
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from rasterstats import zonal_stats
import geopandas as gpd
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

nations = gpd.read_file('./data/TM_WORLD_BORDERS_SIMPL-0/TM_WORLD_BORDERS_SIMPL-0.3.shp')

nations.head()

aus = nations[nations.ISO3 == "AUS"]

aus.plot()

input_raster = "./data/modis_example.nc"
with rasterio.open(input_raster) as modisData:
    profile = modisData.profile
    cloudFraction = modisData.read()

plt.imshow(cloudFraction[0],interpolation=None, cmap=cm.gist_earth)

aus.to_crs(epsg="4326").plot()

output = zonal_stats(aus.to_crs(epsg="4326"), cloudFraction[0], band=1, all_touched=True,
                     raster_out=True, affine=profile['transform'], nodata=-999)

plt.imshow(output[0]['mini_raster_array'], interpolation=None, cmap=cm.gist_earth)

print(f"Pixels of data = {output[0]['count']}")
print(f"Maximum cloud fraction pixel = {output[0]['max']:5.2f}")
print(f"Min. cloud fraction pixel ={output[0]['min']:5.2f}")
print(f"Mean cloud fraction ={output[0]['min']:5.2f}")

import geopandas as gpd
from geopandas.tools import sjoin
import shapely

def point_maker(lon, lat):
    """Use shapely to craete a geometry object out of a pair of coordinates"""
    return shapely.geometry.Point(lon, lat)

points = []
rows = []
for n in range(50):
    random_lon = (np.random.random_sample() * 50) + 110        # random longitude between 110 - 160
    random_lat = ((np.random.random_sample() * 50) + 5) * -1   # random lat between -5 and -55
    points.append(point_maker(random_lon, random_lat))
    rows.append(np.random.random_sample())

series = gpd.GeoDataFrame(rows, crs={'init':'epsg:4326'}, geometry=points, columns=['random_value'])
series.head()

base = aus.plot(color='green')
series.plot(ax=base, marker='o', markersize=5)

inside_country = sjoin(series, aus, how='inner', op='intersects')
inside_country.head()

base = aus.plot(color='green')
inside_country.plot(ax=base, marker='o', markersize=5)

random_mean = inside_country.random_value.mean()
random_SEM = inside_country.random_value.std() / np.sqrt(inside_country.random_value.count() - 1)

print("Random values within country had an average of {0:4.2f}±{1:4.2f}".format(random_mean, random_SEM))

