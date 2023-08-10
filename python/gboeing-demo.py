import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.wkt import loads

# change for different ref location for geometries csv
filepath = '../example_geometries.csv'

# set this so that you have ~2-4 partitions per worker
# e.g. 60 machines each with 4 workers should have ~500 partitions
nparts = 4
head_count = 500 # set to a # if you want to just do a subset of the total for faster ops, else None

# get original csv as pandas dataframe
pdf = pd.read_csv(filepath)[['id', 'geometry']].reset_index(drop=True)

# convert to geopandas dataframe
geometries = gpd.GeoSeries(pdf['geometry'].map(lambda x: loads(x)))
crs = {'init': 'epsg:32154'},
gdf = gpd.GeoDataFrame(data=pdf[['id']], crs=crs, geometry=geometries)

# trim if desired
if head_count is not None:
    gdf = gdf.head(head_count)
print('Working with a dataframe of length {}.'.format(len(gdf)))

# clean the ids column
gdf = gdf.drop('id', axis=1)
gdf['id'] = gdf.index
gdf['id'] = gdf['id'].astype(int)

# we need some generic column to perform the many-to-many join on
gdf = gdf.assign(tmp_key=0)

# then convert into a dask dataframe
gdf1 = gdf.copy()
ddf = dd.from_pandas(gdf1, name='ddf', npartitions=nparts)

def calc_distances(grouped_result):
    # we just need one geometry from the left side because
    first_row = grouped_result.iloc[0]
    from_geom = first_row['geometry_from'] # a shapely object

    # and then convert to a GeoSeries
    to_geoms = gpd.GeoSeries(grouped_result['geometry_to'])

    # get an array of distances from the GeoSeries comparison
    distances = to_geoms.distance(from_geom)
    return distances.values

get_ipython().run_cell_magic('time', '', "\n# use dask to calculate distance matrix with geopandas\ntall_list = (dd.merge(ddf, gdf, on='tmp_key', suffixes=('_from', '_to'), npartitions=nparts).drop('tmp_key', axis=1))\ndistances = (tall_list.groupby('id_from').apply(calc_distances, meta=pd.Series()))\ncomputed = distances.compute()")

# show results
pd.Series(computed[0][:5])

gdf2 = gdf.copy()

get_ipython().run_cell_magic('time', '', "\n# convert polygons into xy centroids\ncentroids = gdf2.centroid\ngdf2['x'] = centroids.map(lambda coords: coords.x)\ngdf2['y'] = centroids.map(lambda coords: coords.y)\ngdf2.drop('geometry', axis='columns', inplace=True) # makes merge faster and more memory efficient\n\n# create OD pairs by a many-to-many merge and index by from/to keys\ngdf_od = pd.merge(gdf2, gdf2, on='tmp_key', suffixes=('_from', '_to')).drop('tmp_key', axis=1)\ngdf_od = gdf_od.set_index(['id_from', 'id_to'])\n\n# calculate euclidean distance matrix, vectorized\nx1 = gdf_od['x_from']\nx2 = gdf_od['x_to']\ny1 = gdf_od['y_from']\ny2 = gdf_od['y_to']\ndist_matrix = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)")

print(head_count ** 2)
print(len(dist_matrix))
dist_matrix.head()

gdf3 = gdf.copy()

get_ipython().run_cell_magic('time', '', "\n# convert polygons into xy centroids\ncentroids = gdf3.centroid\ngdf3['x'] = centroids.map(lambda coords: coords.x)\ngdf3['y'] = centroids.map(lambda coords: coords.y)\ngdf3.drop('geometry', axis='columns', inplace=True) # makes merge faster and more memory efficient\n\n# create a dask dataframe of OD pairs\nddf = dd.from_pandas(gdf3, name='ddf', npartitions=nparts)\nddf_od = dd.merge(ddf, gdf3, on='tmp_key', suffixes=('_from', '_to'), npartitions=nparts).drop('tmp_key', axis=1)\n\n# calculate euclidean distance matrix, vectorized and with dask series\nx1 = ddf_od['x_from']\nx2 = ddf_od['x_to']\ny1 = ddf_od['y_from']\ny2 = ddf_od['y_to']\neuclidean_distances = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5\ndist_matrix = euclidean_distances.compute()")

print(head_count ** 2)
print(len(dist_matrix))
dist_matrix.head()



