get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import geopandas as gpd

from shapely.geometry import Polygon
poly_1 = Polygon([ (0,0), (0,10), (10, 10), (10, 0) ] )
poly_2 = Polygon([ (10,0), (10,10), (20, 10), (20, 0) ] )
poly_3 = Polygon([ (20,0), (20,10), (30, 10), (30, 0) ] )

polys = gpd.GeoSeries([poly_1, poly_2, poly_3])
polys.plot(edgecolor='k')

polys

type(polys)

polys.bounds

polys.total_bounds

from shapely.geometry import Point
p_1 = Point(15, 5)
p_2 = Point(25, 9)
points = gpd.GeoSeries([p_1, p_2])
points.plot()

polys.contains(p_1)

polys.contains(p_2)

ax = plt.gca()
polys.plot(ax=ax, edgecolor='k')
points.plot(ax=ax, edgecolor='r', facecolor='r')
plt.show()

polys.contains(points)

points = gpd.GeoSeries([Point(5,5), Point(15, 6), Point([25,9])])
polys.contains(points)

points = gpd.GeoSeries([Point(5,5), Point(25, 9), Point([15,6])])
polys.contains(points)

polys_df = gpd.GeoDataFrame({'names': ['west', 'central', 'east'], 'geometry': polys})
polys_df

polys_df['Unemployment'] = [ 7.8, 5.3, 8.2]
polys_df

polys_df[polys_df['Unemployment']>6.0]

polys_df.geometry

points = gpd.GeoSeries([Point(5,5), Point(15, 6), Point([25,9])])
polys_df['points'] = points
polys_df.geometry

polys_df

polys_df.plot(edgecolor='k')

polys_df = polys_df.set_geometry('points')
polys_df.plot()

polys_df.geometry

tracts_df = gpd.read_file('data/california_tracts.shp')

tracts_df.head()

tracts_df.shape

tracts_df.plot()

tracts_df.crs

tracts_df.columns

clinics_df = gpd.read_file('data/behavioralHealth.shp')

clinics_df.plot()

clinics_df.columns

clinics_df.shape

clinics_df['geometry'].head()

riverside_tracts = tracts_df[tracts_df['GEOID10'].str.match("^06065")]

riverside_tracts.plot()

clinics_df.plot()

clinics_df.to_crs(riverside_tracts.crs).plot()

# convert crs of clinics to match that of tracts
clinics_df = clinics_df.to_crs(riverside_tracts.crs)

clinics_df.plot()

clinics_tracts = gpd.sjoin(clinics_df, riverside_tracts, op='within')

clinics_tracts.head()

clinics_tracts.shape

clinics_df.columns

clinics_tracts.columns

# GEOID10 is now attached to each clinic (i.e., tract identifier)

clinics_tracts[['GEOID10', 'index_right']].groupby('GEOID10').agg('count')

clinics_tracts.groupby(['GEOID10']).size()

clinics_tracts.groupby(['GEOID10']).size().reset_index(name='clinics')

twc = clinics_tracts.groupby(['GEOID10']).size().reset_index(name='clinics')

twc.plot()

riverside_tracts_clinics = riverside_tracts.merge(twc, how='left', on='GEOID10')

riverside_tracts_clinics.head()

riverside_tracts_clinics.fillna(value=0, inplace=True)

riverside_tracts_clinics.head()

riverside_tracts_clinics['clinics'].sum()

# save to a new shapefile
riverside_tracts_clinics.to_file('data/clinics.shp')

