get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import geopandas as gpd

routes_df = gpd.read_file('data/Truck_Route_Network.shp')

routes_df.plot()

routes_df.head()

routes_df['geometry'].head()

routes_df.crs

tracts_df = gpd.read_file('data/clinics.shp')

tracts_df.head()

tracts_df.plot()

tracts_df.plot(edgecolor='k')

tracts_df['dummy'] = 1.0
county = tracts_df.dissolve(by='dummy')

county.plot()

county.shape

county_uu = tracts_df['geometry'].unary_union
county_uu

r = routes_df['geometry']

type(r)

r.apply(lambda x: x.intersects(county.iloc[0]['geometry']))

rc_routes = r[r.apply(lambda x: x.intersects(county.iloc[0]['geometry']))]

rc_routes.shape

rc_routes.plot()

ax = plt.gca()
rc_routes.plot(ax=ax, edgecolor='k')
county.plot(ax=ax)
plt.show()

geoms = []
for idx, route in enumerate(rc_routes):
    print(idx)
    geoms.append(route.intersection(county.iloc[0]['geometry']))

rc_hw = gpd.GeoSeries(geoms)
rc_hw.plot()


ax = plt.gca()
county.plot(ax=ax)
rc_hw.plot(ax=ax, edgecolor='k')
ax.set_xlim(-118.0, -114.0); ax.set_ylim(33.25, 34.25)
ax.set_aspect('equal')
plt.show()

plt.rcParams['figure.figsize'] = (10, 8)
ax = plt.gca()
county.plot(ax=ax)
rc_hw.plot(ax=ax, edgecolor='k')
ax.set_xlim(-118.0, -114.0); ax.set_ylim(33.25, 34.25)
ax.set_aspect('equal')
plt.show()

plt.rcParams['figure.figsize'] = (12, 10)
ax = plt.gca()
tracts_df.plot(ax=ax, edgecolor='grey', alpha=0.2)
rc_hw.plot(ax=ax, edgecolor='k')
ax.set_xlim(-118.0, -114.0); ax.set_ylim(33.25, 34.25)
ax.set_aspect('equal')
plt.show()

type(rc_hw)

rc_hw = gpd.GeoDataFrame({'geometry': rc_hw})

rc_hw.shape

tracts_df.shape

# spatial join, tracts with roads
tracts_with_roads = gpd.sjoin(tracts_df, rc_hw, how='inner', op='intersects')

tracts_df.crs

rc_hw.crs

rc_hw.crs = tracts_df.crs # create a crs for the rc_hw
rc_hw = rc_hw.to_crs(tracts_df.crs) # update the coordinates accordingly

# spatial join, tracts with roads
tracts_with_roads = gpd.sjoin(tracts_df, rc_hw, how='inner', op='intersects')

tracts_with_roads.head()

tracts_with_roads.plot(edgecolor='grey', alpha=0.2)

tracts_with_roads.shape

tracts_df.shape

# spatial join, tracts with roads
tracts_with_roads = gpd.sjoin(tracts_df, rc_hw, how='left', op='intersects')

tracts_with_roads.shape

## 'how=left' uses keys from left_df and retains left_df geometry
# shows all tracts with or withing intersection with network
tracts_with_roads.plot(edgecolor='grey', alpha=0.2)

tracts_with_roads.head()

len(tracts_with_roads['GEOID10'].unique())

tracts_with_roads.groupby(['GEOID10']).size()

tracts_with_roads[tracts_with_roads['GEOID10']=='06065030502']

# spatial join, tracts with roads
tracts_with_roads = gpd.sjoin(tracts_df, rc_hw, how='right', op='intersects')

tracts_with_roads.shape

## 'how=right' uses keys from right DataFrame and retains right df geometry
tracts_with_roads.plot(edgecolor='grey', alpha=0.2)

tracts_with_roads = gpd.sjoin(tracts_df, rc_hw, how='inner', op='intersects')

tracts_with_roads.shape

# Let's create an indicator (dummy) variable for use later
import numpy as np
geoids = tracts_df['GEOID10'].values
tract_hw = np.array([geoid in tracts_with_roads['GEOID10'].values for geoid in geoids])

tract_hw

tracts_df['intersectshw'] = tract_hw*1.

tracts_df.plot()

tracts_df.plot(column='intersectshw')

plt.rcParams['figure.figsize'] = (12, 10)
ax = plt.gca()
tracts_df.plot(ax=ax, column='intersectshw',edgecolor='grey', alpha=0.2)
rc_hw.plot(ax=ax, edgecolor='k')
ax.set_xlim(-118.0, -114.0); ax.set_ylim(33.25, 34.25)
ax.set_aspect('equal')
plt.show()

# save our work to an augmented shapefile
tracts_df.to_file('data/tracts_routes.shp')

city = gpd.read_file('data/riverside_city.shp')

city.plot()

city_tracts = gpd.sjoin(tracts_df, city, how='inner', op='intersects')

city_tracts.head()

city_tracts.shape

city_tracts.plot(edgecolor='grey',facecolor='white')

city_tracts.plot(column='intersectshw', edgecolor='grey')

city_tracts.head()

plt.rcParams['figure.figsize'] = (12, 10)
ax = plt.gca()
city_tracts.plot(ax=ax, column='intersectshw',edgecolor='grey', alpha=0.2)
rc_hw.plot(ax=ax, edgecolor='k')
ax.set_xlim(-118.0, -114.0); ax.set_ylim(33.25, 34.25)
ax.set_aspect('equal')
plt.show()

w, s, e, n = city_tracts.total_bounds
w, s, e, n

plt.rcParams['figure.figsize'] = (12, 10)
ax = plt.gca()
city_tracts.plot(ax=ax, column='intersectshw',edgecolor='grey', alpha=0.2)
rc_hw.plot(ax=ax, edgecolor='k')
ax.set_xlim(w, e); ax.set_ylim(s, n)
#ax.set_aspect('equal')
plt.show()

plt.rcParams['figure.figsize'] = (12, 10)
ax = plt.gca()
city_tracts.plot(ax=ax, column='intersectshw',edgecolor='grey', alpha=0.2)
rc_hw.plot(ax=ax, edgecolor='k')
ax.set_xlim(-117.53, -117.37); ax.set_ylim(33.875, 33.975)
#ax.set_aspect('equal')
plt.show()

tracts_df.crs

clinics = gpd.read_file('data/behavioralHealth.shp')

clinics.crs

city_tracts = city_tracts.to_crs(clinics.crs)

city_tracts.plot()

rc_hw.plot()

type(rc_hw)

rc_hw = rc_hw.to_crs(city_tracts.crs)

rc_hw.plot()

buf = rc_hw.buffer(500)

buf.plot()

rc_hw.columns

city_tracts.columns

city_hw = gpd.sjoin(routes_df, city, how='inner', op ='intersects')

city_hw.plot()

city_hw = city_hw.to_crs(city_tracts.crs)

city_hw.plot()

b500 = city_hw.buffer(500)

b500.plot()

ct = city_tracts[['GEOID10', 'geometry']]
b500 = gpd.GeoDataFrame({'geometry': b500})
b500.crs = ct.crs

tracts_intersecting_hw = gpd.sjoin(ct, b500, how='inner', op='intersects')

tracts_intersecting_hw.plot()

tracts_intersecting_hw.shape

geoids = city_tracts['GEOID10'].values
tract_hw = np.array([geoid in tracts_intersecting_hw['GEOID10'].values for geoid in geoids])
tract_hw

city_tracts['b500'] = tract_hw * 1

city_tracts.plot()

city_tracts.plot(column='b500',edgecolor='grey')

w, s, e, n = city_tracts.total_bounds
plt.rcParams['figure.figsize'] = (12, 10)
ax = plt.gca()
city_tracts.plot(ax=ax, column='intersectshw',edgecolor='grey', alpha=0.2)
b500.plot(ax=ax, edgecolor='k')
ax.set_xlim(w, e); ax.set_ylim(s, n)
#ax.set_aspect('equal')
plt.show()

w, s, e, n = city_tracts.total_bounds
plt.rcParams['figure.figsize'] = (12, 10)
ax = plt.gca()
city_tracts.plot(ax=ax, column='b500',edgecolor='grey', alpha=0.2)
b500.plot(ax=ax, edgecolor='k')
ax.set_xlim(w, e); ax.set_ylim(s, n)
#ax.set_aspect('equal')
plt.show()

city_tracts.to_file('data/city_tracts.shp')
b500.to_file('data/b500.shp')

