import matplotlib.pyplot as plt
import numpy as np, geopandas as gpd, pandas as pd, triangle
from descartes import PolygonPatch
from shapely.geometry import Point, MultiPolygon, MultiPoint, LineString
from shapely.ops import triangulate
from triangle import plot as tplot

poly = Point((0, 0)).buffer(100).simplify(tolerance=10)
poly

# perform a delaunay triangulation of this convex polygon
triangles = MultiPolygon(triangulate(poly))
triangles

# get the centroids of each triangle
MultiPoint([tri.centroid for tri in triangles])

# make some concave shape
circle1 = Point((0, 0)).buffer(100).simplify(10)
circle2 = Point((20, 20)).buffer(80).simplify(10)
poly = circle1.difference(circle2)
poly

# see the vertices
print(len(list(poly.exterior.coords)))
vertices = MultiPoint([Point(x, y) for x, y in list(poly.exterior.coords)])
vertices

# perform a delaunay triangulation of the polygon
# problem is polygon is concave so you get a bad triangulation
triangles = MultiPolygon(triangulate(poly))
triangles

# we need to perform a constrained delaunay triangulation to handle concave polygons
coords = list(poly.exterior.coords)
vertices = np.array(coords)
indices = list(range(len(coords)))
segments = np.array(list(zip(indices, indices[1:] + [indices[0]])))
poly_dict = {'vertices':vertices, 'segments':segments}

# flag p indicates this is a planar straight line graph, for constraining
# flag q20 indicates that triangles must have a minimum angle of 20 degrees
cndt = triangle.triangulate(tri=poly_dict, opts='pq20')

fig, ax = plt.subplots(figsize=(6,6))
tplot.plot(ax, **cndt)
plt.show()

# what percent of the envelope does this poly take up?
# this could influence how fine-meshed we make the point grid
env = poly.envelope
poly.area / env.area

# how far apart should the points be, in polygon's units
n = 15 #ie, 15 meters

# make a grid of points
left, bottom, right, top = env.bounds
x_points = np.linspace(left, right, int((right-left)/n))
y_points = np.linspace(bottom, top, int((top-bottom)/n))
xy_points = []
for x in x_points:
    for y in y_points:
        xy_points.append((x, y))

# give it a look...
points = MultiPoint([Point(x, y) for x, y in xy_points])
poly.union(points)

# now sample the points
points = gpd.GeoSeries([Point(x, y) for x, y in xy_points])
mask = points.intersects(poly)
points_sample = points[mask]
MultiPoint(points_sample.tolist())

# make some "parcel" centroids and view them in relation to the polygon
centroid_points = MultiPoint([(0, 0), (75, 30), (10, 80), (-75, -95)])
centroids = gpd.GeoSeries(list(centroid_points))
poly.union(centroid_points)

# create an O-D dataframe for the distance matrix calculation
df_origin = pd.DataFrame()
df_origin['x'] = centroids.map(lambda coords: coords.x)
df_origin['y'] = centroids.map(lambda coords: coords.y)
df_origin = df_origin.assign(tmp_key=0)
df_origin = df_origin.reset_index()

df_dest = pd.DataFrame()
df_dest['x'] = points_sample.map(lambda coords: coords.x)
df_dest['y'] = points_sample.map(lambda coords: coords.y)
df_dest = df_dest.assign(tmp_key=0)
df_dest = df_dest.reset_index()

df_od = pd.merge(df_origin, df_dest, on='tmp_key', suffixes=('_from', '_to')).drop('tmp_key', axis=1)
df_od = df_od.set_index(['index_from', 'index_to'])

# calculate euclidean distance matrix, vectorized
# if these aren't projected (like meters), you can vectorize great-circle instead
x1 = df_od['x_from']
x2 = df_od['x_to']
y1 = df_od['y_from']
y2 = df_od['y_to']
dist_matrix = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
nearest = dist_matrix.unstack().idxmin(axis='columns')
nearest

fig, ax = plt.subplots(figsize=(6,6))
ax.add_patch(PolygonPatch(poly, fc='#dddddd', ec='none', zorder=0))

x = centroids.map(lambda coords: coords.x)
y = centroids.map(lambda coords: coords.y)
ax.scatter(x=x, y=y, facecolor='c', edgecolor='none', marker='o', s=100, zorder=2)

x = points_sample.loc[nearest].map(lambda coords: coords.x)
y = points_sample.loc[nearest].map(lambda coords: coords.y)
ax.scatter(x=x, y=y, facecolor='k', edgecolor='none', marker='o', s=20, zorder=2)

for label, value in nearest.iteritems():
    coords = df_od.loc[label, value]
    plt.plot([coords['x_from'], coords['x_to']], [coords['y_from'], coords['y_to']], c='#999999', zorder=1)

left, bottom, right, top = poly.bounds
ax.set_xlim(left * 1.2, right * 1.2)
ax.set_ylim(bottom * 1.2, top * 1.2)
plt.show()



