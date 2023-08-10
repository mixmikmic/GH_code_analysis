import pandas as pd
import geopandas as gpd
import pandana as pdna
import osmnx as ox
from shapely.geometry import Point, Polygon
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("redfin_2017-09-20-14-39-25.csv")
df = df.dropna(subset=['$/SQUARE FEET', 'ADDRESS'])
df = df[df['$/SQUARE FEET'] < 1500]
df = df[df['$/SQUARE FEET'] > 380]
df = df[df['PROPERTY TYPE'] == 'Single Family Residential']
df = df.set_index("ADDRESS")
df["ZIP"] = df.ZIP.apply(lambda x: x[:5])
print len(df)
df.head()

pd.set_option('display.max_colwidth', 100)
df.sort_values('$/SQUARE FEET', ascending=False).head().URL

df.sort_values('$/SQUARE FEET').head().URL

gdf = gpd.GeoDataFrame(df, geometry=[Point(row.LONGITUDE, row.LATITUDE) for _, row in df.iterrows()])
gdf.plot(column='$/SQUARE FEET', legend=True, figsize=(20, 16))

s = df.groupby('ZIP')['$/SQUARE FEET'].mean().sort_values(ascending=False)
s.head(10)

gdf = gpd.GeoDataFrame.from_file("zips.json")
gdf["median_price_per_sqft"] = s.loc[gdf.ZIP_CODE_5].values
gdf[~gdf.median_price_per_sqft.isnull()].plot(column="median_price_per_sqft", legend=True, figsize=(12, 10))

print df.ZIP.value_counts()
df[df.ZIP == "94609"].transpose()

G = ox.graph_from_place('Berkeley, California, USA')

ox.plot_graph(G)

nodes, edges = ox.save_load.graph_to_gdfs(G)
# since we're in lat/lng, we need the full precision (something in osmnx lowers the precision?)
nodes["x"] = [p.x for p in nodes.geometry.values]
nodes["y"] = [p.y for p in nodes.geometry.values]
nodes["point_geometry"] = nodes.geometry
nodes.head()

edges.head()

net = pdna.Network(nodes.x, nodes.y, edges.u, edges.v, edges[["length"]])
net.precompute(4000)

df["node_ids"] = net.get_node_ids(df.LONGITUDE, df.LATITUDE)
df.node_ids.head()

net.set(df.node_ids, variable=df["$/SQUARE FEET"], name="$/SQUARE FEET")

for dist in [500, 1000, 2000, 3000, 4000]:
    nodes["%dmeters" % dist] = net.aggregate(dist, type="mean", decay="flat", name="$/SQUARE FEET")
nodes.head()

nodes[nodes["2000meters"] > 0].plot(figsize=(30, 25), column="2000meters", legend=True)

nodes["tmp"] = net.aggregate(2000, type="count", decay="flat", name="$/SQUARE FEET")
nodes.plot(figsize=(30, 25), column="tmp", legend=True)

net.set(df.node_ids, variable=df.BEDS, name="beds")
nodes["tmp"] = net.aggregate(4000, type="sum", decay="linear", name="beds")
nodes.plot(figsize=(30, 25), column="tmp", legend=True)

# pytess gives you the polygon back whereas the scipy doesn't
import pytess

points = [(p.x, p.y) for p in nodes.geometry.values]
polys = pytess.voronoi(points)
# not sure what's up with this
polys = polys[:len(nodes)]
# drop the point, turn into shapely poly
nodes["voronoi_geometry"] = [Polygon(p[1]) for p in polys]

nodes["geometry"] = nodes["voronoi_geometry"]
ax = nodes.plot(figsize=(30, 25), column="tmp", legend=True)
ax.set_ylim([37.85, 37.89])
ax.set_xlim([-122.29, -122.24])
nodes["geometry"] = nodes["point_geometry"]
nodes.plot(ax=ax, color='black', alpha=0.5)

tmp_df = pd.DataFrame({"y": [37.87], "x": [-122.27], "value": [1.0]})
tmp_df["node_ids"] = net.get_node_ids(tmp_df.x, tmp_df.y)
net.set(tmp_df.node_ids, variable=tmp_df.value, name="tmp")
nodes["tmp"] = net.aggregate(1800, type="sum", decay="flat", name="tmp")
nodes.plot(figsize=(30, 25), column="tmp", legend=False)

net.set_pois("redfin", 4000, 10, df.LONGITUDE, df.LATITUDE)
k = 10
results = net.nearest_pois(4000, "redfin", num_pois=k, include_poi_ids=True)
results.head()

knn_values = pd.DataFrame()
for i in range(1, k+1):
    col = "poi%d" % i
    knn_values[col] = df.loc[results[col]]["$/SQUARE FEET"].values
s = knn_values.mean(axis=1)
filtered_nodes = nodes.loc[results.index]
filtered_nodes["knn_ave"] = s.values
filtered_nodes.plot(column="knn_ave", figsize=(30, 25), legend=True)

nodes[nodes["2000meters"] > 0].plot(figsize=(30, 25), column="2000meters", legend=True)



