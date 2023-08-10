import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
get_ipython().magic('matplotlib inline')
import sys
sys.path.append('../')

from intersections_and_roads import *
from shapely.geometry import *
import ast

intersections_points = gpd.read_file('../sf_data/street_intersections.geojson')
street_centerlines = gpd.read_file('../sf_data/San_Francisco_Basemap_Street_Centerlines.geojson')
street_centerlines_2 = gpd.read_file('../sf_data/street_intersections.geojson')

street_centerlines_2.columns

# bike_accidents = pd.read_json('../sf_data/sanfrancisco_crashes.json')
import json
with open('../sf_data/sanfrancisco_crashes.json') as sf_data:    
    bike_accidents = json.load(sf_data)

# # bike_accidents = json.loads('../sf_data/sanfrancisco_crashes.json')

from pandas.io.json import json_normalize
bike_accidents_df = json_normalize(bike_accidents['accidents'])
bike_accidents_df.head(5)
# type(json_normalize(bike_accidents['accidents']))

bike_accidents_df['Street_ID'] = 0
street_centerlines.columns

for i in range(len(bike_accidents_df)):
    new_point = Point(bike_accidents_df['lng'][i], bike_accidents_df['lat'][i])
    test_distances = [new_point.distance(x) for x in street_centerlines['geometry']]
    index = np.argmin(test_distances)
    bike_accidents_df.iloc[i]['Street_ID'] = street_centerlines.iloc[index]['id']
    

bike_crashes_per_street = bike_accidents_df.groupby(['Street_ID'], as_index=False).size().reset_index()

bike_crashes_per_street.rename(index=str, inplace=True, columns={0: 'num_accidents'})

bike_crashes_per_street.head()

bike_crashes_per_street.to_pickle("sf_bike_crashes.pkl")

bike_crashes_per_street = pd.read_pickle("../sf_data/sf_bike_crashes.pkl")

bike_crashes_per_street.head(5)

street_centerlines[street_centerlines['id']==512000]

sf_elevation = pd.read_csv('../sf_data/node_elevation_2.csv', delimiter=' ', header=None, names=['id', 'elevation'])

sf_elevation['id'] = sf_elevation['id'].astype(int)
sf_elevation.head(5)

def f(x):    
    return Point(x['lon'],x['lat']) 
sf_elevation['point']=sf_elevation[['lon', 'lat']].apply(f, axis=1)
# print sf_elevation[['lon','lat']]['lon']
# for i in range(len(sf_elevation)):
#     sf_elevation.iloc[i]['point'] = Point(sf_elevation.iloc[i]['lon'],sf_elevation.iloc[i]['lat'])

sf_elevation.head(5)

# def elev(node):
# #     print node
#     point = Point(node.get_x_y()[0], node.get_x_y()[1])
#     dist = [point.distance(elev_point) for elev_point in sf_elevation['point']]
#     closest = np.argmin(dist)
#     return sf_elevation.iloc[closest]['alt']

def elev(node):
#     print node
#     point = Point(node.get_x_y()[0], node.get_x_y()[1])
#     dist = [point.distance(elev_point) for elev_point in sf_elevation['point']]
#     closest = np.argmin(dist)
#     print int(node.id)
    return sf_elevation[sf_elevation['id']==int(node.id)]['elevation']

street_centerlines['ToNode'] = street_centerlines['t_node_cnn'].astype(np.int32)
street_centerlines['FromNode'] = street_centerlines['f_node_cnn'].astype(np.int32)
street_centerlines['id'] =  street_centerlines['cnn'].apply(lambda x: int(float(x)))
# Get the direction
street_centerlines['Direction'] = street_centerlines['oneway'].apply(lambda x: 0 if x == 'B' else 1 if x == 'F' else -1)
street_centerlines.head(2)

intersections_points['NodeNumber'] = intersections_points['cnntext'].apply(lambda x: int(float(x)))
intersections_points['id'] = intersections_points['cnntext']
intersections_points.head(2)

fig, ax = plt.subplots(1,1, figsize=(40,40))
intersections_points.plot(ax=ax)
street_centerlines.plot(ax=ax)

intersection_graph, connection_dict = build_intersection_graph(intersections_points, street_centerlines, sf_elevation, bike_crashes_per_street)

intersections_points.head()



bike_crashes_per_street = pd.read_pickle("../sf_data/sf_bike_crashes.pkl")
for street in bike_crashes_per_street.itertuples():
    if connection_dict.get(street[1], None):
        connection_dict[street[1]].add_accidents(street[2])

# node_elevation = []
for node in intersection_graph.values():
    node.set_elevation(elev(node))
#     node_elevation.append([node.id, node.elevation])
#     if len(node_elevation)%1000==0:
#         print len(node_elevation)

print len(node_elevation)
np_elevation = np.array(node_elevation)
np_elevation_float = np_elevation.astype('float64')

print np_elevation_float.dtype

np.savetxt("node_elevation_2.csv", np_elevation_float)

fig, ax = plt.subplots(1,1, figsize=(15, 15))

xs = [intersection_graph[key].get_x_y()[0] for key in intersection_graph]
ys = [intersection_graph[key].get_x_y()[1] for key in intersection_graph]

for key in intersection_graph:
    node = intersection_graph[key]
    for connection in node.get_connections():
        child = connection_dict[connection]
        line_x = [child.get_source(intersection_graph).get_x_y()[0], child.get_target(intersection_graph).get_x_y()[0]]
        line_y = [child.get_source(intersection_graph).get_x_y()[1], child.get_target(intersection_graph).get_x_y()[1]]
        ax.plot(line_x, line_y)

ax.scatter(xs, ys, s=10)
plt.show()

# lat = np.apply_along_axis(lambda p: p.x, 1, bike_accident_locations[['geometry']].values,)
accident_lat = bike_accidents_df['lat']
accident_lng = bike_accidents_df['lng']

sf_elevation.head()

intersections_points.head()

intersections_points[intersections_points['id'] == '27380000']

intersections_points['id'] = intersections_points['id'].astype(int)

node_elev_locations = sf_elevation.merge(intersections_points, on='id', how='right')

node_elev_locations.head()

elev_lat = node_elev_locations['geometry'].apply(lambda p: p.y)
elev_lng = node_elev_locations['geometry'].apply(lambda p: p.x)
elev_vals = node_elev_locations['elevation'].values

elev_vals

# randomly select a start and an end point on the graph for test
begin = intersection_graph[np.random.choice(intersection_graph.keys())]
goal = intersection_graph[np.random.choice(intersection_graph.keys())]

# randomly select a start and an end point on the graph for test
begin = intersection_graph[p2.id]
goal = intersection_graph[p1.id]

a1 = intersection_graph[np.random.choice(intersection_graph.keys())]
a2 = intersection_graph[np.random.choice(intersection_graph.keys())]
a3 = intersection_graph[np.random.choice(intersection_graph.keys())]
a4 = intersection_graph[np.random.choice(intersection_graph.keys())]
a5 = intersection_graph[np.random.choice(intersection_graph.keys())]
a6 = intersection_graph[np.random.choice(intersection_graph.keys())]

plt.figure(figsize=(15,15))
plt.scatter(a1.get_x_y()[0], a1.get_x_y()[1], c='red')
plt.scatter(a2.get_x_y()[0], a2.get_x_y()[1], c='green')
plt.scatter(a3.get_x_y()[0], a3.get_x_y()[1], c='blue')
plt.scatter(a4.get_x_y()[0], a4.get_x_y()[1], c='yellow')
plt.scatter(a5.get_x_y()[0], a5.get_x_y()[1], c='orange')
plt.scatter(a6.get_x_y()[0], a6.get_x_y()[1], c='magenta')

start_id = a6

begin = intersection_graph[start_id.id]
goal = intersection_graph[a6.id]

end_id = a6

with open('san_fran_start_end_points', 'wb') as f:
    pickle.dump([start_id, end_id], f)

def get_safe_road_cost_with_elevation(road_list, connection_list, intersection_graph, connection_dict):
    distance = 0
    for connection_id in connection_list:
        multiplier = 10000
        weight = 5*connection_dict[connection_id].get_accidents() + 1
        distance += (max(multiplier*connection_dict[connection_id].get_distance(), 1))*weight
        distance += np.abs(connection_dict[connection_id].delta_elevation)
    return distance

import time
start = time.time()
route = a_star_search(begin, goal, intersection_graph, connection_dict, get_road_cost)
end = time.time()
print 'Cost: Distance, Time: ', (end-start)
start = time.time()
safe_route = a_star_search(begin, goal, intersection_graph, connection_dict, get_safe_road_cost, null_heuristic)
end = time.time()
print 'Cost: Accidents + distance, Time: ', (end-start)
start = time.time()
safe_route_with_elevation = a_star_search(begin, goal, intersection_graph, connection_dict, get_safe_road_cost_with_elevation)
end = time.time()
print 'Cost: Accidents + distance + elevation, Time: ', (end-start)

distance_cost = get_road_cost(route['nodes'], route['connections'], intersection_graph, connection_dict)
distance_accidents_cost = get_road_cost(safe_route['nodes'], safe_route['connections'], intersection_graph, connection_dict)
distance_accidents_elevation_cost = get_road_cost(safe_route_with_elevation['nodes'], safe_route_with_elevation['connections'], intersection_graph, connection_dict)

print 'distance', distance_cost
print 'distance', distance_accidents_cost
print 'distance', distance_accidents_elevation_cost

def plot_graph(intersection_graph, connection_dict, routes = [], safe_routes=[], ax = None):
    if ax == None:
        fig, ax = plt.subplots(1,1, figsize=(15, 15))

    xs = [intersection_graph[key].get_x_y()[0] for key in intersection_graph]
    ys = [intersection_graph[key].get_x_y()[1] for key in intersection_graph]

    for key in intersection_graph:
        node = intersection_graph[key]
        for connection in node.get_connections():
            child = connection_dict[connection]
            line_x = [child.get_source(intersection_graph).get_x_y()[0], child.get_target(intersection_graph).get_x_y()[0]]
            line_y = [child.get_source(intersection_graph).get_x_y()[1], child.get_target(intersection_graph).get_x_y()[1]]
            ax.plot(line_x, line_y, color='#d3d3d3', zorder=1)

    ax.scatter(xs, ys, s=10, color='#7e7e7e', zorder=1)

    for route in routes:
        xs = [intersection_graph[node].get_x_y()[0] for node in route]
        ys = [intersection_graph[node].get_x_y()[1] for node in route]
        ax.plot(xs, ys, linewidth=7, linestyle='dashed', zorder=3)

    for route in safe_routes:
        xs = [intersection_graph[node].get_x_y()[0] for node in route]
        ys = [intersection_graph[node].get_x_y()[1] for node in route]
        ax.plot(xs, ys, linewidth=3, zorder=3)

    # plt.show()
    return ax

axis = plot_graph(intersection_graph, connection_dict, [route['nodes']], [safe_route['nodes'], safe_route_with_elevation['nodes']])
# axis.scatter(accident_lng, accident_lat, color='yellow')
colors = plt.cm.coolwarm(np.divide(elev_vals-np.min(elev_vals), np.max(elev_vals-np.min(elev_vals))))
axis.scatter(elev_lng, elev_lat, c=colors, zorder=2)

axis = plot_graph(intersection_graph, connection_dict, [route['nodes']], [safe_route['nodes'], safe_route_with_elevation['nodes']])
axis.scatter(accident_lng, accident_lat, color='yellow')



[p1,p2,p3,p4,best_centroid, best_centroid, k_points] = vals
begin = intersection_graph[np.random.choice(intersection_graph.keys())]
goal = intersection_graph[np.random.choice(intersection_graph.keys())]

begin = start_id
goal = end_id

def combined_heuristic(node, goal, intersection_graph, connection_dict):
    accident_heuristic = 1
    if node.get_connections():
        accident_heuristic += (np.min([connection_dict[c].get_accidents() for c in node.get_connections()]))
    distance = euclidean_distance(node.get_x_y(), goal.get_x_y())
    elevation = goal.get_elevation() - node.get_elevation()
    return (distance*10 + elevation)*accident_heuristic

start = time.time()
route = a_star_search(begin, goal, intersection_graph, connection_dict, get_safe_road_cost_with_elevation, null_heuristic)
end = time.time()
print 'null heuristic, Time: ', (end-start)
start = time.time()
route = a_star_search(begin, goal, intersection_graph, connection_dict, get_safe_road_cost_with_elevation, euclidean_heuristic)
end = time.time()
print 'euclidean heuristic, Time: ', (end-start)
start = time.time()
route = a_star_search(begin, goal, intersection_graph, connection_dict, get_safe_road_cost_with_elevation, combined_heuristic)
end = time.time()
print 'combined heuristic, Time: ', (end-start)

axis = plot_graph(intersection_graph, connection_dict, [route['nodes']], [])
axis.scatter(accident_lng, accident_lat, color='yellow')



