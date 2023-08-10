get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import pymongo
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
from shapely import geometry
from descartes import PolygonPatch
import pickle

from backend.models.time_distance_model import TimeDistanceModel
from backend.models.time_distance_model import enlarge_box
from backend.models import handle_points
from backend.models import isoline
from backend.models.geocode import Geocoder
from backend.models.distance import HaversineDistance
from backend.models.fetch_zips import fetch_zip_boundary_polygons

import warnings
warnings.filterwarnings('ignore')

client = pymongo.MongoClient('mongo', 27017)
na_db = client['na-db']

zip_code = '95666'
county_name='Amador'

fetched_obj = fetch_zip_boundary_polygons([zip_code])
zip_geometry = fetched_obj['features'][0]['geometry']
zip_polygon = geometry.shape(zip_geometry)
zip_polygon

osmisoliner = isoline.OSMIsonliner()
isoline_model = TimeDistanceModel(osmisoliner.get_single_isodistance_polygon, service_area=zip_polygon)

isoline_model.clear_graph()
isoline_model.set_inclusive_bounding_box()

get_ipython().run_line_magic('time', 'isoline_model.fetch_graph()')

point_as = handle_points.fetch_point_as(na_db, zip_code, county_name)
_ = isoline_model.build_all_isolines(point_as[:30], 15)

csv_path = '../../data/all_pcps_addresses.csv'
providers = handle_points.fetch_provider_addresses(csv_path)
close_providers = providers[providers.County.isin(['Amador', 'Sacramento', 'El Dorado', 'Calaveras', 'San Joaquin', 'Alpine'])]
print('After filtering for close providers, there are {} rows in the df.'.format(close_providers.shape[0]))

geocoder = Geocoder()
get_ipython().run_line_magic('time', 'close_providers_geocodes = close_providers.full_address.apply(geocoder.geocode)')

with open('data.pkl', 'rb') as pkl_file:
    close_providers_geocodes = pickle.load(pkl_file)

msg_base = 'Was able to geocode {} out of {} addresses. {:.2f}% success rate.'
total = len(close_providers_geocodes)
success_count = close_providers_geocodes.count()
print(msg_base.format(success_count, total, success_count * 100 / total))

close_providers['geo_location'] = close_providers_geocodes
close_providers[['covering_points', 'not_covering_points']] = close_providers.geo_location.apply(isoline_model.if_single_provider_within_all_isolines)

covered_point_as = set(close_providers.covering_points.apply(pd.Series).stack().unique())
all_point_as = set(close_providers.not_covering_points.apply(pd.Series).stack().unique())
non_covered_point_as = all_point_as - covered_point_as
print('List of covered point As: {}'.format(covered_point_as))
print('In this zip code, {} out of {} point As are covered.'.format(len(covered_point_as), len(all_point_as)))

fig, ax = ox.plot_graph(isoline_model.graph, fig_height=10, show=False, close=False, edge_color='k', edge_alpha=0.2, node_color='none')
patch = PolygonPatch(zip_polygon, fc='green', ec='none', alpha=0.5, zorder=3)
ax.add_patch(patch)
for _, isochrones in isoline_model.all_isolines:
    patch = PolygonPatch(isochrones, fc='yellow', ec='green', alpha=0.6, zorder=-1)
    ax.add_patch(patch)
for p in close_providers_geocodes:
    if p:
        patch = PolygonPatch(p.buffer(0.005), fc='red', ec='none', alpha=0.6, zorder=-1)
        ax.add_patch(patch)
for p_id in list(map(int, covered_point_as)):
    p = point_as[p_id]
    patch = PolygonPatch(p.buffer(0.005), fc='black', ec='none', alpha=1, zorder=1)
    ax.add_patch(patch)
for p_id in list(map(int, non_covered_point_as)):
    p = point_as[p_id]
    patch = PolygonPatch(p.buffer(0.005), fc='grey', ec='none', alpha=1, zorder=2)
    ax.add_patch(patch)

plt.show()

distancer = HaversineDistance()

distance_res = pd.DataFrame(columns=range(len(close_providers_geocodes)))
total_number_of_distance_measurements = 0
for far_point_a_id in list(map(int, non_covered_point_as)):
    far_point_a = point_as[far_point_a_id]
    far_point_a_provider_distance = []
    for provider in list(close_providers_geocodes):
        # Only look at providers inside the bbox.
        if isoline_model._is_point_within_polygon(provider, geometry.Polygon(isoline_model.bounding_box)):
            total_number_of_distance_measurements += 1
            d = distancer.get_distance_in_miles(far_point_a, provider)
        else:
            d = None
        far_point_a_provider_distance.append(d)
    distance_res.loc[far_point_a_id] = pd.Series(far_point_a_provider_distance)
print('I have run {} distance API calls.'.format(total_number_of_distance_measurements))
distance_res.dropna(axis=1, inplace=True)
distance_res.head()

farthest_point_a_id = distance_res.min(axis=1, skipna=True).idxmax()
farthest_point_a_closest_providers_id = distance_res.loc[farthest_point_a_id].sort_values()[:10]
print('The farthest point A from any provider is point #{}.'.format(farthest_point_a_id))
print('Here is the list of its distance from closest 10 providers:\n{}'.format(farthest_point_a_closest_providers_id))

fig, ax = ox.plot_graph(isoline_model.graph, fig_height=10, show=False, close=False, edge_color='k', edge_alpha=0.2, node_color='none')
patch = PolygonPatch(zip_polygon, fc='green', ec='none', alpha=0.5, zorder=3)
ax.add_patch(patch)
patch = PolygonPatch(point_as[int(farthest_point_a_id)].buffer(0.005), fc='blue', ec='none', alpha=0.6, zorder=-1)
ax.add_patch(patch)
for p_id in list(map(int, list(farthest_point_a_closest_providers_id.index))):
    p = list(close_providers_geocodes)[p_id]
    if p:
        patch = PolygonPatch(p.buffer(0.005), fc='red', ec='none', alpha=0.6, zorder=-1)
        ax.add_patch(patch)
plt.show()



