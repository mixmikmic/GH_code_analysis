import requests

def get_chicago_community_areas():
    url = 'https://data.cityofchicago.org/api/geospatial/cauq-8yn6?method=export&format=GeoJSON'
    resp = requests.get(url, verify=False)
    return resp.json()

community_areas = get_chicago_community_areas()

from shapely.geometry import shape
# Get the shapes as a map between community area number and shape as we'll need the IDs anyway to build our index later
community_area_shapes = {int(f['properties']['area_num_1']): shape(f['geometry']) for f in community_areas['features']}
community_area_properties = {int(f['properties']['area_num_1']): f['properties'] for f in community_areas['features']}

from rtree import index

communty_area_index = index.Index()
for ca_number, ca_shape in community_area_shapes.items():
    communty_area_index.add(ca_number, ca_shape.bounds, obj=community_area_properties[ca_number])

from shapely.geometry import Point

def point_to_bounds(point):
    """
    Convert a point to a bounding box
    
    It makes sense to represent points as an x,y pair, but RTree only operates
    on bounding boxes. Convert the point to a bounding box where left == right
    and top == bottom.

    """
    return (point[0], point[1], point[0], point[1])

def get_community_area(point, ca_idx, ca_shapes):
    areas = []
    for n in ca_idx.intersection(point_to_bounds(point), objects=True):
        ca_number = int(n.object['area_num_1'])
        ca_shape = ca_shapes[ca_number]
        if ca_shape.contains(Point(*point)):
            areas.append(n.object)
    return areas
        
# Turkey Chop is a restaurant that is most definitely in Humboldt Park
# Let's use it to spot-check our index
turkey_chop_coords = [-87.7141142377237, 41.8955710581678]

turkey_chop_ca = get_community_area(turkey_chop_coords, communty_area_index, community_area_shapes)
assert turkey_chop_ca[0]['community'] == "HUMBOLDT PARK"

import os
import requests

# Some constants
NEWSROOMDB_URL = os.environ['NEWSROOMDB_URL']

# A big object to hold all our data between steps
data = {}

def get_table_url(table_name, base_url=NEWSROOMDB_URL):
    return '{}table/json/{}'.format(base_url, table_name)

def get_table_data(table_name):
    url = get_table_url(table_name)
    
    try:
        r = requests.get(url)
        return r.json()
    except:
        print("Request failed. Probably because the response is huge.  We should fix this.")
        return get_table_data(table_name)

data['shooting_victims'] = get_table_data('shootings')
print("Loaded {} shooting victims".format(len(data['shooting_victims'])))

data['homicides'] = get_table_data('homicides')
print("Loaded {} homicides".format(len(data['homicides'])))

import pandas as pd
import numpy as np

data['shooting_victims_df'] = pd.DataFrame(data['shooting_victims'])
data['homicides_df'] = pd.DataFrame(data['homicides'])

from datetime import datetime

def parse_date(s):
    try:
        return datetime.strptime(s, '%Y-%m-%d').date()
    except ValueError:
        return None
    
data['shooting_victims_df']['Date'] = data['shooting_victims_df']['Date'].apply(parse_date)
data['shooting_victims_df']['month'] = data['shooting_victims_df']['Date'].apply(lambda x: x.month if x else None)
data['shooting_victims_df']['year'] = data['shooting_victims_df']['Date'].apply(lambda x: x.year if x else None)

import pprint
import re

def parse_coordinates(coordinate_str):
    """Convert a lat, lng string to a pair of lng, lat floats"""
    lat, lng = [float(c) for c in re.sub(r'[\(\) ]', '', coordinate_str).split(',')]
    return lng, lat

shooting_victim_community_areas = {}

for victim in data['shooting_victims']:
    try:
        coords = parse_coordinates(victim['Geocode Override'])
    except ValueError:
        shooting_victim_community_areas[victim['_id']] = '__invalid__'
        continue
        
    ca = get_community_area(coords, communty_area_index, community_area_shapes)
    
    if len(ca) == 0:
        shooting_victim_community_areas[victim['_id']] = '__invalid__'
        print("No community area found for record with coordinates {}".format(coords))
    elif len(ca) > 1:
        raise ValueError("Multiple community areas found for record with coordinates {}".format(coords))
    else:
        shooting_victim_community_areas[victim['_id']] = ca[0]['community']
        
data['shooting_victim_community_areas'] = pd.DataFrame([{'_id': k, 'community': v} for k, v in shooting_victim_community_areas.items()])

data['shooting_victims_df__with_ca'] = data['shooting_victims_df'].merge(
    data['shooting_victim_community_areas'],
    how='left',
    on='_id')

data['shooting_victims_by_ca'] = pd.DataFrame(data['shooting_victims_df__with_ca'].groupby(['community', 'year', 'month']).size())

df = data['shooting_victims_by_ca']
df[(df.index.get_level_values('year') == 2016) & (df.index.get_level_values('month') == 3)].sort_values(by=0, ascending=False)

df = data['shooting_victims_by_ca']
df[(df.index.get_level_values('community') == "HUMBOLDT PARK") & (df.index.get_level_values('month') == 3)]



