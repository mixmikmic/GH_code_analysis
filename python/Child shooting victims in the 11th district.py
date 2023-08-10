import os
import requests

def get_table_url(table_name, base_url=os.environ['NEWSROOMDB_URL']):
    return '{}table/json/{}'.format(os.environ['NEWSROOMDB_URL'], table_name)

def get_table_data(table_name):
    url = get_table_url(table_name)
    
    try:
        r = requests.get(url)
        return r.json()
    except:
        print("Request failed. Probably because the response is huge.  We should fix this.")
        return get_table_data(table_name)

shooting_victims = get_table_data('shootings')

print("Loaded {} shooting victims".format(len(shooting_victims)))

import requests
from shapely.geometry import shape

# The City of Chicago's Socrata-based Data Portal provides a GeoJSON export of its spatial datasets.
# We'll use this so we don't have to save spatial data to the repo.
POLICE_DISTRICT_BOUNDARIES_GEOJSON_URL = "https://data.cityofchicago.org/api/geospatial/fthy-xz3r?method=export&format=GeoJSON"

r = requests.get(POLICE_DISTRICT_BOUNDARIES_GEOJSON_URL)
police_districts = r.json()

# Get the District 11 GeoJSON feature
district_11_feature = next(f for f in police_districts['features']
                           if f['properties']['dist_num'] == "11")

# Convert it to a Shapely shape so we can detect our 
district_11_boundary = shape(district_11_feature['geometry'])

from datetime import datetime
import re

import pandas as pd

def parse_date(s):
    try:
        return datetime.strptime(s, '%Y-%m-%d').date()
    except ValueError:
        return None
    
def parse_coordinates(coordinate_str):
    """Convert a lat, lng string to a pair of lng, lat floats"""
    try:
        lat, lng = [float(c) for c in re.sub(r'[\(\) ]', '', coordinate_str).split(',')]
        return lng, lat
    except ValueError:
        return None
    
def parse_age(age_str):
    try:
        return int(age_str)
    except ValueError:
        return None
    
def get_year(shooting_date):
    try:
        return shooting_date.year
    except AttributeError:
        return None

shooting_victims_df = pd.DataFrame(shooting_victims)
shooting_victims_df['Date'] = shooting_victims_df['Date'].apply(parse_date)
shooting_victims_df['Age'] = shooting_victims_df['Age'].apply(parse_age)
shooting_victims_df['coordinates'] = shooting_victims_df['Geocode Override'].apply(parse_coordinates)
shooting_victims_df['year'] = shooting_victims_df['Date'].apply(get_year)

child_shooting_victims = shooting_victims_df[shooting_victims_df['Age'] < 18]
child_shooting_victims_16_and_under = child_shooting_victims[child_shooting_victims['Age'] <= 16]

child_shooting_victims_since_2012 = child_shooting_victims[child_shooting_victims['year'] >= 2012]
child_shooting_victims_16_and_under_since_2012 = child_shooting_victims_16_and_under[child_shooting_victims_16_and_under['year'] >= 2012]

from shapely.geometry import Point

def is_in_11th_district(shooting_coordinates):
    try:
        shooting_point = Point(shooting_coordinates[0], shooting_coordinates[1])
        return district_11_boundary.contains(shooting_point)
    except TypeError:
        return False
    
child_shooting_victims_since_2012_in_11th_dist = child_shooting_victims_since_2012[
    child_shooting_victims_since_2012['coordinates'].apply(is_in_11th_district)
]


child_shooting_victims_16_and_under_since_2012_in_11th_dist = child_shooting_victims_16_and_under_since_2012[
    child_shooting_victims_16_and_under_since_2012['coordinates'].apply(is_in_11th_district)
]

print("There have been {} victims, under 18 years of age, who have been shot in the 11th district since 2012".format(
    len(child_shooting_victims_since_2012_in_11th_dist)))

print("There have been {} victims, age 16 or under, who have been shot in the 11th district since 2012".format(
    len(child_shooting_victims_16_and_under_since_2012_in_11th_dist)))

# Sanity check our filter
# It looks like one of our rows has a district of 10.  Maybe this is because of bad
# data entry
for i, victim in child_shooting_victims_16_and_under_since_2012_in_11th_dist.iterrows():
    print(victim['District'])

child_shooting_victims_since_2012_in_11th_dist_by_year = pd.DataFrame(
    child_shooting_victims_since_2012_in_11th_dist.groupby('year').size(),
    columns=['num_victims']
)
child_shooting_victims_since_2012_in_11th_dist_by_year

child_shooting_victims_16_and_under_since_2012_in_11th_dist_by_year = pd.DataFrame(
    child_shooting_victims_16_and_under_since_2012_in_11th_dist.groupby('year').size(),
    columns=['num_victims']
)
child_shooting_victims_16_and_under_since_2012_in_11th_dist_by_year

def is_11th_district(district):
    try:
        return int(district) == 11
    except ValueError:
        return False

child_shooting_victims_since_2012_in_11th_dist = child_shooting_victims_since_2012[child_shooting_victims_since_2012['District'].apply(is_11th_district)]
print("There have been {} victims, under age 18, who have been shot in the 11th district since 2012".format(
    len(child_shooting_victims_since_2012_in_11th_dist)))
child_shooting_victims_16_and_under_since_2012_in_11th_dist = child_shooting_victims_16_and_under_since_2012[child_shooting_victims_16_and_under_since_2012['District'].apply(is_11th_district)]
print("There have been {} victims, under age 18, who have been shot in the 11th district since 2012".format(
    len(child_shooting_victims_since_2012_in_11th_dist)))

child_shooting_victims_since_2012_in_11th_dist_by_year = pd.DataFrame(
    child_shooting_victims_since_2012_in_11th_dist.groupby('year').size(),
    columns=['num_victims']
)
child_shooting_victims_since_2012_in_11th_dist_by_year

child_shooting_victims_16_and_under_since_2012_in_11th_dist_by_year = pd.DataFrame(
    child_shooting_victims_16_and_under_since_2012_in_11th_dist.groupby('year').size(),
    columns=['num_victims']
)
child_shooting_victims_16_and_under_since_2012_in_11th_dist_by_year



