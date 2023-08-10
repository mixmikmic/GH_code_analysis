get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from pymongo import MongoClient
import datetime
import numpy as np
import pandas as pd
import getpass
from utility_functions import *

country_geojson_filename = '../website/static/geojson/countries.json'
prettify_JSON(country_geojson_filename)
get_ipython().system("cat 'test.json'")

def geojson_name_to_geojson_id_filetype1(geojson_content):
    # read geojson file and map names of countries to corresponding ids
    # returns dict
    
    location_name_to_gson_id_mapping = dict()
    
    all_locations = geojson_content['features']
    for entry in all_locations:
        country_id = entry['id']
        country_name = entry['properties']['name']
        #print("Id: {}, name: {}".format(country_id, country_name))
        location_name_to_gson_id_mapping[country_name] = country_id

    return location_name_to_gson_id_mapping

# for a different type of geoJSON file
def geojson_name_to_geojson_id_filetype2(geojson_content):
    # read geojson file and map names of countries to corresponding ids
    # returns dict
    
    location_name_to_gson_id_mapping = dict()
    
    all_locations = geojson_content['features']
    for entry in all_locations:
        country_id = entry['properties']['iso_a3']
        country_name = entry['properties']['sovereignt']
        #print("Id: {}, name: {}".format(country_id, country_name))
        location_name_to_gson_id_mapping[country_name] = country_id

    return location_name_to_gson_id_mapping

# for a different type of geoJSON file
def geojson_name_to_geojson_id_filetype3(geojson_content):
    # read geojson file and map names of countries to corresponding ids
    # returns dict
    
    location_name_to_gson_id_mapping = dict()
    
    all_locations = geojson_content['features']
    for entry in all_locations:
        country_id = entry['id']
        country_name = entry['properties']['city']
        if country_id != country_name:
            print("ID Not the same: {}".format(country_id))
        #print("Id: {}, name: {}".format(country_id, country_name))
        location_name_to_gson_id_mapping[country_name] = country_id

    return location_name_to_gson_id_mapping



geojson_filenames_with_parser_fcn = {
    'continent': [None, None],
    'subcontinent': [None, None],
    'country': ['../website/static/geojson/custom.geo.json', geojson_name_to_geojson_id_filetype2],
    'region': ['../website/static/geojson/us-states.json', geojson_name_to_geojson_id_filetype1],
    'city': ['../website/static/geojson/cities.geo.json', geojson_name_to_geojson_id_filetype3],
}

geojson_name_to_id_mapping = dict()
for (location_type, geojson_filename_with_parser) in geojson_filenames_with_parser_fcn.items():
    geojson_filename = geojson_filename_with_parser[0]
    parse_fcn = geojson_filename_with_parser[1]
    if geojson_filename is not None:
        geojson_content = load_JSON(geojson_filename)
        location_name_to_gson_id_mapping = parse_fcn(geojson_content)
        geojson_name_to_id_mapping[location_type] = location_name_to_gson_id_mapping
    else:
        geojson_name_to_id_mapping[location_type] = dict() # empty
    

geojson_name_to_id_mapping

# get all locations used in the database
all_locations_database_filename = '../data-scraping/importedTags_true.json'
all_locations_database_content = load_JSON(all_locations_database_filename)

def get_location_type_name(location_type_code):
    # convention used in the file to denote type of location
    
    if location_type_code == 1:
        return 'continent'
    elif location_type_code == 2:
        return 'country'
    elif location_type_code == 3:
        return 'region'
    elif location_type_code == 4:
        return 'city'
    elif location_type_code == 5:
        return 'subcontinent'
    elif location_type_code == 0:
        #print('Ignoring')
        return -1
    else:
        print("Error: Location code {} not implemented yet".format(location_type_code))
        return -1

from geopy.geocoders import GoogleV3


# hide key to avoid github scrapers
def encrypt(character, shift):
    startChar = '-'
    endChar = 'z'
    lengthDif = ord(endChar) - ord(startChar) + 1
    return chr(ord(startChar) + (ord(character) - ord(startChar) + shift) % lengthDif)

def decrypt(character, shift):
    return encrypt(character, -shift)

def encryptText(text, shift):
    return "".join(list(map(lambda c: encrypt(c, shift),text)))

def decryptText(text, shift):
    return encryptText(text, -shift)

encryptedGoogleApiKey = 'KS6k]5MiT:6M<7zCi@zKosTXO`OqiUsLmYv2tWu'
googleApiKey = decryptText(encryptedGoogleApiKey, shift = 10)

geolocator = GoogleV3(api_key=googleApiKey, domain='maps.googleapis.com', scheme='https')
geolocator.geocode(query='european', exactly_one=True)

saved_geolocations_filename = 'saved_geolocations.pickled'
import pickle
from collections import defaultdict

use_saved_file = True # False, # Only set to false if database names have changed

loaded_geolocations = False
saved_geolocations = defaultdict(dict)
if use_saved_file:
    try:
        saved_geolocations = pickle.load(open(saved_geolocations_filename, "rb"))
        loaded_geolocations = True
    except (OSError, IOError) as e:
        saved_geolocations = defaultdict(dict)
        loaded_geolocations = False

# Only run if database names have changed

database_names_to_nicer_names = defaultdict(dict) # an entry for each location type, e.g. continent, subcontinent, countries, regions/states

#display_names to show on maps afterwards
i = 0
nbIterations = len(all_locations_database_content)
for (location_name, location_type_code) in all_locations_database_content.items():
    i = i+1
    if i%20 == 1:
        print("Starting iteration {}/{}".format(i, nbIterations))
    
    location_type_name = get_location_type_name(location_type_code)
    if location_type_name == -1:
        # could not be parsed or invalid
        continue
    
    try:
        if loaded_geolocations and (location_name in saved_geolocations[location_type_name]):
            api_location = saved_geolocations[location_type_name][location_name]
            #print("Already computed")
        else:
            query_location_name = location_name
            if location_name.endswith('an') and location_type_name != 'continent': # e.g. korean, russian
                query_location_name = query_location_name[:-1]
            api_location = geolocator.geocode(query=query_location_name, exactly_one=True) # probably matches with GeoJSON
            saved_geolocations[location_type_name][location_name] = api_location
        
        if location_type_name == 'region':
            # because the geojson uses the shortname as "name"
            nicer_location_name = api_location.raw['address_components'][0]['short_name']
            display_name = api_location.raw['address_components'][0]['long_name']
            for compon in api_location.raw['address_components']:
                if 'administrative_area_level_1' in compon['types']:
                    nicer_location_name = compon['short_name']
                    break
        else:
            nicer_location_name = api_location.raw['address_components'][0]['long_name']
            display_name = nicer_location_name
    except Exception as myExc:
        print('Could not translate {} ({}) into nicer name, error: {}'.format(location_name, location_type_name, myExc))
        nicer_location_name = 'error'
        display_name = 'error'
    
    #print("{} ({}): {}".format(location_name, location_type_name, nicer_location_name))
    database_names_to_nicer_names[location_type_name][location_name] = {
        'nice_name': nicer_location_name, 'display_name': display_name # display_name used to show on map, i.e. not an id/or ugly name for the geojson
    }
    
    
    #if i >= 2:
    #    break

# save for later use
pickle.dump(saved_geolocations, open(saved_geolocations_filename, "wb"))
loaded_geolocations = True

#saved_geolocations['state']#['california']
#saved_geolocations['continent']['chile'].raw#.raw['address_components'][0]['long_name']

location = geolocator.geocode(query='indiana', exactly_one=True)
location.raw


for (location_type_key, locations) in database_names_to_nicer_names.items():
    for (location_key, val) in locations.items():
        if str.isnumeric(val['nice_name']):
            print("{}, {}, {}".format(location_type_key, location_key, val))
        

# Manually assign what did not work
{'nice_name': 'United States of America', 'display_name': 'United States of America'}
database_names_to_nicer_names['country']['us-recipes'] = {'nice_name': 'United States of America', 'display_name': 'United States of America'}
database_names_to_nicer_names['country']['danish'] = {'nice_name': 'Denmark', 'display_name': 'Denmark'}
database_names_to_nicer_names['country']['finnish'] = {'nice_name': 'Finland', 'display_name': 'Finland'}
database_names_to_nicer_names['country']['greek'] = {'nice_name': 'Greece', 'display_name': 'Greece'}
database_names_to_nicer_names['country']['polish'] = {'nice_name': 'Poland', 'display_name': 'Poland'}
database_names_to_nicer_names['country']['swedish'] = {'nice_name': 'Sweden', 'display_name': 'Sweden'}
database_names_to_nicer_names['country']['lebanese'] = {'nice_name': 'Lebanon', 'display_name': 'Lebanon'}
database_names_to_nicer_names['country']['turkish'] = {'nice_name': 'Turkey', 'display_name': 'Turkey'}
database_names_to_nicer_names['country']['dutch'] = {'nice_name': 'Netherlands', 'display_name': 'Netherlands'}
database_names_to_nicer_names['country']['vietnamese'] = {'nice_name': 'Vietnam', 'display_name': 'Vietnam'}
database_names_to_nicer_names['country']['filipino'] = {'nice_name': 'Philippines', 'display_name': 'Philippines'}
database_names_to_nicer_names['country']['french'] = {'nice_name': 'France', 'display_name': 'France'}
database_names_to_nicer_names['country']['irish'] = {'nice_name': 'Ireland', 'display_name': 'Ireland'}
database_names_to_nicer_names['country']['australian-and-new-zealander'] = {'nice_name': 'Australia', 'display_name': 'Australia'} 

#database_names_to_nicer_names['region']['new-hampshire'] = {'nice_name': 'NH', 'display_name': 'New Hampshire'}

database_names_to_nicer_names['city']['new-york-city'] = {'nice_name': 'New York City', 'display_name': 'New York City'} 
database_names_to_nicer_names['city']['new-york'] = {'nice_name': 'New York City', 'display_name': 'New York City'} 
database_names_to_nicer_names['city']['kansas-city'] = {'nice_name': 'Kansas City', 'display_name': 'Kansas City'} 

# Check again
for (location_type_key, locations) in database_names_to_nicer_names.items():
    for (location_key, val) in locations.items():
        if str.isnumeric(val['nice_name']):
            print("{}, {}, {}".format(location_type_key, location_key, val))
        

database_names_to_nicer_names

geojson_name_to_id_mapping


database_names_to_json_id = dict()

for location_type in database_names_to_nicer_names:
    if len(geojson_name_to_id_mapping[location_type]) == 0:
        print('No geojson data for location type "{}"'.format(location_type))
        continue
        
    database_names_to_json_id[location_type] = dict()
    for (location_name, location_nicer) in database_names_to_nicer_names[location_type].items():
        location_nice_name = location_nicer['nice_name']
        try:
            gson_id = geojson_name_to_id_mapping[location_type][location_nice_name]
            location_gson_id_and_display_name = {'geo_identifier': gson_id, 'display_name': location_nicer['display_name']}
        except KeyError as myExc:
            print(myExc)
            print("Could not map {}/{} ({}): {}".format(location_nice_name, location_name, location_type, myExc))
            location_gson_id_and_display_name = {'geo_identifier': 'invalid', 'display_name': location_name}
            
        database_names_to_json_id[location_type][location_name] = location_gson_id_and_display_name

#geojson_name_to_id_mapping['region']['CA']#['california']
#database_names_to_nicer_names['region']['california']
database_names_to_json_id['region']['california']
#saved_geolocations['country']['israeli'].raw
#geojson_name_to_id_mapping.keys()

def create_double_dict(geoId, displName):
    return dict({'geo_identifier': geoId, 'display_name': displName})

# remap some that did not work, some are not in the geojson, others are not found because of GoogleV3
database_names_to_json_id['country']['scottish'] = create_double_dict(geojson_name_to_id_mapping['country']['United Kingdom'], 'Scotland')
database_names_to_json_id['country']['puerto-rican'] = create_double_dict(geojson_name_to_id_mapping['country']['Dominican Republic'], 'Puerto Rico')
database_names_to_json_id['country']['german'] = create_double_dict(geojson_name_to_id_mapping['country']['Germany'], 'Germany')
database_names_to_json_id['country']['portuguese'] = create_double_dict(geojson_name_to_id_mapping['country']['Portugal'], 'Portugal')
database_names_to_json_id['country']['korean'] = create_double_dict(geojson_name_to_id_mapping['country']['South Korea'], 'South Korea')
database_names_to_json_id['country']['spanish'] = create_double_dict(geojson_name_to_id_mapping['country']['Spain'], 'Spain')
database_names_to_json_id['country']['english'] = create_double_dict(geojson_name_to_id_mapping['country']['United Kingdom'], 'United Kingdom')
database_names_to_json_id['country']['welsh'] = create_double_dict(geojson_name_to_id_mapping['country']['United Kingdom'], 'Wales')
database_names_to_json_id['country']['czech'] = create_double_dict(geojson_name_to_id_mapping['country']['Czech Republic'], 'Czech Republic')

database_names_to_json_id['city']['california'] = create_double_dict(geojson_name_to_id_mapping['city']['Riverside'], 'California (Riverside)')
database_names_to_json_id['city']['green-bay'] = create_double_dict(geojson_name_to_id_mapping['city']['Milwaukee'], 'Green Bay')
database_names_to_json_id['city']['georgia'] = create_double_dict(geojson_name_to_id_mapping['city']['Atlanta'], 'Atlanta')
database_names_to_json_id['city']['tampa-bay'] = create_double_dict(geojson_name_to_id_mapping['city']['Tampa'], 'Tampa Bay')
database_names_to_json_id['city']['oakland'] = create_double_dict(geojson_name_to_id_mapping['city']['San Francisco'], 'Oakland')

database_names_to_json_id

mapping_from_database_names_to_geojson_ids_filename = '../website/data/database_names_to_geojson_ids.json'

with open(mapping_from_database_names_to_geojson_ids_filename, 'w') as file:
        # indent = spaces per tab for pretty printing
        json.dump(database_names_to_json_id, file, indent=4)
        
get_ipython().system('cat {mapping_from_database_names_to_geojson_ids_filename}')







