# from distance_request import *
from datetime import datetime
import googlemaps
import pandas as pd

# Read in call & ambulance files
calls_df = pd.read_csv('calls.csv', parse_dates=[3])
amb_df = pd.read_csv('ambulance_loc.csv')

# Select the most recent call and the available ambulances
current_call = calls_df.loc[calls_df.TIME == max(calls_df.TIME)]
available_amb = amb_df.loc[amb_df.AVAILABLE == 1]

available_amb.head()

current_call

# Get coordinates of most recent call and available ambulances
call_coord = [(current_call.LAT[0], current_call.LONG[0])]
amb_coord = [(row[2], row[3]) for row in available_amb.itertuples()]

# Use call time as proxy for departure time
dep_time = current_call.TIME[0]

# Get api key to make call
def get_api_key(filepath):
    with open(filepath) as f:
        content = f.readlines()
    # remove whitespace characters at the end of each line
    content = [x.strip() for x in content]
    key = content[0]
    return key
key = get_api_key('/Users/melaniecostello/Desktop/master_api_key.txt')
# key = get_api_key('/Users/melaniecostello/Desktop/maps_distance_key.txt')

# Make API call & store results in obj 'result'
gmaps = googlemaps.Client(key=key)
result = gmaps.distance_matrix(amb_coord, call_coord, mode="driving", units="imperial", departure_time=dep_time)

result

# Helper function to parse results
def parse_api_obj(result, available_amb, current_call):
    output_mat = pd.DataFrame()
    for idx, row in enumerate(result['rows']):
        row_mat = pd.DataFrame()
        mat = row['elements'][0]
        for key, val in mat.items():
            if key != 'status':
                df = pd.DataFrame.from_dict(val, orient='index')
                df = df.transpose()
                df.columns = [key + "_" + c for c in df.columns]
                if row_mat.empty:
                    row_mat = df
                else:
                    row_mat = pd.concat([row_mat, df], axis=1)
        if output_mat.empty:
            output_mat = row_mat
        else:
            output_mat = output_mat.append(row_mat)
#     output_mat.index = [amb for amb in available_amb.AMB_ID]
    output_mat['ambulance'] = [amb for amb in available_amb.AMB_ID]
    output_mat['amb_lat'] = [lat for lat in available_amb.LAT]
    output_mat['amb_long'] = [long for long in available_amb.LONG]
    output_mat['call_lat'] = current_call.LAT[0]
    output_mat['call_long'] = current_call.LONG[0]
    output_mat.reset_index(inplace=True, drop=True)
    return output_mat

# Get data frame of available ambulances
options = parse_api_obj(result, available_amb, current_call)

options

# Select ambulance for response based on desired metric
chosen = options.loc[options.duration_in_traffic_value == min(options.duration_in_traffic_value)].index[0]

chosen

# Get API key (same function as used above)
def get_api_key(filepath):
    with open(filepath) as f:
        content = f.readlines()
    # remove whitespace characters at the end of each line
    content = [x.strip() for x in content]
    key = content[0]
    return key

key = get_api_key('/Users/melaniecostello/Desktop/master_api_key.txt')

def get_coordinates(key, addr):
    """Runs google maps geocoding api to return lat/long coords
    for a list of addresses.
    key: string (API key)
    addr: list of strings (addresses)"""
    gmaps = googlemaps.Client(key=key)
    coords = []
    for ad in addr:
        geocode_result = gmaps.geocode(ad)
        lat_long = geocode_result[0]['geometry']['location']
        # Add tuple with lat & long to coords output
        coords.append((lat_long['lat'], lat_long['lng']))
    return coords
        
get_coordinates(key, ['NW 4th & Couch', '2500 NW Crosswater Ter, Beaverton OR', '4065 Paradise Valley Rd, Chinook MT'])



