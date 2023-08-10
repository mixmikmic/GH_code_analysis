# We'll use this to output some of our intermediate data structures in a somewhat standard format
import json

# We'll use this to break our address into street, city, state pieces for us
import usaddress

# Lots of meetings are at the Cook County Building.  This is one example from the data.
location_name = "Cook County Building, Board Room, 118 North Clark Street, Chicago, Illinois"

# Many geocoders, especially free ones need us to break the address into pieces.
# Let's use the awesome usaddress package to do this for us
parsed_location_name, detected_type = usaddress.tag(location_name)

print(json.dumps(parsed_location_name, indent=4))

# Let's merge these very atomic fields into ones that are closer to those
# accepted by many geocoding APIs
street_address = ' '.join([
    parsed_location_name['AddressNumber'],
    parsed_location_name['StreetNamePreDirectional'],
    parsed_location_name['StreetName'],
    parsed_location_name['StreetNamePostType']
])
address = {
    'street_address': street_address,
    'city': parsed_location_name['PlaceName'],
    'state': parsed_location_name['StateName']
}

print(json.dumps(address, indent=4))

import csv
import io

# We'll use this to simplify making our HTTP requests to the geocoding APIs
import requests

# Let's merge our parsed address into fields that are

# We'll use the Texas A&M Geoservices API because it's free (for few requests)
# and has pretty liberal terms of service.  It does however require an API
# key.
TAMU_GEOSERVICES_API_KEY = os.environ.get('TAMU_GEOSERVICES_API_KEY')

# API parameters, passed via the URL
# That is:
# https://geoservices.tamu.edu/Services/Geocode/WebService/GeocoderWebServiceHttpNonParsed_V04_01.aspx?version=4.01&streetAddres=118+North+Clark+Street...
api_params = {
    'apiKey': TAMU_GEOSERVICES_API_KEY,
    'version': '4.01',
    'streetAddress': address['street_address'],
    'city': address['city'],
    'state': address['state'],
    'includeHeader': 'true',
}

api_url = 'https://geoservices.tamu.edu/Services/Geocode/WebService/GeocoderWebServiceHttpNonParsed_V04_01.aspx'
r = requests.get(api_url, params=api_params)


# The TAMU geocoder returns the data as CSV.  We need to parse this out

geocoded = None
inf = io.StringIO(r.text)
reader = csv.DictReader(inf)

for row in reader:
    geocoded = row
    # Only read one row
    break
    
print(json.dumps(geocoded, indent=2))

# Our longitude and latitude are available as keys of the parsed dictionary
print("[{0}, {1}]".format(geocoded['Longitude'], geocoded['Latitude']))

from datetime import datetime

from pytz import timezone

start_time = "March 8, 2017 5:00pm"

parsed_start_time = datetime.strptime(start_time, "%B %d, %Y %I:%M%p")

# Convert to Chicago's time zone
central = timezone('US/Central')
parsed_start_time = central.localize(parsed_start_time)

print(parsed_start_time.isoformat())

# http://colorbrewer2.org/?type=sequential&scheme=BuGn&n=5
COLORS_BLUE_GREEN = [
    '#edf8fb',
    '#b2e2e2',
    '#66c2a4',
    '#2ca25f',
    '#006d2c',
]

import os

import requests

# Read our API key from an environment variable.
# This is a good way to avoid hard-coding credentials like API keys in our code.
MAPZEN_API_KEY = os.environ.get('MAPZEN_API_KEY')

# See https://mapzen.com/documentation/mobility/isochrone/api-reference/ for the API reference
# These are API parameters we'll use for all of the transportation types we're trying
base_api_params = {
    'api_key': MAPZEN_API_KEY,
    # Location and costing (transportation type)
    'json': {
        'locations': [
            {
                'lat': geocoded['Latitude'],
                'lon': geocoded['Longitude'],
            },
        ], 
        # Contours
        # You can have a maximum of 4
        'contours': [
            {
                'time': 30,
                'color': COLORS_BLUE_GREEN[0].lstrip('#'),
            },
            {
                'time': 60,
                'color': COLORS_BLUE_GREEN[2].lstrip('#'),
            },
            {
                'time': 90,
                'color': COLORS_BLUE_GREEN[4].lstrip('#'),
            },
        ],
        # Return GeoJSON polygons instead of linestrings.
        # Linestrings are smaller and faster to render, but we use polygons for colored shading.
        'polygons': True,
        # Distance in meters used as an input to simplify the polygons.
        # A higher number will result in more simplifications.
        # I'm doing this here, because without some simplification, the GeoJSON is too long
        # to pass as a URL parameter to geojson.io, which we're using to display our maps.
        'generalize': 300,
    },
}

from copy import deepcopy
import datetime
import json

api_params = deepcopy(base_api_params)

# Calculate a start time 1 hour before the meeting time.  This is hacky, but the API doesn't support an arrival time
# yet.
departure_time = parsed_start_time - datetime.timedelta(hours=1)

# Specify transit/walking
api_params['json']['costing'] = 'multimodal'
api_params['json']['date_time'] = {
    # `1` means specified departure time.  Ideally we would use `2`, the arrival time, but the API docs say this
    # isn't supported yet.
    'type': 1,
    'value': departure_time.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M"),
}

# Convert `json` parameter to JSON
api_params['json'] = json.dumps(api_params['json'])

r = requests.get('https://matrix.mapzen.com/isochrone', params=api_params)

contour_geojson = json.dumps(r.json())

import geojsonio

# Embed a map in this notebook
# This should work, but doesn't.  Maybe it's just a problem on Linux
#geojsonio.embed(contour_geojson)

geojsonio.display(contour_geojson)

from copy import deepcopy
import json

api_params = deepcopy(base_api_params)

# Specify bicycling costing methods
# There are a lot more options, just use the defaults for now
# See https://mapzen.com/documentation/mobility/turn-by-turn/api-reference/#bicycle-costing-options
api_params['json']['costing'] = 'bicycle'

# Convert `json` parameter to JSON
api_params['json'] = json.dumps(api_params['json'])

r = requests.get('https://matrix.mapzen.com/isochrone', params=api_params)

contour_geojson = json.dumps(r.json())

geojsonio.display(contour_geojson)

import os

import requests

# Let's get an isochrone using the HERE Routing API Isoline endpoint

HERE_APP_ID = os.environ.get('HERE_APP_ID')
HERE_APP_CODE = os.environ.get('HERE_APP_CODE')

api_params = {
    'app_id': HERE_APP_ID,
    'app_code': HERE_APP_CODE,
    'mode': "fastest;car;traffic:enabled",
    'destination': "geo!{lat},{lng}".format(lat=geocoded['Latitude'], lng=geocoded['Longitude']),
    'arrival': parsed_start_time.isoformat(),
    # 1 hour
    'range': 1 * 60 * 60,
    'rangetype': 'time'
}

r = requests.get("https://isoline.route.cit.api.here.com/routing/7.2/calculateisoline.json", params=api_params)

isoline_response = r.json()['response']

# Convert the isoline API response to GeoJSON
import json

isoline_geojson = {
    'type': 'FeatureCollection',
    'features': [],
}

isoline_geojson['features'].append({
    'type': 'Feature',
    'geometry': {
        'type': 'Point',
        'coordinates': [
            isoline_response['center']['longitude'],
            isoline_response['center']['latitude']
        ],
    },
    'properties': {},
})

isoline_geojson['features'].append({
    'type': 'Feature',
    'geometry': {
        'type': 'Polygon',
        'coordinates': [
            [
                [float(c) for c in x.split(',')[::-1]]
                for x
                in isoline_response['isoline'][0]['component'][0]['shape']
            ],
        ],
    },
    'properties': {},
})

# And display it using geojson.io

import geojsonio
geojsonio.display(json.dumps(isoline_geojson))

# Factor this example into functions so we can get multiple areas

import requests

def get_isoline(lat, lng, mode, rng, arrival, app_id, app_code):
    api_params = {
        'app_id': app_id,
        'app_code': app_code,
        'mode': mode,
        'destination': "geo!{lat},{lng}".format(lat=lat, lng=lng),
        'arrival': arrival.isoformat(),
        'range': rng,
        'rangetype': 'time'
    }

    r = requests.get("https://isoline.route.cit.api.here.com/routing/7.2/calculateisoline.json", params=api_params)

    return r.json()['response']

def isoline_geojson_feature(isoline):
    return {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [
                [
                    [float(c) for c in x.split(',')[::-1]]
                    for x
                    in isoline['isoline'][0]['component'][0]['shape']
                ],
            ],
        },
        'properties': {},
    }


ranges = [
    {
        'range': 1 * 60 * 60,
        'properties': {
            'fill': COLORS_BLUE_GREEN[0],
        },
    },
    {
        'range': 1 * 60 * 30,
        'properties': {
            'fill': COLORS_BLUE_GREEN[1],
        },
    },
]

isolines = {
    'type': 'FeatureCollection',
    'features': [],
}

isolines['features'].append({
    'type': 'Feature',
    'geometry': {
        'type': 'Point',
        'coordinates': [
            float(geocoded['Longitude']),
            float(geocoded['Latitude']), 
        ],
    },
    'properties': {},
})

for rng in ranges:
    isoline = get_isoline(
        geocoded['Latitude'], 
        geocoded['Longitude'], 
        'fastest;car;traffic:enabled',
        rng['range'],
        parsed_start_time,
        HERE_APP_ID,
        HERE_APP_CODE
    )

    isoline_feature = isoline_geojson_feature(isoline)
    isoline_feature['properties'].update(**rng['properties'])
    isoline_feature['properties']['range'] = rng['range']
    isolines['features'].append(isoline_feature)

# And display it using geojson.io

import geojsonio

# TODO: Simplify GeoJSON because it's too long to pass to geojson.io via the URL
geojsonio.display(json.dumps(isolines))

print(json.dumps(isolines))



