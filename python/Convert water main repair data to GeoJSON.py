STREET_SEGMENT_DATA_URL = "http://graphics.chicagotribune.com/lead-water/data/geocoded_water_projects.json"

# Download the data

import requests
r = requests.get(STREET_SEGMENT_DATA_URL)
street_segments = r.json()

# Convert to GeoJSON

def point_geometry(segment):
    return {
        'type': 'Point',
        'coordinates': [segment['points'][0]['lng'], segment['points'][0]['lat']],
    }

def linestring_geometry(segment):
    return {
        'type': 'LineString',
        'coordinates':  [
            [segment['points'][0]['lng'], segment['points'][0]['lat'],],
            [segment['points'][1]['lng'], segment['points'][1]['lat'],],
        ],
    }

def to_geojson(segment):
    assert segment['type'] in ('line', 'point'), segment['type']
    assert len(segment['points']) >= 1 and len(segment['points']) <= 2
    segment_geojson = {
        'type': 'Feature',
        'properties': {
            'from': segment['From'],
            'to': segment['To'],
            'project_number': segment['Project #'],
            'pipe_completion_date': segment['Pipe Completion Date'],
            'location': segment['Location'],
            'construction_start_date': segment['Construction Start Date'],
            'year': segment['Year'],
            'description': segment['description'],
        },
        'geometry': linestring_geometry(segment) if segment['type'] == 'line' else point_geometry(segment),
    }
    
    return segment_geojson
    
street_segements_geojson = {
    'type': 'FeatureCollection',
    'features': [to_geojson(ss) for ss in street_segments],
}

# Output JSON string
import json
import os 

working_dir = os.getcwd()
output_path = os.path.join(working_dir, 'street_segments.geojson')

with open(output_path, 'w') as f:
    f.write(json.dumps(street_segements_geojson))
    print("Wrote {0}".format(output_path))



