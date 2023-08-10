import json

import arcgis
from IPython import display
import pandas as pd
from shapely import geometry

gis = arcgis.gis.GIS()

# For an explanation see markdown cell below.
SIMPLIFICATION_TOLERANCE = 0.05

API_URL = 'https://idpgis.ncep.noaa.gov/arcgis/rest/services/NWS_Forecasts_Guidance_Warnings/watch_warn_adv/MapServer/1'
WATCHES_JSON_PATH = '../data/watches.json'

# TODO: Uncomment once the API is fixed.
# layer = arcgis.features.FeatureLayer(API_URL, gis)
# events = [e.as_dict for e in layer.query().features]

watches = json.load(open(WATCHES_JSON_PATH))
print('%d watches loaded from API' % len(watches))

watches_attrs = pd.DataFrame([watch['attributes'] for watch in watches])
watches_attrs.head(2).transpose()

watches_attrs.prod_type.value_counts()

for watch in watches[:10]:
    display.display(geometry.Polygon(watch['geometry']['rings'][0]))
    print(watch['attributes']['prod_type'])

len(watches[0]['geometry']['rings'][0])

len(watches[3]['geometry']['rings'][0])

def shapely_coords_to_list(shaply_coords):
    return list(zip(shaply_coords[0], shaply_coords[1]))

polygon = geometry.Polygon(watches[0]['geometry']['rings'][0])
simplified_polygon = polygon.simplify(SIMPLIFICATION_TOLERANCE)
display.display(polygon)
print('The original polygon has %d points' % len(watches[0]['geometry']['rings'][0]))
display.display(simplified_polygon)
print('The simplified polygon has %d points' % len(shapely_coords_to_list(simplified_polygon.exterior.coords.xy)))

