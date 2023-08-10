# Establish a connection to your GIS.
from arcgis.gis import GIS
from IPython.display import display
gis = GIS() # anonymous connection to www.arcgis.com

# Search for 'USA major cities' feature layer collection
search_results = gis.content.search('title: USA Major Cities and owner:esri',
                                    'Feature Layer')

# Access the first Item that's returned
major_cities_item = search_results[0]

major_cities_item

major_cities_layers = major_cities_item.layers
major_cities_layers

freeways = gis.content.get('91c6a5f6410b4991ab0db1d7c26daacb')
freeways

freeways.layers 

for lyr in freeways.layers:
    print(lyr.properties.name)

from arcgis.features import FeatureLayerCollection

fs_url = 'http://sampleserver3.arcgisonline.com/ArcGIS/rest/services/SanFrancisco/311Incidents/FeatureServer'
sanfran = FeatureLayerCollection(fs_url)

sanfran.layers

sanfran.tables

from arcgis.features import FeatureLayer

lyr_url = 'http://sampleserver3.arcgisonline.com/ArcGIS/rest/services/SanFrancisco/311Incidents/FeatureServer/0'

layer = FeatureLayer(lyr_url)
layer

feature_layer = major_cities_layers[0]
feature_layer

feature_layer.properties.extent

feature_layer.properties.capabilities

feature_layer.properties.drawingInfo.renderer.type

for f in feature_layer.properties.fields:
    print(f['name'])

query_result1 = feature_layer.query(where='POP2007>1000000', 
                                    out_fields='WHITE,BLACK,MULT_RACE,HISPANIC')
len(query_result1.features)

query_result1.fields

feature_layer.query(where='POP2007>1000000', return_count_only=True)

query_result1.spatial_reference

query_result1.features[0].geometry

query_result1.spatial_reference

query2 = feature_layer.query(where="POP2007>1000000")
query2.df

query_geographic = feature_layer.query(where='POP2007>1000000', out_sr='4326')
query_geographic.features[0].geometry

major_cities_l1 = major_cities_layers[0]
major_cities_l1_fset = major_cities_l1.query(where= 'FID < 11')
type(major_cities_l1_fset)

major_cities_l1_features = major_cities_l1_fset.features
len(major_cities_l1_features)

major_cities_l1_features[0].geometry

major_cities_l1_features[0].attributes

search_fc = gis.content.search("title:AVL_Direct_FC", item_type='Feature Collection')
iowa_fc_item = search_fc[0]
iowa_fc_item

iowa_fc_item.layers

iowa_fc = iowa_fc_item.layers[0]

iowa_fset = iowa_fc.query()

iowa_features = iowa_fset.features
iowa_features[0].geometry

