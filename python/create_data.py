import getpass
from arcgis.gis import *

password = getpass.getpass("Please enter password: ")
dev_gis = GIS('https://www.arcgis.com', 'username', password)
print("Successfully logged in to {} as {}".format(dev_gis.properties.urlKey + '.' + dev_gis.properties.customBaseUrl,
                                                 dev_gis.users.me.username))

feature_layer_srch_results = dev_gis.content.search(query='title: "Griffith*" AND type: "Feature Service"', 
                                         max_items=10)
feature_layer_srch_results

feature_layer_coll_item = feature_layer_srch_results[0]
feature_layers = feature_layer_coll_item.layers
feature_layer = feature_layers[0]
feature_layer.properties.name

for field in feature_layer.properties['fields']:
    print(field['name'])

from arcgis import geometry
from arcgis import features

def create_feature(map1, g):
    try:
        oid = 1
        pt = geometry.Point(g)
        feat = features.Feature(geometry=pt, attributes={'OBJECTID': 1,
                                                        'name': 'name',
                                                        'type': 'park',
                                                        'surface': 'dirt'})
        feature_layer.edit_features(adds=[feat])
        print(str(g))
        map1.draw(g)
    except:
        print("Couldn't create the feature. Try again, please...")

map1 = dev_gis.map('Los Angeles', 10)
map1

map1.on_click(create_feature)

map1.clear_graphics()

map1.add_layer(feature_layer)

