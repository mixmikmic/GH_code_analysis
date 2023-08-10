from arcgis.gis import *

dev_gis = GIS("https://www.arcgis.com", "username", "password")

feature_layer_srch_results = dev_gis.content.search(query='title: "Griffith*" AND type: "Feature Service"')
feature_layer_srch_results

feature_layer_coll_item = feature_layer_srch_results[0]
feature_layer_coll_item

feature_layer_coll_item.share(everyone=True)

