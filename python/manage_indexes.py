from arcgis.gis import GIS
from arcgis.features import FeatureLayer

gis = GIS(username="mpayson_startups")

lyr = FeatureLayer("<MY LAYER URL>", gis=gis)

lyr.properties.indexes

# build serializable dictionary instead of PropertyMap
index_list = [dict(i) for i in lyr.properties.indexes]
update_dict = {"indexes": index_list}

# "updating" existing indexes will rebuild them
lyr.manager.update_definition(update_dict)

# see available fields
lyr.properties.fields

new_index = {
    "name" : "<MY INDEX NAME>", 
    "fields" : "<FIELD.name(s) TO INDEX>"
#     "isUnique" : False,
#     "isAscending" : False,
#     "description" : "MY INDEX" 
}
add_dict = {"indexes" : [new_index]}

lyr.manager.add_to_definition(add_dict)

