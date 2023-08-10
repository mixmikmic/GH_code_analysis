#connect to GIS
from arcgis.gis import GIS
gis = GIS("portal url", "username", "password")

#access an Item
earthquakes_item = gis.content.get('7ff6299367fa4a789bae81149b7ceae9')
earthquakes_item

# item id
earthquakes_item.id

# title
earthquakes_item.title

# tags
earthquakes_item.tags

# update the tags
earthquakes_item.update(item_properties={'tags':'python, seismometer, empirical, in-situ'})

earthquakes_item.tags

# updating thumbnail
earthquakes_item.update(thumbnail=r'E:\GIS_Data\file_formats\CSV\sensors2.jpg')

earthquakes_item

ports_csv_item = gis.content.get('a1623d78753a4213b1cc59790f54d15c')
ports_csv_item

ports_csv_item.get_data()

ports_csv_item.download_metadata(save_folder=r'E:\temp')

ports_csv_item.download_thumbnail(save_folder= r'E:\temp')

item_for_deletion = gis.content.get('a558ea98067c44049be3d2be18660774')
item_for_deletion

item_for_deletion.delete()

# let us protect the ports item we accessed earlier
ports_csv_item.protect(enable = True)

# attempting to delete will return an error
ports_csv_item.delete()

ports_feature_layer = gis.content.get('b0cb0c9f63e74e8480af0286eb9ac01f')
ports_feature_layer

ports_feature_layer.related_items('Service2Data', 'forward')

ports_csv_item.related_items('Service2Data', 'reverse')

webmap_item = gis.content.get('cc1876f1708e494d81a93014c1f56c58')
webmap_item

webmap_item.dependent_upon()

webmap_item.dependent_to()

#from the example above, use the item id of first relationship to get the related item
webmap_related_item = gis.content.get('31ad8b9a8ed3461992607eb8309816e2')
webmap_related_item

# add a relationship
webmap_item.add_relationship(rel_item= webmap_related_item, rel_type= 'Map2FeatureCollection')

webmap_related_item.dependent_to()

webmap_item.related_items('Map2FeatureCollection', 'forward')

