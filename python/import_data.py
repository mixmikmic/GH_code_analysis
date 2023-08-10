import getpass

password = getpass.getpass("Enter password, please: ")
gis = GIS('https://arcgis.com', 'username', password)

parks_properties = {'title': 'Parks and Open Space',
                   'tags': 'parks, open data, devlabs',
                   'type': 'Shapefile'}
parks_shp = gis.content.add(parks_properties,
                               data='./LA_Hub_datasets/Parks_and_Open_Space.zip')

parks_shp

parks_feature_layer = parks_shp.publish()

parks_feature_layer.url

