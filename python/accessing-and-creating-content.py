from arcgis.gis import GIS
gis = GIS("portal url", "username", "password")

type(gis.content)

search_result = gis.content.search(query="title:Ports along west coast", item_type="Feature Layer")
search_result

# search and list all feature layers in my contents
search_result = gis.content.search(query="", item_type="Feature Layer")
search_result

search_my_contents = gis.content.search(query="owner:arcgis_python_api", item_type="csv")
search_my_contents

# search for content that begin with a prefix - say 'USA'
search_result_USA = gis.content.search(query="title:USA*")
search_result_USA

from IPython.display import display
for item in search_result_USA:
    display(item)

# lets get the itemid of first item from previous query
first_item = search_result_USA[0]
known_item_id = first_item.id
print(known_item_id)

# lets use the get() to access this item
online_banking_item = gis.content.get(known_item_id)
online_banking_item

# connect to ArcGIS Online
gis2 = GIS("https://www.arcgis.com", "username", "password")

public_3d_city_scenes = gis2.content.search(query="3d cities", item_type = "web scene",
                                           sort_field="numViews" ,sort_order="asc",
                                           max_items = 15, outside_org=True)
for item in public_3d_city_scenes:
    display(item)

csv_path = r"E:\GIS_Data\file_formats\CSV\world earthquakes.csv"
csv_properties={'title':'Earthquakes around the world from 1800s to early 1900s',
                'description':'Measurements from globally distributed seismometers',
                'tags':'arcgis, python, earthquake, natural disaster, emergency'}
thumbnail_path = r"E:\GIS_Data\file_formats\CSV\remote_sensor.png"

earthquake_csv_item = gis.content.add(item_properties=csv_properties, data=csv_path,
                                     thumbnail = thumbnail_path)

earthquake_csv_item

earthquake_feature_layer_item = earthquake_csv_item.publish()

earthquake_feature_layer_item

# read csv as a pandas dataframe
import pandas
ports_df = pandas.read_csv(r'E:\GIS_Data\file_formats\CSV\ports.csv')
ports_df

# find latitude of SFO
lat = ports_df.loc[ports_df.port_name == 'SAN FRANCISCO']['latitude']
lat

# only select ports that are to the south of SFO
ports_south_of_SFO = ports_df.loc[ports_df.latitude < lat[0]]
ports_south_of_SFO

ports_fc = gis.content.import_data(ports_south_of_SFO)
ports_fc

import json
ports_fc_dict = dict(ports_fc.properties)
ports_json = json.dumps(ports_fc_dict)

ports_item_properties = {'title': 'Ports to the south of SFO along west coast of USA',
                        'description':'Example demonstrating conversion of pandas ' + \
                         'dataframe object to a GIS item',
                        'tags': 'arcgis python api, pandas, csv',
                        'text':ports_json,
                        'type':'Feature Collection'}
ports_item = gis.content.add(ports_item_properties)
ports_item

# check if service name is available
gis.content.is_service_name_available(service_name= "awesome_python", service_type = 'featureService')

# let us publish an empty service
empty_service_item = gis.content.create_service(name='awesome_python', service_type='featureService')
empty_service_item

# access the layers property of the item
empty_service_item.layers

# create new folder
gis.content.create_folder(folder= 'ports')

# move the ports_item into this folder
ports_item.move(folder= 'ports')

# move back to root
ports_item.move(folder='/')

