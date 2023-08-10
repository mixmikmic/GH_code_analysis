#connect to GIS
from arcgis.gis import GIS
from IPython.display import display
gis = GIS("portal url", 'username', 'password')

#search for the feature layer named Ports along west coast
search_result = gis.content.search('title:Ports along west coast')
search_result[0]

#access the item's feature layers
ports_item = search_result[0]
ports_layers = ports_item.layers
ports_layers

#query all the features and display it on a map
ports_fset = ports_layers[0].query() #an empty query string will return all 
                                        #the features or the first 1000 which ever is smaller

ports_fset.df

ports_flayer = ports_layers[0]
ports_flayer.properties.capabilities

ports_features = ports_fset.features

# select San Francisco feature
sfo_feature = [f for f in ports_features if f.attributes['port_name']=='SAN FRANCISCO'][0]
sfo_feature.attributes

sfo_edit = sfo_feature
sfo_edit.attributes['short_form'] = 'SFO'

display(sfo_edit)

update_result = ports_flayer.edit_features(updates=[sfo_edit])
update_result

# construct a Feature object for Los Angeles.
la_dict = {"attributes": 
           {"latitude": 33.75,
            "longitude": -118.25,
            "country": "US",
            "harborsize": "L",
            "label_position": "SW",
            "port_name": "LOS ANGELES",
            "short_form": "LAX"}, 
           "geometry": 
           {"x": -13044788.958999995, "y": 3857756.351200014}}

add_result = ports_flayer.edit_features(adds = [la_dict])

add_result

# find object id for Redlands
Redlands_feature = [f for f in ports_features if f.attributes['port_name'] == 'REDLANDS'][0]
Redlands_objid = Redlands_feature.get_value('objectid')
Redlands_objid

type(Redlands_objid)

# pass the object id as a string to the delete parameter
delete_result = ports_flayer.edit_features(deletes=str(Redlands_objid))
delete_result

ports_fset_edited = ports_flayer.query()
ports_fset_edited.df

