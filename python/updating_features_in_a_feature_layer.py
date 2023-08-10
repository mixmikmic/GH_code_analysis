# Connect to the GIS
from arcgis.gis import GIS
from arcgis import features
from getpass import getpass #to accept passwords in an interactive fashion
import pandas as pd

#Access the portal using "amazing_arcgis_123" as password for the given Username. 
password = getpass()
gis = GIS("https://python.playground.esri.com/portal", "arcgis_python", password)

# read the initial csv
csv1 = 'data/updating_gis_content/capitals_1.csv'
cities_df_1 = pd.read_csv(csv1)
cities_df_1.head()

# print the number of records in this csv
cities_df_1.shape

# add the initial csv file and publish that as a web layer
item_prop = {'title':'USA Capitals spreadsheet'}
csv_item = gis.content.add(item_properties=item_prop, data=csv1)
csv_item

# publish the csv item into a feature layer
cities_item = csv_item.publish()
cities_item

# update the item metadata
item_prop = {'title':'USA Capitals'}
cities_item.update(item_properties = item_prop, thumbnail='data/updating_gis_content/capital_cities.png')
cities_item

# read the second csv set
csv2 = 'data/updating_gis_content/capitals_2.csv'
cities_df_2 = pd.read_csv(csv2)
cities_df_2.head()

# get the dimensions of this csv
cities_df_2.shape

cities_flayer = cities_item.layers[0]
cities_fset = cities_flayer.query() #querying without any conditions returns all the features
cities_fset.df.head()

overlap_rows = pd.merge(left = cities_fset.df, right = cities_df_2, how='inner',
                       on = 'city_id')
overlap_rows

features_for_update = [] #list containing corrected features
all_features = cities_fset.features

# inspect one of the features
all_features[0]

# get the spatial reference of the features since we need to update the geometry
cities_fset.spatial_reference

from arcgis import geometry #use geometry module to project Long,Lat to X and Y
from copy import deepcopy

for city_id in overlap_rows['city_id']:
    # get the feature to be updated
    original_feature = [f for f in all_features if f.attributes['city_id'] == city_id][0]
    feature_to_be_updated = deepcopy(original_feature)
    
    print(str(original_feature))
    
    # get the matching row from csv
    matching_row = cities_df_2.where(cities_df_2.city_id == city_id).dropna()
    
    #get geometries in the destination coordinate system
    input_geometry = {'y':float(matching_row['latitude']),
                       'x':float(matching_row['longitude'])}
    output_geometry = geometry.project(geometries = [input_geometry],
                                       in_sr = 4326, 
                                       out_sr = cities_fset.spatial_reference['latestWkid'],
                                      gis = gis)
    
    # assign the updated values
    feature_to_be_updated.geometry = output_geometry[0]
    feature_to_be_updated.attributes['longitude'] = float(matching_row['longitude'])
    feature_to_be_updated.attributes['city_id'] = int(matching_row['city_id'])
    feature_to_be_updated.attributes['state'] = matching_row['state'].values[0]
    feature_to_be_updated.attributes['capital'] = matching_row['capital'].values[0]
    feature_to_be_updated.attributes['latitude'] = float(matching_row['latitude'])
    feature_to_be_updated.attributes['name'] = matching_row['name'].values[0]
    feature_to_be_updated.attributes['pop2000'] = int(matching_row['pop2000'])
    feature_to_be_updated.attributes['pop2007'] = int(matching_row['pop2007'])
    
    #add this to the list of features to be updated
    features_for_update.append(feature_to_be_updated)
    
    print(str(feature_to_be_updated))
    print("========================================================================")

features_for_update

cities_flayer.edit_features(updates= features_for_update)

#select those rows in the capitals_2.csv that do not overlap with those in capitals_1.csv
new_rows = cities_df_2[~cities_df_2['city_id'].isin(overlap_rows['city_id'])]
print(new_rows.shape)

new_rows.head()

features_to_be_added = []

# get a template feature object
template_feature = deepcopy(features_for_update[0])

# loop through each row and add to the list of features to be added
for row in new_rows.iterrows():
    new_feature = deepcopy(template_feature)
    
    #print
    print("Creating " + row[1]['name'])
    
    #get geometries in the destination coordinate system
    input_geometry = {'y':float(row[1]['latitude']),
                       'x':float(row[1]['longitude'])}
    output_geometry = geometry.project(geometries = [input_geometry],
                                       in_sr = 4326, 
                                       out_sr = cities_fset.spatial_reference['latestWkid'],
                                      gis = gis)
    
    # assign the updated values
    new_feature.geometry = output_geometry[0]
    new_feature.attributes['longitude'] = float(row[1]['longitude'])
    new_feature.attributes['city_id'] = int(row[1]['city_id'])
    new_feature.attributes['state'] = row[1]['state']
    new_feature.attributes['capital'] = row[1]['capital']
    new_feature.attributes['latitude'] = float(row[1]['latitude'])
    new_feature.attributes['name'] = row[1]['name']
    new_feature.attributes['pop2000'] = int(row[1]['pop2000'])
    new_feature.attributes['pop2007'] = int(row[1]['pop2007'])
    
    #add this to the list of features to be updated
    features_to_be_added.append(new_feature)

# take a look at one of the features we created
features_to_be_added[0]

cities_flayer.edit_features(adds = features_to_be_added)

# read the third csv set
csv3 = 'data/updating_gis_content/capitals_annex.csv'
cities_df_3 = pd.read_csv(csv3)
cities_df_3.head()

#find the number of rows in the third csv
cities_df_3.shape

#Get the existing list of fields on the cities feature layer
cities_fields = cities_flayer.manager.properties.fields

# Your feature layer may have multiple fields, 
# instead of printing all, let us take a look at one of the fields:
cities_fields[1]

for field in cities_fields:
    print(field.name, "\t|\t", field.alias, "\t|\t", field.type, "\t|\t", field.sqlType)

# get a template field
template_field = dict(deepcopy(cities_fields[1]))
template_field

# get the list of new fields to add from the third spreadsheet, that are not in spread sheets 1,2
new_field_names = list(cities_df_3.columns.difference(cities_df_1.columns))
new_field_names

fields_to_be_added = []
for new_field_name in new_field_names:
    current_field = deepcopy(template_field)
    if new_field_name.lower() == 'class':
        current_field['sqlType'] = 'sqlTypeVarchar'
        current_field['type'] = 'esriFieldTypeString'
        current_field['length'] = 8000
        
    current_field['name'] = new_field_name.lower()
    current_field['alias'] = new_field_name
    fields_to_be_added.append(current_field)
    
len(fields_to_be_added)

#inspect one of the fields
fields_to_be_added[3]

cities_flayer.manager.add_to_definition({'fields':fields_to_be_added})

new_cities_fields = cities_flayer.manager.properties.fields
len(new_cities_fields)

for field in new_cities_fields:
    print(field.name + "\t|\t", field.type)

# Run a fresh query on the feature layer so it includes the new features from
# csv2 and new columns from csv3
cities_fset2 = cities_flayer.query()
cities_features2 = cities_fset2.features

features_for_update = []
for city_id in cities_df_3['city_id']:
    # get the matching row from csv
    matching_row = cities_df_3.where(cities_df_3.city_id == city_id).dropna()
    
    print(str(city_id) + " Adding additional attributes for: " + matching_row['name'].values[0])
    
    # get the feature to be updated
    original_feature = [f for f in cities_features2 if f.attributes['city_id'] == city_id][0]
    feature_to_be_updated = deepcopy(original_feature)
    
    # assign the updated values
    feature_to_be_updated.attributes['class'] = matching_row['class'].values[0]
    feature_to_be_updated.attributes['white'] = int(matching_row['white'])
    feature_to_be_updated.attributes['black'] = int(matching_row['black'])
    feature_to_be_updated.attributes['ameri_es'] = int(matching_row['ameri_es'])
    feature_to_be_updated.attributes['asian'] = int(matching_row['asian'])
    feature_to_be_updated.attributes['hawn_pl'] = int(matching_row['hawn_pl'])
    feature_to_be_updated.attributes['hispanic'] = int(matching_row['hispanic'])
    feature_to_be_updated.attributes['males'] = int(matching_row['males'])
    feature_to_be_updated.attributes['females'] = int(matching_row['females'])
    
    #add this to the list of features to be updated
    features_for_update.append(feature_to_be_updated)

# inspect one of the features
features_for_update[-1]

# apply the edits to the feature layer
cities_flayer.edit_features(updates= features_for_update)

cities_fset3 = cities_flayer.query()
cities_fset3.df.head(5)

