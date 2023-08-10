# connect to GIS
from arcgis.gis import GIS
gis = GIS("portal url", "username", "password")

#search for earthquakes data - point data
eq_search = gis.content.search("world earthquakes", "feature layer", max_items=1)
eq_item = eq_search[0]
eq_item

# search for USA states - area / polygon data
states_search = gis.content.search("title:USA_states and owner:arcgis_python_api", 
                                   "feature layer", max_items=1)
states_item = states_search[0]
states_item

map1 = gis.map("USA")
map1

map1.add_layer(states_item)

map1.add_layer(eq_item)

eq_fl = eq_item.layers[0]
states_fl = states_item.layers[0]

#query the fields in eq_fl layer
for field in eq_fl.properties.fields:
    print(field['name'])

# similarly for states data
for field in states_fl.properties.fields:
    print(field['name'], end="\t")

from arcgis.features import summarize_data
sum_fields = ['magnitude Mean', 'depth Min']
eq_summary = summarize_data.aggregate_points(point_layer = eq_fl,
                                            polygon_layer = states_fl,
                                            keep_boundaries_with_no_points=False,
                                            summary_fields=sum_fields)

eq_summary

# access the aggregation feature colleciton
eq_aggregate_fc = eq_summary['aggregated_layer']

#query this feature collection to get a data as a feature set
eq_aggregate_fset = eq_aggregate_fc.query()

aggregation_df = eq_aggregate_fset.df
aggregation_df

aggregation_df.plot('state_name','Point_Count', kind='bar')

aggregation_df.plot('state_name',['MEAN_magnitude', 'MIN_depth'],kind='bar', subplots=True)

