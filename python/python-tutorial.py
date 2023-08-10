### We will be using geopandas and matplotlib to visualise our output
import geopandas as gpd
import os
from pathlib import Path
from matplotlib import pyplot as plt

home_dir = str(Path.home())

get_ipython().magic('matplotlib inline')

### Please replace your path to the recipe.py file located in the digital-connector-python
get_ipython().magic("run os.path.join(home_dir, 'Desktop/python_library_dc/digital-connector-python/recipe.py')")
dc_dir = '/Desktop/TomboloDigitalConnector'

### subject_geometry is the geometry of our final exporter file. We specify the geometry level and we 
### define a match rule that extracts the Greater London area geometry. 
subject_geometry = Subject(subject_type_label='localAuthority',provider_label='uk.gov.ons',
                  match_rule=Match_Rule(attribute_to_match_on="label", pattern="E090%"))

### localAuthority is the geometry level that we are interested in. 

localAuthority = Datasource(importer_class='uk.org.tombolo.importer.ons.OaImporter',
                            datasource_id='localAuthority')


### We will wrap the above in a Dataset object and build the recipe. Notice that there are no fields yet as we haven't 
### specified any
dataset = Dataset(subjects=[subject_geometry], fields=[],
                  datasources=[localAuthority])

recipe = Recipe(dataset)
recipe.build_recipe(output_location="test.json",
                   console_print=True)

dc_dir = '/Desktop/TomboloDigitalConnector'
recipe.run_recipe(tombolo_path=home_dir + dc_dir,
                  clear_database_cache=False,
                  output_path = home_dir + '/local_authority.json')

gdf = gpd.read_file(home_dir + '/local_authority.json')
gdf.to_crs(epsg=27700).plot()

### This is what the output file looks like

gdf.head()

### First we define our subject geometry as before
subject_geometry = Subject(subject_type_label='localAuthority', provider_label='uk.gov.ons',
                  match_rule=Match_Rule(attribute_to_match_on="label", pattern="E090%"))

### We then specify our importers

localAuthority = Datasource(importer_class='uk.org.tombolo.importer.ons.OaImporter',
                            datasource_id='localAuthority')

trafficCounts = Datasource(importer_class='uk.org.tombolo.importer.dft.TrafficCountImporter',
                           datasource_id='trafficCounts',
                           geography_scope = ["London"]) ## Note that geography scope is specific to that importer

airQualityControl = Datasource(importer_class='uk.org.tombolo.importer.lac.LAQNImporter',
                               datasource_id='airQualityControl')

### For our convenience we will wrap all our importers in a list

importers = [localAuthority, trafficCounts, airQualityControl]

### We grab the attribute from our importer for NO2. We do that by invoking an AttributeMatcher which will essentially look 
### look through the database and find the attribute with the corresponding label. 
### Notice that we only need to specify the provider and the attribute label, and the attribute label should
### be the SAME as it appears in the importer (and as an result, in the psql database).

no_2_attribute = AttributeMatcher(provider='erg.kcl.ac.uk',
                                     label='NO2 40 ug/m3 as an annual mean')

### We then pass the attribute to a LatestValueField field. Digital Connector can store timeseries within the 
### LatestValueField, but we are only interested in the most recent one. Invoking LatestValueField 
### will allow us to extract the latest value of the time series. We can now use any label we wish to name the field.
### For consistency, the label is the same as the attribute in the importer

no_2_field = LatestValueField(attribute_matcher=no_2_attribute,
                         label='NO2 40 ug/m3 as an annual me')

subject_geometry_laq = Subject(provider_label='erg.kcl.ac.uk',subject_type_label='airQualityControl')

g_no_2_field = GeographicAggregationField(field=no_2_field,
                                          function='mean',
                                          label='NitrogenDioxide',
                                          subject = subject_geometry_laq)

dataset = Dataset(subjects=[subject_geometry], fields=[g_no_2_field],
                  datasources=importers)
recipe = Recipe(dataset)
recipe.build_recipe(output_location=None,
                   console_print=True)

### First, lets get our attributes

countPedalCycles_attribute = AttributeMatcher(provider='uk.gov.dft',
                                     label='CountPedalCycles')

countPedalCycles_field = LatestValueField(attribute_matcher=countPedalCycles_attribute,
                                          label='CountPedalCycles')

countCarTaxis_attribute = AttributeMatcher(provider='uk.gov.dft',
                                     label='CountCarsTaxis')

countCarTaxis_field = LatestValueField(attribute_matcher=countCarTaxis_attribute,
                                          label='CountCarsTaxis')



### Since we will be doing the same operation twice, one for the cycles and one for the cars, it is convenient
### to use a for loop on the attribute labels.
fields = ['countPedalCycles_field','countCarTaxis_field']

f={}

subject_geometry_dft = Subject(provider_label='uk.gov.dft',
                               subject_type_label='trafficCounter')
for i in fields:
    f['geo_{0}'.format(i)] = GeographicAggregationField(subject= subject_geometry_dft,
                                                           field=eval(('{0}').format(i)),
                                                           function='sum',
                                                           label='geo_{0}'.format(i))

    
    

bicycleFraction = ArithmeticField(operation_on_field_1=f['geo_countPedalCycles_field'],
                                  operation_on_field_2=f['geo_countCarTaxis_field'],
                                  operation='div',
                                  label='BicycleFraction')

dataset = Dataset(subjects=[subject_geometry], fields=[g_no_2_field, bicycleFraction],
                  datasources=importers)

recipe = Recipe(dataset)
recipe.build_recipe(output_location=None,
                   console_print=False)

recipe.run_recipe(tombolo_path=home_dir + dc_dir,
                  clear_database_cache=False,
                  output_path = home_dir + '/london-cycle-traffic-air-quality.json')

gdf = gpd.read_file(home_dir + '/london-cycle-traffic-air-quality.json')
gdf.to_crs(epsg=27700).plot(column='NitrogenDioxide',
                        cmap='viridis', linewidth=0.1, legend=True)

gdf.to_crs(epsg=27700).plot(column='BicycleFraction',
                        cmap='viridis', linewidth=0.1, legend=True)



### First lets specify our subject geometries. We will be using the london local authority boundaries to subset our
### OSM and air quality subjects by using a geo_relation='within'. Have a look at the Digital Connector codebase for
### other spatial join relationships
subject_geometry_la = Subject(subject_type_label='localAuthority', provider_label='uk.gov.ons',
                  match_rule=Match_Rule(attribute_to_match_on="label", pattern="E090000%"))

subject_geometry_osm = Subject(provider_label='org.openstreetmap',
                           subject_type_label='OSMEntity',
                           geo_match_rule=Geo_Match_Rule(geo_relation='within',subjects=[subject_geometry_la]))

subject_geometry_laq = Subject(provider_label='erg.kcl.ac.uk',subject_type_label='airQualityControl',
                              geo_match_rule=Geo_Match_Rule(geo_relation='within',subjects=[subject_geometry_la]))


### We can now define our importers. 

osm_importer = Datasource(importer_class='uk.org.tombolo.importer.osm.OSMImporter',
                          datasource_id='OSMHighways',
                          geography_scope = ["europe/great-britain/england/greater-london"])

airQualityControl = Datasource(importer_class='uk.org.tombolo.importer.lac.LAQNImporter',
                               datasource_id='airQualityControl')

localAuthority = Datasource(importer_class='uk.org.tombolo.importer.ons.OaImporter',
                            datasource_id='localAuthority')


### Lets build our attributes. This time we will be usig a MapToNearestSubjectField to assign the values of 
### london air quality data to our export subject geometry which in this case is the OSM geometries with "highways" tag

no_2_attribute = AttributeMatcher(provider='erg.kcl.ac.uk',
                                     label='NO2 40 ug/m3 as an annual mean')

no_2_field = LatestValueField(attribute_matcher=no_2_attribute,
                         label='NO2 40 ug/m3 as an annual me')

g_no_2_field = MapToNearestSubjectField(field=no_2_field,
                                        label='NitrogenDioxide',
                                        subject = subject_geometry_laq,
                                        max_radius = 1.)

### For demonstration purposes, we assign a constant value of "1" to all OSM elements. This might be usefull in 
### other applications such as counting the number of elements within a polygon during a geographic aggregation. 
### We use another field class to achieve that which is the FixedAnnotationField. There are many more field classes
### to explore, details can be found in the Digital Connector codebase.

osm_roads_field = AttributeMatcherField(field=ConstantField(value="1"),
                                        attributes=[AttributeMatcher(provider='org.openstreetmap',label='highway')],
                                        label='highway')

dataset = Dataset(subjects=[subject_geometry_osm], fields=[osm_roads_field, g_no_2_field],
                  datasources=[localAuthority, osm_importer, airQualityControl])

recipe = Recipe(dataset)
recipe.build_recipe(output_location=None,
                   console_print=True)

recipe.run_recipe(tombolo_path=home_dir + dc_dir,
                  clear_database_cache=False,
                  output_path = home_dir + '/osm-air-quality.json')

import geopandas as gpd
gdf = gpd.read_file(home_dir + '/osm-air-quality.json')
gdf.head()

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
gdf.to_crs(epsg=27700).plot(column='NitrogenDioxide',
                        cmap='viridis', linewidth=0.1, legend=True)


