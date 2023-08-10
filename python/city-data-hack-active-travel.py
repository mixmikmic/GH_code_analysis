import geopandas as gpd
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
get_ipython().magic("run '/Users/fcc/Documents/digital-connector-python/recipe.py'")
dc_dir = '/Documents/TomboloDigitalConnector'


# import geopandas as gpd
# import os
# from pathlib import Path
# from matplotlib import pyplot as plt

# home_dir = str(Path.home())

# %matplotlib inline
# %run os.path.join(home_dir, 'Desktop/python_library_dc/digital-connector-python/recipe.py')
# dc_dir = '/Desktop/TomboloDigitalConnector'

### recipe's subject

# Notice that we are using two more subjects here. These will serve different purposes. 
# We will use subject_la to subset the road network for the borough of Islington. We will use an intersect spatial operation
# to do that. 
subject_la = Subject(subject_type_label='localAuthority',provider_label='uk.gov.ons',
                  match_rule=Match_Rule(attribute_to_match_on="name", pattern="Islington"))

# This is our road network subject
subject = Subject(subject_type_label='space_syntax',provider_label='com.spacesyntax',
                  geo_match_rule=Geo_Match_Rule(geo_relation="intersects", subjects=[subject_la]))

subject_lsoa = Subject(subject_type_label='lsoa',provider_label='uk.gov.ons')

# subject_geometry_dft fetches the traffic counter locations represented as points
subject_geometry_dft = Subject(provider_label='uk.gov.dft',
                               subject_type_label='trafficCounter')


### recipe's datasources

la = Datasource(importer_class='uk.org.tombolo.importer.ons.OaImporter',
                            datasource_id='localAuthority')

openmap = Datasource(importer_class='uk.org.tombolo.importer.spacesyntax.OpenMappingImporter',
                            datasource_id='SpaceSyntaxOpenMapping')

trafficCounts = Datasource(importer_class='uk.org.tombolo.importer.dft.TrafficCountImporter',
                           datasource_id='trafficCounts',
                           geography_scope = ["London"]) ## Note that geography scope is specific

### First, lets get our attributes

countPedalCycles_attribute = AttributeMatcher(provider='uk.gov.dft',
                                     label='CountPedalCycles')

countCarTaxis_attribute = AttributeMatcher(provider='uk.gov.dft',
                                     label='CountCarsTaxis')

integration_2km = AttributeMatcher(label='integration2km',provider='com.spacesyntax')

### DC fields

# A very basic field that can handle numeric and time series attributes is LatestValueField. This essentially
# fetches the latest value within a time series if a time series exists. If not, it will just fetch the default value.
integration_2km_f = LatestValueField(attribute_matcher=integration_2km,
                                           label = 'Integration 2km')


count_pedal_cycles_f = LatestValueField(attribute_matcher=countCarTaxis_attribute,
                                          label='count_pedal_cycles')


count_car_taxis_f = LatestValueField(attribute_matcher=countPedalCycles_attribute,
                                          label='count_car_taxis')


# Next we need to assign the traffic counts to our subject
# As DfT traffic count geometry is points (the traffic count sensor location) we need to assign it to the nearest 
# road segment. For that we use MapToNearestSubjectField

m_count_pedal_cycles_f = MapToNearestSubjectField(field=count_pedal_cycles_f,
                                        label='Pedal traffic count',
                                        subject = subject_geometry_dft,
                                        max_radius = 1.)

m_count_car_taxis_f = MapToNearestSubjectField(field=count_car_taxis_f,
                                        label='Car/Taxi traffic count',
                                        subject = subject_geometry_dft,
                                        max_radius = 1.)
### running DC
dataset = Dataset(subjects=[subject],
                  fields=[integration_2km_f,
                          m_count_pedal_cycles_f,
                          m_count_car_taxis_f],
                  datasources=[la,
                               openmap,
                               trafficCounts])



recipe = Recipe(dataset,timestamp=False)
recipe.build_recipe(console_print=False)

recipe.run_recipe(tombolo_path='/Users/fcc/Documents/TomboloDigitalConnector',
                  clear_database_cache=False,
                  output_path = 'Documents/test.json')

gdf = gpd.read_file("/Users/fcc/Documents/test.json")
gdf.head()

vmin=gdf['Integration 2km'].min()
vmax=gdf['Integration 2km'].max()
    
ax = gdf.plot(column='Integration 2km', cmap='viridis',
              vmin=vmin,
              vmax=vmax)

# add colorbar
fig = ax.get_figure()
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
fig.colorbar(sm, cax=cax)


