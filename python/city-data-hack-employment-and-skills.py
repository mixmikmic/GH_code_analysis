import os
from pathlib import Path

home_dir = str(Path.home())
tdc = os.path.join(home_dir, 'Desktop/python_library_dc/digital-connector-python')
digital_connector = os.path.join(home_dir, 'Desktop/UptodateProject/TomboloDigitalConnector')
os.chdir(tdc)

from recipe import Recipe, Subject, Dataset, Geo_Match_Rule, Match_Rule, Datasource, GeographicAggregationField, FixedValueField, AttributeMatcherField, AttributeMatcher, LatestValueField, MapToContainingSubjectField, BackOffField, PercentilesField, LinearCombinationField

subject_geometry = Subject(subject_type_label='localAuthority', provider_label='uk.gov.ons', 
                           match_rule=Match_Rule(attribute_to_match_on='label', pattern='E0900%'))

localAuthority = Datasource(importer_class='uk.org.tombolo.importer.ons.OaImporter',
                            datasource_id='localAuthority')

englandGeneralisedBoundaries = Datasource(importer_class='uk.org.tombolo.importer.ons.OaImporter' ,
                                          datasource_id='englandBoundaries')

NOMISIncome = Datasource(datasource_id='ONSGrossAnnualIncome',
                         importer_class='uk.org.tombolo.importer.ons.ONSEmploymentImporter')

ONSBusiness = Datasource(datasource_id='ONSBusiness',
                        importer_class='uk.org.tombolo.importer.ons.ONSBusinessDemographyImporter')

NOMISJobs = Datasource(datasource_id='ONSJobsDensity',
                      importer_class='uk.org.tombolo.importer.ons.ONSEmploymentImporter')

NOMISEmployment = Datasource(datasource_id='APSEmploymentRate',
                            importer_class='uk.org.tombolo.importer.ons.ONSEmploymentImporter')

NOMISUnEmployment = Datasource(datasource_id='APSUnemploymentRate',
                            importer_class='uk.org.tombolo.importer.ons.ONSEmploymentImporter')

NOMISBenefits = Datasource(datasource_id='ESAclaimants',
                          importer_class='uk.org.tombolo.importer.ons.ONSEmploymentImporter')

PopulationDensity = Datasource(datasource_id='qs102ew', 
                              importer_class='uk.org.tombolo.importer.ons.CensusImporter')


importers_list = [localAuthority,englandGeneralisedBoundaries, NOMISIncome, ONSBusiness, NOMISJobs,
                  NOMISEmployment, NOMISUnEmployment,NOMISBenefits, PopulationDensity]



### Fields ###

### Defining our attributes and passing them to fields ###

### Unemployment 

unemployment_attribute = AttributeMatcher(label='APSUnemploymentRate',
                                                   provider='uk.gov.ons')
unemployment = LatestValueField(attribute_matcher=unemployment_attribute,
                                                label='APSUnemploymentRate')

### Employment 

employment_attribute = AttributeMatcher(label='APSEmploymentRate',
                                                   provider='uk.gov.ons')
employment = LatestValueField(attribute_matcher=employment_attribute,
                                                label='APSEmploymentRate')

### Claiming allowance 

claimants_attribute = AttributeMatcher(label='ESAclaimants',
                                                   provider='uk.gov.ons')
claimants = LatestValueField(attribute_matcher=claimants_attribute,
                                                label='ESAclaimants')


### Tranforming them to percentiles after taking care of the missing values ###

fields = ['unemployment','employment', 'claimants']

f={}
for i in fields:
    f['geo_{0}'.format(i)] = GeographicAggregationField(subject=subject_geometry,
                                                           field=eval(('{0}').format(i)),
                                                           function='mean',
                                                           label='geo_{0}'.format(i))


    f['map_{0}'.format(i)] = MapToContainingSubjectField(field=f['geo_{0}'.format(i)],
                                                                   subject=Subject(subject_type_label='englandBoundaries',
                                                                                  provider_label='uk.gov.ons'),
                                                                   label='map_{0}'.format(i))

    f['backoff_{0}'.format(i)] = BackOffField(fields=[eval(('{0}').format(i)),
                                                             f['map_{0}'.format(i)]],
                                         label='backoff_{0}'.format(i))
    if i == 'employment':
        f['percentile_{0}'.format(i)] = PercentilesField(field=f['backoff_{0}'.format(i)],
                                                         inverse=False,
                                                         percentile_count=10,
                                                         normalization_subjects=[subject_geometry],
                                                         label='percentile_{0}'.format(i))
    else:
        f['percentile_{0}'.format(i)] = PercentilesField(field=f['backoff_{0}'.format(i)],
                                                         inverse=True,
                                                         percentile_count=10,
                                                         normalization_subjects=[subject_geometry],
                                                         label='percentile_{0}'.format(i))        


### Combining the resulting fields with a LinearCombinationField and convering the result to percentiles ###

combined_employment = LinearCombinationField(fields=[f['percentile_claimants'],
                                                 f['percentile_employment'],
                                                 f['percentile_unemployment']],
                                         scalars = [1.,1.,1.],
                                         label='Unemployment lower than the East London average')

percentile_combined_employment = PercentilesField(field=combined_employment,
                                                     inverse=False,
                                                     label='unemployment',
                                                     percentile_count=10,
                                                     normalization_subjects=[subject_geometry])

### Run the exporter and plot the result ###

importers = [localAuthority,englandGeneralisedBoundaries,
            NOMISEmployment,NOMISUnEmployment,NOMISBenefits]

dataset = Dataset(subjects=[subject_geometry], fields=[f['percentile_claimants'], 
                                                       f['percentile_employment'], f['percentile_unemployment']],
                  datasources=importers)

recipe = Recipe(dataset,timestamp=False)
recipe.build_recipe(console_print=False)

recipe.run_recipe(tombolo_path=digital_connector,
                  output_path = 'Desktop/employment_and_skills.json', console_print=False)

import geopandas as gpd

data = gpd.read_file(home_dir + '/Desktop/employment_and_skills.json')
data.head()



