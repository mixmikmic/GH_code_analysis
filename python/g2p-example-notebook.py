from ga4gh.client import client
ga4gh_endpoint = "http://1kgenomes.ga4gh.org"
c = client.HttpClient(ga4gh_endpoint)

datasets = c.search_datasets()
phenotype_association_set_id = None
phenotype_association_set_name = None
for  dataset in datasets:
  phenotype_association_sets = c.search_phenotype_association_sets(dataset_id=dataset.id)
  for phenotype_association_set in phenotype_association_sets:
    phenotype_association_set_id = phenotype_association_set.id
    phenotype_association_set_name = phenotype_association_set.name
    print 'Found G2P phenotype_association_set:', phenotype_association_set.id, phenotype_association_set.name
    break

assert phenotype_association_set_id
assert phenotype_association_set_name

feature_set_id = None
datasets = c.search_datasets()
for  dataset in datasets:
  featuresets = c.search_feature_sets(dataset_id=dataset.id)
  for featureset in featuresets:
    if phenotype_association_set_name in featureset.name:
      feature_set_id = featureset.id
      print 'Found G2P feature_set:', feature_set_id
      break        
assert feature_set_id

feature_generator = c.search_features(feature_set_id=feature_set_id,
                        reference_name="chr7",
                        start=55249005,
                        end=55249006
                    )

features = list(feature_generator)
assert len(features) == 1
print "Found {} features in G2P feature_set {}".format(len(features),feature_set_id)
feature = features[0]
print [feature.name,feature.gene_symbol,feature.reference_name,feature.start,feature.end]


feature_generator = c.search_features(feature_set_id=feature_set_id, name='EGFR S768I missense mutation')
features = list(feature_generator)
assert len(features) == 1
print "Found {} features in G2P feature_set {}".format(len(features),feature_set_id)
feature = features[0]
print [feature.name,feature.gene_symbol,feature.reference_name,feature.start,feature.end]


feature_phenotype_associations =  c.search_genotype_phenotype(
                                    phenotype_association_set_id=phenotype_association_set_id,
                                    feature_ids=[f.id  for f in features])
associations = list(feature_phenotype_associations)
assert len(associations) >= len(features)
print "There are {} associations".format(len(associations))
print "\n".join([a.description for a in associations])


from IPython.display import IFrame
IFrame(associations[0].evidence[0].info['publications'][0], "100%",300)

phenotypes_generator = c.search_phenotype(
                phenotype_association_set_id=phenotype_association_set_id,
                description="Adenosquamous carcinoma .*"
                )
phenotypes = list(phenotypes_generator)

assert len(phenotypes) >= 0
print "\n".join(set([p.description for p in phenotypes])) 

feature_phenotype_associations =  c.search_genotype_phenotype(
                                    phenotype_association_set_id=phenotype_association_set_id,
                                    phenotype_ids=[p.id for p in phenotypes])
associations = list(feature_phenotype_associations)
assert len(associations) >= len(phenotypes)
print "There are {} associations. First five...".format(len(associations))
print "\n".join([a.description for a in associations][:5])

import ga4gh_client.protocol as protocol
evidence = protocol.EvidenceQuery()
evidence.description = "MEK inhibitors"
    
feature_phenotype_associations =  c.search_genotype_phenotype(
                                    phenotype_association_set_id=phenotype_association_set_id,
                                    phenotype_ids=[p.id for p in phenotypes],
                                    evidence = [evidence]
                                    )
associations = list(feature_phenotype_associations)
print "There are {} associations. First five...".format(len(associations))
print "\n".join([a.description for a in associations][:5])

feature_generator = c.search_features(feature_set_id=feature_set_id, name='.*KIT.*')
features = list(feature_generator)
assert len(features) > 0
print "Found {} features. First five...".format(len(features),feature_set_id)
print "\n".join([a.description for a in associations][:5])

feature_phenotype_associations =  c.search_genotype_phenotype(
                                    phenotype_association_set_id=phenotype_association_set_id,
                                    feature_ids=[f.id  for f in features])
associations = list(feature_phenotype_associations)
print "There are {} associations.  First five...".format(len(associations))
print "\n".join([a.description for a in associations][:5])

from bokeh.charts import HeatMap, output_notebook, output_file, show 

from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable,   TableColumn
from bokeh.models import HoverTool


feature_ids = {}
for feature in features:
    feature_ids[feature.id]=feature.name

phenotype_descriptions = []
feature_names = []
association_count = [] 
association_descriptions = []

for association in associations:
    for feature_id in association.feature_ids:
        phenotype_descriptions.append(association.phenotype.description)
        feature_names.append(feature_ids[feature_id])
        association_count.append(1)
        association_descriptions.append(association.description)

output_notebook()
output_file("g2p_heatmap.html")
  
data = {'feature': feature_names  ,
        'association_count': association_count,
        'phenotype': phenotype_descriptions,
        'association_descriptions': association_descriptions
        }

hover = HoverTool(
        tooltips=[
            ("associations", "@values")
        ]
    )

hm = HeatMap(data, x='feature', y='phenotype', values='association_count',
             title='G2P Associations for KIT', stat='sum',
             legend=False,width=1024,
             tools=[hover], #"hover,pan,wheel_zoom,box_zoom,reset,tap",
             toolbar_location="above")

source = ColumnDataSource(data)
columns = [
        TableColumn(field="association_descriptions", title="Description"),
    ]
data_table = DataTable(source=source, columns=columns,width=1024 )

show( column(hm,data_table)  )

