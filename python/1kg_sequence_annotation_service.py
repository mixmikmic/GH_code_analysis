from ga4gh.client import client
c = client.HttpClient("http://1kgenomes.ga4gh.org")

#Obtain dataSet id REF: -> `1kg_metadata_service`
dataset = c.search_datasets().next() 

for feature_set in c.search_feature_sets(dataset_id=dataset.id):
    print feature_set
    if feature_set.name == "gencode_v24lift37":
        gencode = feature_set

feature_set = c.get_feature_set(feature_set_id=gencode.id)
print feature_set

counter = 0
for features in c.search_features(feature_set_id=feature_set.id):
    if counter > 3:
        break
    counter += 1
    print"Id: {},".format(features.id)
    print" Name: {},".format(features.name)
    print" Gene Symbol: {},".format(features.gene_symbol)
    print" Parent Id: {},".format(features.parent_id)
    if features.child_ids:
        for i in features.child_ids:
            print" Child Ids: {}".format(i)
    print" Feature Set Id: {},".format(features.feature_set_id)
    print" Reference Name: {},".format(features.reference_name)
    print" Start: {},\tEnd: {},".format(features.start, features.end)
    print" Strand: {},".format(features.strand)
    print"  Feature Type Id: {},".format(features.feature_type.id)
    print"  Feature Type Term: {},".format(features.feature_type.term)
    print"  Feature Type Sorce Name: {},".format(features.feature_type.source_name)
    print"  Feature Type Source Version: {}\n".format(features.feature_type.source_version)

for feature in c.search_features(feature_set_id=feature_set.id, reference_name="chr17", start=42000000, end=42001000):
    print feature.name, feature.start, feature.end

feature = c.get_feature(feature_id=features.id)
print"Id: {},".format(feature.id)
print" Name: {},".format(feature.name)
print" Gene Symbol: {},".format(feature.gene_symbol)
print" Parent Id: {},".format(feature.parent_id)
if feature.child_ids:
    for i in feature.child_ids:
        print" Child Ids: {}".format(i)
print" Feature Set Id: {},".format(feature.feature_set_id)
print" Reference Name: {},".format(feature.reference_name)
print" Start: {},\tEnd: {},".format(feature.start, feature.end)
print" Strand: {},".format(feature.strand)
print"  Feature Type Id: {},".format(feature.feature_type.id)
print"  Feature Type Term: {},".format(feature.feature_type.term)
print"  Feature Type Sorce Name: {},".format(feature.feature_type.source_name)
print"  Feature Type Source Version: {}\n".format(feature.feature_type.source_version)
for vals in feature.attributes.vals:
    print"{}: {}".format(vals, feature.attributes.vals[vals].values[0].string_value)

