from ga4gh.client import client
c = client.HttpClient("http://1kgenomes.ga4gh.org")

#Obtain dataSet id REF: -> `1kg_metadata_service`
dataset = c.search_datasets().next() 

#Obtain functional-annotations set id REF: -> `1kg_variant_service`
for variant_sets in c.search_variant_sets(dataset_id=dataset.id):
    if variant_sets.name == "functional-annotations":
        variant_sets = variant_sets.id
        break

for variant_annotation_sets in c.search_variant_annotation_sets(variant_set_id=variant_sets.id):
    print "\nName: {},".format(variant_annotation_sets.name)
    print" Id: {},".format(variant_annotation_sets.id)
    print" Variant Set Id: {},".format(variant_annotation_sets.variant_set_id)

counter = 3
for variant_annotations in c.search_variant_annotations(variant_annotation_set_id=variant_annotation_sets.id, reference_name="1", start=0, end=1000000):
    if counter <= 0:
        break
    counter -= 1 
    print"Id: {},".format(variant_annotations.id)
    print" Variant Id: {},".format(variant_annotations.variant_id)
    print" Variant Annotation Set Id: {}".format(variant_annotations.variant_annotation_set_id)
    print" Created: {}".format(variant_annotations.created)
    print" Transcript Effects Id: {},".format(variant_annotations.transcript_effects[0].id)
    print" Featured Id: {},".format(variant_annotations.transcript_effects[0].feature_id)
    print" Alternate Bases: {},".format(variant_annotations.transcript_effects[0].alternate_bases)
    print" Effects Id: {},".format(variant_annotations.transcript_effects[0].effects[0].id)
    print" Effect Term: {},".format(variant_annotations.transcript_effects[0].effects[0].term)
    print" Effect Sorce Name: {},".format(variant_annotations.transcript_effects[0].effects[0].source_name)
    print" Effect Source Version: {}\n".format(variant_annotations.transcript_effects[0].effects[0].source_version)

variant_annotation_set = c.get_variant_annotation_set(variant_annotation_set_id=variant_annotation_sets.id)

print"Name: {}".format(variant_annotation_set.name)
print" Id: {} ".format(variant_annotation_set.id)
print" Variant Set Id: {}".format(variant_annotation_set.variant_set_id)

