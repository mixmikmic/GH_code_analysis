from ga4gh.client import client
c = client.HttpClient("http://1kgenomes.ga4gh.org")

#Obtain dataSet id REF: -> `1kg_metadata_service`
dataset = c.search_datasets().next() 

for variant_set in c.search_variant_sets(dataset_id=dataset.id):
    if variant_set.name == "phase3-release":
        var_set_id = variant_set
    print "Variant Set: {}".format(variant_set.name)
    print " id: {}".format(variant_set.id)
    print " dataset_id: {}".format(variant_set.dataset_id)
    print " reference_set_id: {}\n".format(variant_set.reference_set_id)

variant_set = c.get_variant_set(variant_set_id=var_set_id.id)
print "name: {}".format(variant_set.name)
print "dataset_id: {}".format(variant_set.dataset_id)
print "reference_set_id: {}".format(variant_set.reference_set_id)
for metadata in variant_set.metadata[0:3]:
    print metadata

counter = 0
for variant in c.search_variants(variant_set_id=var_set_id.id, reference_name="1", start=10176, end= 40176):
    if counter > 5:
        break
    counter += 1
    print "Variant id: {}...".format(variant.id[0:10])
    print "Variant Set Id: {}".format(variant.variant_set_id)
    print "Names: {}".format(variant.names)
    print "Reference Chromosome: {}".format(variant.reference_name)
    print "Start: {}, End: {}".format(variant.start, variant.end)
    print "Reference Bases: {}".format(variant.reference_bases)
    print "Alternate Bases: {}\n".format(variant.alternate_bases)

single_variant = c.get_variant(variant_id=variant.id)
print "idd: {}".format(single_variant.id)
print "Variant Set Id: {}".format(single_variant.variant_set_id)
print "Names: {}".format(single_variant.names)
print "Reference Name: {}".format(single_variant.reference_name)
print "Start: {}, End: {}".format(single_variant.start, single_variant.end)
print "Reference Bases: {}".format(single_variant.reference_bases)
print "Alternate Bases: {}\n".format(single_variant.alternate_bases)
for info in single_variant.info:
    print "Key: {},\tValues: {}".format(info, single_variant.info[info].values[0].string_value)

metadata_dictionary = {}
for metadata in variant_set.metadata:
    metadata_dictionary[metadata.key] = metadata # Load the metadata elements into a dictionary
for key in single_variant.info:
    metadata_entry = metadata_dictionary["INFO." + key]
    print "\nKey: {}     Value: {}     Type: {}".format(
        key,
        single_variant.info[key].values[0].string_value,
        metadata_entry.type)
    print " " + metadata_entry.description

counter = 0
list_of_callset_ids = [] # Will use this list near the end to make a search variants query
for call_set in c.search_call_sets(variant_set_id=single_variant.variant_set_id):
    if counter > 3:
        break
    else:
        counter += 1
        list_of_callset_ids.append(call_set.id)
        print "Call Set Name: {}".format(call_set.name)
        print "  id: {}".format(call_set.name)
        print "  bio_sample_id: {}".format(call_set.name)
        print "  variant_set_ids: {}\n".format(call_set.variant_set_ids)

call_set = c.get_call_set(call_set_id=call_set.id)
print call_set

for variant_with_calls in  c.search_variants(call_set_ids=list_of_callset_ids, variant_set_id=call_set.variant_set_ids[0], reference_name="1", start=10176, end= 10502):
    print "Variant Id: {}".format(variant_with_calls.id)
    print "Variant Set Id: {}".format(variant_with_calls.variant_set_id)
    print "Names:{}".format(variant_with_calls.names)
    print "Reference Chromosome: {}".format(variant_with_calls.reference_name)
    print "Start: {}, End: {}".format(variant_with_calls.start, variant_with_calls.end)
    print "Reference Bases: {}".format(variant_with_calls.reference_bases)
    print "Alternate Bases: {}\n".format(variant_with_calls.alternate_bases)
    for call in variant_with_calls.calls:
        print "  Call Set Name: {}".format(call.call_set_name)
        print "  Genotype: {}".format(call.genotype)
        print "  Phase set: {}".format(call.phaseset)
        print "  Call Set Id: {}\n".format(call.call_set_id)



