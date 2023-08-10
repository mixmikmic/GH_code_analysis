from ga4gh.client import client
c = client.HttpClient("http://1kgenomes.ga4gh.org")

#Obtain dataSet id REF: -> `1kg_metadata_service`
dataset = c.search_datasets().next() 

#Obtain reference set id REF:-> `1kg_reference_service`
reference_set = c.search_reference_sets().next()
reference = c.search_references(reference_set_id=reference_set.id).next()

counter = 0
for read_group_set in c.search_read_group_sets(dataset_id=dataset.id):
    counter += 1
    if counter < 4:
        print "Read Group Set: {}".format(read_group_set.name)
        print "id: {}".format(read_group_set.id)
        print "dataset_id: {}".format(read_group_set.dataset_id)
        print "Aligned Read Count: {}".format(read_group_set.stats.aligned_read_count)
        print "Unaligned Read Count: {}\n".format(read_group_set.stats.unaligned_read_count)
        if read_group_set.name == "NA19675":
            rgSet = read_group_set
        for read_group in read_group_set.read_groups:
            print "  Read group:"
            print "  id: {}".format(read_group.id)
            print "  Name: {}".format(read_group.name)
            print "  Description: {}".format(read_group.description)
            print "  Biosample Id: {}\n".format(read_group.bio_sample_id)
    else: 
        break

read_group_set = c.get_read_group_set(read_group_set_id=rgSet.id)
print "Read Group Set: {}".format(read_group_set.name)
print "id: {}".format(read_group_set.id)
print "dataset_id: {}".format(read_group_set.dataset_id)
print "Aligned Read Count: {}".format(read_group_set.stats.aligned_read_count)
print "Unaligned Read Count: {}\n".format(read_group_set.stats.unaligned_read_count)
for read_group in read_group_set.read_groups:
    print " Read Group: {}".format(read_group.name)
    print " id: {}".format(read_group.bio_sample_id)
    print " bio_sample_id: {}\n".format(read_group.bio_sample_id)

for read_group in read_group_set.read_groups:
    print "Alignment from {}\n".format(read_group.name)
    alignment = c.search_reads(read_group_ids=[read_group.id], start=0, end=1000000, reference_id=reference.id).next()
    print " id: {}".format(alignment.id)
    print " fragment_name: {}".format(alignment.fragment_name)
    print " aligned_sequence: {}\n".format(alignment.aligned_sequence)

