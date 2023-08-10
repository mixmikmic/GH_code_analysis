from ga4gh.client import client
c = client.HttpClient("http://1kgenomes.ga4gh.org")

for reference_set in c.search_reference_sets():
    ncbi37 = reference_set
    print "name: {}".format(ncbi37.name)
    print "ncbi_taxon_id: {}".format(ncbi37.ncbi_taxon_id)
    print "description: {}".format(ncbi37.description)
    print "source_uri: {}".format(ncbi37.source_uri)

reference_set = c.get_reference_set(reference_set_id=ncbi37.id)
print reference_set

counter = 0
for reference in c.search_references(reference_set_id=ncbi37.id):
    if reference.name == "1":
        base_id_ref = reference
    counter += 1
    if counter > 5:
        break
    print reference

reference = c.get_reference(reference_id=base_id_ref.id)
print reference

reference_bases = c.list_reference_bases(base_id_ref.id, start=15000, end= 16000)
print reference_bases
print len(reference_bases)

