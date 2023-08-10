import hca, hca.dss, json
client = hca.dss.DSSClient()

help(client.get_file)

client.post_search(replica="aws", es_query={
    "query": {
        "bool": {
            "must": [{
                "match": {
                    "files.sample_json.donor.species": "Homo sapiens"
                }
            }, {
                "match": {
                    "files.assay_json.single_cell.method": "Fluidigm C1"
                }
            }, {
                "match": {
                    "files.sample_json.ncbi_biosample": "SAMN04303778"
                }
            }]
        }
    }
})

search_response = client.post_search(replica="aws", es_query={})
search_response["total_hits"]

search_response["results"][0]

client.get_bundle(uuid=search_response["results"][0]["bundle_fqid"], replica="aws")

client.get_bundle(uuid=search_response["results"][0]["bundle_fqid"][:36], replica="aws")

for result in search_response["results"]:
    bundle_uuid = result["bundle_fqid"][:36]
    bundle_dict = client.get_bundle(uuid=bundle_uuid, replica="aws")
    found_file = False
    for file_dict in bundle_dict["bundle"]["files"]:
        if file_dict["name"].endswith(".bam"):
            print("Name: {}, UUID: {}".format(file_dict["name"], file_dict["uuid"]))
            found_file = True
            break
    if found_file:
        break

for result in search_response["results"]:
    bundle_uuid = result["bundle_fqid"][:36]
    bundle_dict = client.get_bundle(uuid=bundle_uuid, replica="aws")
    found_file = False
    for file_dict in bundle_dict["bundle"]["files"]:
        if file_dict["name"].endswith(".fastq.gz"):
            print("Name: {}, UUID: {}".format(file_dict["name"], file_dict["uuid"]))
            found_file = True
            break
    if found_file:
        break

help(client.post_search)

results = client.post_search.iterate(replica="aws", es_query={})
for result in results:
    bundle_uuid = result["bundle_fqid"][:36]
    bundle_dict = client.get_bundle(uuid=bundle_uuid, replica="aws")
    found_file = False
    for file_dict in bundle_dict["bundle"]["files"]:
        if file_dict["name"].endswith(".bam"):
            print("Name: {}, UUID: {}".format(file_dict["name"], file_dict["uuid"]))
            found_file = True
            break
    if found_file:
        break

bam_file = client.get_file(uuid="f9fa3804-ce46-456a-9023-50c8d5bf822f", replica="aws")
with open("Aligned.sortedByCoord.out.bam", "wb") as output_bam:
    output_bam.write(bam_file)

import pysam
bam = pysam.AlignmentFile("Aligned.sortedByCoord.out.bam", "rb")
print(bam.header)

