from SPARQLWrapper import SPARQLWrapper, JSON
import json

# Use the public endpoint

sparql_endpoint = "https://opensparql.sbgenomics.com/blazegraph/namespace/tcga_metadata_kb/sparql"

# Initialize the SPARQL wrapper with the endpoint
sparql = SPARQLWrapper(sparql_endpoint)

query = """
prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
prefix tcga: <https://www.sbgenomics.com/ontologies/2014/11/tcga#>

select distinct ?case ?sample ?file_name ?path ?xs_label ?subtype_label
where
{
 ?case a tcga:Case .
 ?case tcga:hasDiseaseType ?disease_type .
 ?disease_type rdfs:label "Lung Adenocarcinoma" .
 
 ?case tcga:hasHistologicalDiagnosis ?hd .
 ?hd rdfs:label "Lung Adenocarcinoma Mixed Subtype" .
 

 

 
 ?case tcga:hasFollowUp ?follow_up .
 ?follow_up tcga:hasDaysToLastFollowUp ?days_to_last_follow_up .
 filter(?days_to_last_follow_up>550) 
  
 ?follow_up tcga:hasVitalStatus ?vital_status .
 ?vital_status rdfs:label ?vital_status_label .
 filter(?vital_status_label="Alive")
 
 ?case tcga:hasDrugTherapy ?drug_therapy .
 ?drug_therapy tcga:hasPharmaceuticalTherapyType ?pt_type .
 ?pt_type rdfs:label ?pt_type_label .
 filter(?pt_type_label="Chemotherapy")
  
 ?case tcga:hasSample ?sample .
 ?sample tcga:hasSampleType ?st .
 ?st rdfs:label ?st_label
 filter(?st_label="Primary Tumor")
     
 ?sample tcga:hasFile ?file .
 ?file rdfs:label ?file_name .
 
 ?file tcga:hasStoragePath ?path.
  
 ?file tcga:hasExperimentalStrategy ?xs.
 ?xs rdfs:label ?xs_label .
 filter(?xs_label="WXS")
  
 ?file tcga:hasDataSubtype ?subtype .
 ?subtype rdfs:label ?subtype_label

}





"""


sparql.setQuery(query)

sparql.setReturnFormat(JSON)
results = sparql.query().convert()



# From results, we grab a list of files. TCGA metadata database returns a list of filepaths. 
filelist = [result['path']['value'] for result in results['results']['bindings']]




# The list of file paths is now in the filelist array, as shown below
print 'Your query returned %s files with paths:' % len(filelist)

for file in filelist:
    print file 

# The following script uses the Python requests library to make a small wrapper around the CGC API
import uuid
import json
import pprint
import requests

def api(api_url, path, auth_token,method='GET', query=None, data=None): 
  data = json.dumps(data) if isinstance(data, dict) or isinstance(data,list) else None 
  base_url = api_url
 
  headers = { 
    'X-SBG-Auth-Token': auth_token, 
    'Accept': 'application/json', 
    'Content-type': 'application/json', 
  } 
 
  response = requests.request(method, base_url + path, params=query, data=data, headers=headers) 
  print "URL: ",  response.url
  print "RESPONSE CODE: ", response.status_code
  print ('--------------------------------------------------------------------------------------------------------------------')
  response_dict = json.loads(response.content) if response.content else {} 
  response_headers = dict(response.headers)

  pprint.pprint(response_headers)
  print('--------------------------------------------------------------------------------------------------------------------')
  pprint.pprint(response_dict)
  return response_dict

# API base URL
base = 'https://cgc-api.sbgenomics.com/v2/' 

auth_token = '<YOUR TOKEN HERE>'


# Get download data for each of the files 

# Note that here we use a special purpose API call on CGC as described on 
# http://docs.cancergenomicscloud.org/v1.0/docs/get-a-files-download-url

download_urls = api(api_url=base,auth_token=auth_token,path='action/files/get_download_url',method='POST',query=None,data=filelist)


outfile = open('download.txt','wb')
for url in download_urls:
    outfile.write(url)
    outfile.write('\n')

outfile.close()

