"""
See Azure Search REST API docs for more info:
    https://docs.microsoft.com/en-us/rest/api/searchservice/index
"""
import requests
import logging
import json
import pandas as pd

# This is the service and index already created in Azure Portal
serviceName = "eyazuresearch2017"
indexName = "mh-eytaxidxer-synonymc512"
apiVersion = "2016-09-01-Preview"

# API key of the service subscription
apiKey = "XXXXXXXXXXXXXXXXXXXXXX"

INPUT_FILE = "keywords_synonym.txt"
file = open(INPUT_FILE,'r')
all_text = file.read()
file.close()

body= "{\"name\":\"synonym-map\",\"format\":\"solr\",\"synonyms\": \"%s\"}" %(all_text)

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

def getServiceUrl():
    return "https://" + serviceName + ".search.windows.net"

def postMethod(body):
    headers = {"Content-type": "application/json", "api-key": apiKey}
    servicePath ="/synonymmaps?api-version=%s" %(apiVersion)
    r = requests.post(getServiceUrl() + servicePath, headers=headers, data=body)
    print(r, r.text)
    return r

r = postMethod(body)

put_body =json.dumps({
        "name": indexName,  
        "fields": [
        {"name": "Index", "type": "Edm.String", "key": True, "retrievable": True, "searchable": False, "filterable": False, "sortable": True, "facetable": False},

        {"name": "File", "type": "Edm.String", "retrievable": True, "searchable": False, "filterable": True, "sortable": True, "facetable": False},

        {"name": "Chapter", "type": "Edm.String", "retrievable": True, "searchable": False, "filterable": True, "sortable": True, "facetable": False},

        {"name": "Title", "type": "Edm.String", "retrievable": True, "searchable": True, "filterable": True, "sortable": True, "facetable": True},

        {"name": "SectionTitle", "type": "Edm.String", "retrievable": True, "searchable": True, "filterable": True, "sortable": False, "facetable": True},

        {"name": "SubsectionTitle", "type": "Edm.String", "retrievable": True, "searchable": True, "filterable": True, "sortable": True, "facetable": False},

        {"name": "Source", "type": "Edm.String", "retrievable": True, "searchable": False, "filterable": False, "sortable": True, "facetable": True},

        {"name": "FeatureType", "type": "Edm.String", "retrievable": True, "searchable": False, "filterable": True, "sortable": True, "facetable": True},

        {"name": "ParaText", "type": "Edm.String", "retrievable": True, "searchable": True, "filterable": False, "sortable": False, "facetable": False, "analyzer": "en.microsoft"},

        {"name": "Keywords", "type": "Edm.String", "retrievable": True, "searchable": True, "filterable": False, "sortable": False, "facetable": False, "analyzer": "en.microsoft","synonymMaps":["synonym-map"]}
        ]
    })

def putMethod(body):
    headers = {"Content-type": "application/json", "api-key": apiKey}
    servicePath ="/indexes/%s?api-version=%s" %(indexName,apiVersion)
    r = requests.put(getServiceUrl() + servicePath, headers=headers, data=put_body)
    print(r, r.text)
    return r

r = putMethod(body)

def getSynonyms():
    #servicePath = '/indexers/%s?api-version=%s' % (indexName, apiVersion)
    headers = {"Content-type": "application/json", "api-key": apiKey}
    r = requests.get("https://eyazuresearch2017.search.windows.net/synonymmaps/synonym-map?api-version=2016-09-01-Preview", headers=headers)
    print(r, r.text)
    return r

getSynonyms()



