from elasticsearch import Elasticsearch
from pprint import pprint

es = Elasticsearch(
    'localhost',
    # sniff before doing anything
    sniff_on_start=True,
    # refresh nodes after a node fails to respond
    sniff_on_connection_fail=True,
    # and also every 60 seconds
    sniffer_timeout=60
)

f = open('examples.json', 'r')
data = f.read()

response = es.bulk(body=data)

assert response['errors'] == False
# Should not produce an AssertionError

es.indices.delete(index=['gb','us'])

index_template = {
  "mappings": {
    "tweet" : {
      "properties" : {
        "tweet" : {
          "type" :    "text",
          "analyzer": "english"
        },
        "date" : {
          "type" :   "date"
        },
        "name" : {
          "type" :   "text"
        },
        "user_id" : {
          "type" :   "long"
        }
      }
    }
  }
}

es.indices.create(index='gb', body=index_template)

es.indices.delete(index='email') # an index we create later on

multi_field_index_template = {
  "mappings": {
    "tweet" : {
      "properties" : {
                
        "tweet": { 
            "type":     "string",
            "analyzer": "english",
            "fields": {
                "raw": { 
                    "type":  "string",
                    "index": "not_analyzed"
                        }
                      }
        },    
        "date" : {
          "type" :   "date"
        },
        "name" : {
          "type" :   "text"
        },
        "user_id" : {
          "type" :   "long"
        }
      }
    }
  }
}
es.indices.create(index='gb', body=multi_field_index_template)


es.indices.exists('gb')

if es.indices.exists(['gb,us']):
    es.indices.delete(index=['gb,us'])

es.indices.exists('gb')

es.indices.create(index='gb', body=multi_field_index_template)



