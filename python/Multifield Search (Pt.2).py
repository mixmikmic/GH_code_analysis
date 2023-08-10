import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import index
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q, Index
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

r = index.load_sid_examples(settings={ "settings": { "number_of_shards": 1 }},set=3)
print('{} items created'.format(len(r['items'])))

# Let's repopulate the index as we deleted 'gb' in earlier chapters:
# Run the script: populate.ipynb

# Let's confirm how the most_fields query works by validating the query
body= {
  "query": {
    "multi_match": {
      "query":   "Poland Street W1V",
      "type":    "most_fields",
      "fields":  [ "street", "city", "country", "postcode" ]
    }
  }
}
es.indices.validate_query(index='my_index', body=body, explain=1)    ['explanations'][0]['explanation']

# Adding the **and** operator
body= {
  "query": {
    "multi_match": {
      "query":   "Poland Street W1V",
      "type":    "most_fields",
      "operator": "and",
      "fields":  [ "street", "city", "country", "postcode" ]
    }
  }
}
es.indices.validate_query(index='my_index', body=body, explain=1)    ['explanations'][0]['explanation']



