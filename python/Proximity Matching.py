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
#print('{} items created'.format(len(r['items'])))

# Let's repopulate the index as we deleted 'gb' in earlier chapters:
# Run the script: populate.ipynb

s = Index('my_index', using=es).search()
s = s.query(Q('match_phrase', title="quick brown fox"))
s.execute()

s = Index('my_index', using=es).search()
s = s.query(Q('match', title={"query": "quick brown fox", "type":"phrase"}))
s.execute()

s = Index('my_index', using=es).search()
s = s.query(Q('prefix', postcode="W1"))
s.execute()

es.indices.analyze(index='my_index', analyzer='standard', text='Quick brown fox')

# "sloppy"
s = Index('my_index', using=es).search()
s = s.query(Q('match_phrase', title={"query": "quick  fox", "slop":1}))
s.execute()



