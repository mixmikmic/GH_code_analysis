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

#r = index.load_sid_examples(settings={ "settings": { "number_of_shards": 1 }},set=3)
#print('{} items created'.format(len(r['items'])))

# Let's repopulate the index as we deleted 'gb' in earlier chapters:
# Run the script: populate.ipynb

# Let's confirm how the most_fields query works by validating the query
body= {
    "settings": { "number_of_shards": 1 },
    "mappings": {
        "address": {
            "properties": {
                "postcode": {
                    "type":  "string",
                    "index": "not_analyzed"
                }
            }
        }
    }
}
index.create_my_index(body=body)

# Adding the **and** operator
zips = [ "W1V 3DG", "W2F 8HW", "W1F 7HW", "WC1N 1LZ", "SW5 0BE" ]
for i,postcode in enumerate(zips):
    body = {}
    body['postcode'] = zips[i]
    print(body)
    r = es.create(index='my_index', doc_type='address', id=i, body=body)

s = Index('my_index', using=es).search()
s = s.query(Q('prefix', postcode="W1"))
s.execute()

# wildcards
s = Index('my_index', using=es).search()
s = s.query(Q('wildcard', postcode="W?F*HW"))
s.execute()

# regex
s = Index('my_index', using=es).search()
s = s.query(Q('regexp', postcode="W[0-9].+"))
s.execute()

s = Index('my_index', using=es).search()
s = s.query(Q('match_phrase_prefix', title="quick brown f"))
s.execute()

# Adding some slop
s = Index('my_index', using=es).search()
s = s.query(Q('match_phrase_prefix', title={"query": "brown quick f", "slop":2}))
s.execute()

# Adding some control on expansions:
s = Index('my_index', using=es).search()
s = s.query(Q('match_phrase_prefix', title={"query": "quick brown f", "max_expansions":2}))
s.execute()

body = {
    "settings": {
        "number_of_shards": 1, 
        "analysis": {
            "filter": {
                "autocomplete_filter": { 
                    "type":     "edge_ngram",
                    "min_gram": 1,
                    "max_gram": 20
                }
            },
            "analyzer": {
                "autocomplete": {
                    "type":      "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "autocomplete_filter" 
                    ]
                }
            }
        }
    }
}
index.create_my_index(body=body)

# Now let's confirm how this analyzer works:
text = "quick brown" 
analyzed_text = [[x['position'],x['token']] for x in es.indices.analyze                 (index='my_index', analyzer='autocomplete', text=text)['tokens']]
for item in analyzed_text:
    print('Pos {}: ({})'.format(item[0],item[1]))

# Update the mapping to try it out:
mapping = {
    "my_type": {
        "properties": {
            "name": {
                "type":     "string",
                "analyzer": "autocomplete"
            }
        }
    }
}
es.indices.put_mapping(index='my_index', doc_type='my_type', body=mapping)

doc = { "name": "Brown foxes"    }
es.create(index='my_index', doc_type='my_type', body = doc, id=1)
doc = { "name": "Yellow furballs"    }
es.create(index='my_index', doc_type='my_type', body = doc, id=2)

#Now try a search:
s = Index('my_index', using=es).search()
s = s.query('match', name="brown fo")
s.execute()

q = Q('match', name="brown fo").to_dict()
es.indices.validate_query(index='my_index', body={"query": q}, explain=1)

# Let's use the standard analyzer at search time
s = Index('my_index', using=es).search()
s = s.query('match', name={"query":"brown fo", "analyzer":"standard"})
s.execute()

# Update the mapping to specify different index vs. search analyzers:
mapping = {
    "my_type": {
        "properties": {
            "name": {
                "type":     "string",
                "analyzer": "autocomplete",
                "search_analyzer": "standard"
            }
        }
    }
}
es.indices.put_mapping(index='my_index', doc_type='my_type', body=mapping)

#Now try a search with specifying an analyzer:
s = Index('my_index', using=es).search()
s = s.query('match', name="brown fo")
s.execute()

# And let's validate again:
q = Q('match', name="brown fo").to_dict()
es.indices.validate_query(index='my_index', body={"query": q}, explain=1)



