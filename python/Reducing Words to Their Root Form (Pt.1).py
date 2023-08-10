import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import index
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
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

r = index.populate()
print('{} items created'.format(len(r['items'])))

# Let's repopulate the index as we deleted 'gb' in earlier chapters:
# Run the script: populate.ipynb

english_token_filter = {
  "settings": {
    "analysis": {
      "filter": {
        "english_stop": {
          "type":       "stop",
          "stopwords":  "_english_"
        },
        "light_english_stemmer": {
          "type":       "stemmer",
          "language":   "light_english" 
        },
        "english_possessive_stemmer": {
          "type":       "stemmer",
          "language":   "possessive_english"
        }
      },
      "analyzer": {
        "my_english": {
          "tokenizer":  "standard",
          "filter": [
            "english_possessive_stemmer",
            "lowercase",
            "english_stop",
            "light_english_stemmer", 
            "asciifolding" 
          ]
        }
      }
    }
  }
}

index.create_my_index(body=english_token_filter)

text = "You're right about jumping jack's Über generation of waiters."
doc = {
    "message": text
}
es.create(index="my_index", doc_type='test', body=doc, id=1)

# test with the standard English analyzer
analyzed_text = [x['token'] for x in es.indices.analyze                 (index='my_index', analyzer='english', text=text)['tokens']]
print(','.join(analyzed_text))

# test with the modified English analyzer - 'my_english'
analyzed_text = [x['token'] for x in es.indices.analyze                 (index='my_index', analyzer='my_english', text=text)['tokens']]
print(','.join(analyzed_text))

s = Search(using=es, index="my_index").query('match', message="jump uber")
s.execute()

res = es.indices.get_mapping(index='my_index', doc_type='test')
res
#es.indices.get_field_mapping(index='my_index', fields='messages')

english_token_filter = {
  "settings": {
    "analysis": {
      "filter": {
        "english_stop": {
          "type":       "stop",
          "stopwords":  "_english_"
        },
        "light_english_stemmer": {
          "type":       "stemmer",
          "language":   "light_english" 
        },
        "english_possessive_stemmer": {
          "type":       "stemmer",
          "language":   "possessive_english"
        }
      },
      "analyzer": {
        "my_english": {
          "tokenizer":  "standard",
          "filter": [
            "english_possessive_stemmer",
            "lowercase",
            "english_stop",
            "light_english_stemmer", 
            "asciifolding" 
          ]
        }
      }
    }
  },
    "mappings": {
    "test" : {
      "properties" : {
        "message" : {
          "type" :    "text",
          "analyzer": "my_english"
        }
      }
    }
  }
}
index.create_my_index(body=english_token_filter)

text = "You're right about those jumping jacks in the Über generation of waiters."
doc = {
    "message": text
}
es.create(index="my_index", doc_type='test', body=doc, id=1)

s = Search(using=es, index="my_index", doc_type='test').query('match', message="jump")
res = s.execute()
print(res.hits.total)
print(res[0].message)

s = Search(using=es, index="my_index", doc_type='test').query('match', message="uber")
res = s.execute()
print(res.hits.total)
print(res[0].message)

