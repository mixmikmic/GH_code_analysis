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

#english (language)
text = 'I\'m not happy about the foxes'
analyzed_text = [x['token'] for x in es.indices.analyze                 (analyzer='english', body=text)['tokens']]
print(','.join(analyzed_text))

index_template = {
  "mappings": {
    "blog": {
      "properties": {
        "title": { 
          "type": "text",
          "fields": {
            "english": { 
              "type":     "text",
              "analyzer": "english"
            }
          }
        }
      }
    }
  }
}

es.indices.create(index='my_index', body=index_template)

data = { "title": "I'm happy for this fox" }
es.create(index='my_index', doc_type='blog', body=data, id=1)

data = { "title": "I'm not happy about my fox problem" }
es.create(index='my_index', doc_type='blog', body=data, id=2)

s = Search(using=es, index='my_index', doc_type='blog')
q = Q('multi_match', type='most_fields', query='not happy foxes', fields=['title', 'title.english'])
s = s.query()
res = s.execute()
for hit in res:
    print(hit.title)

es.indices.delete(index='my_index')
index_template_with_exclusions = {
  "settings": {
    "analysis": {
      "analyzer": {
        "my_english": {
          "type": "english",
          "stem_exclusion": [ "organization", "organizations" ], 
          "stopwords": [ 
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
            "if", "in", "into", "is", "it", "of", "on", "or", "such", "that",
            "the", "their", "then", "there", "these", "they", "this", "to",
            "was", "will", "with"
          ]
        }
      }
    }
  }
}

es.indices.create(index='my_index', body=index_template_with_exclusions)

#english (language) with exclusions - my_english
text = 'The World Health Organization does not sell organs.'
analyzed_text = [x['token'] for x in es.indices.analyze                 (index='my_index', analyzer='my_english', body=text)['tokens']]
print(','.join(analyzed_text))



