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

text = 'The QUICK Brown FOX!'# contains some uppercase words
analyzed_text = [x['token'] for x in es.indices.analyze                 (tokenizer='standard', filter=['lowercase'], text=text)['tokens']]
print(','.join(analyzed_text))

# first delete the index from previous chapters, if it exists
if es.indices.exists('my_index'): 
    es.indices.delete('my_index')

#es.indices.create('my_index')
from elasticsearch_dsl import analyzer, Index
my_custom_analyzer = analyzer('my_lowercaser',
        tokenizer='standard',
        filter='lowercase')
i = Index('my_index')
i.analyzer(my_custom_analyzer)

es.indices.analyze(index='my_index', analyzer='my_lowercaser', text=text)



