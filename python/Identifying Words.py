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

# Whitespace tokenizer
text = "You're the 1st runner home!"
analyzed_text = [x['token'] for x in es.indices.analyze                 (tokenizer='whitespace', body=text)['tokens']]
print(','.join(analyzed_text))

# Standard tokenizer - uses Unicode Text Segmentation standard
text = "You're my co-opted 'favorite' cool_dude." # single quotes 'favorite'
analyzed_text = [x['token'] for x in es.indices.analyze                 (tokenizer='standard', body=text)['tokens']]
print(','.join(analyzed_text))

# Standard tokenizer - uses Unicode Text Segmentation standard
# Note that string contains an email address
text = "You're my co-opted 'favorite' cool_dude. Pls email me friend@dude.it"
analyzed_text = [x['token'] for x in es.indices.analyze                 (tokenizer='standard', body=text)['tokens']]
print(','.join(analyzed_text))

# Standard tokenizer - uses Unicode Text Segmentation standard
text = "You're my co-opted 'favorite' cool_dude. Pls email me friend@dude.it"
analyzed_text = [x['token'] for x in es.indices.analyze                 (tokenizer='uax_url_email', text=text)['tokens']]
print(','.join(analyzed_text))

text = '<p>Some d&eacute;j&agrave; vu <a href="http://somedomain.com>">website</a>'

from elasticsearch_dsl import analyzer, Index

my_custom_analyzer = analyzer('my_html_analyzer',
        tokenizer='standard',
        char_filter='html_strip')

i = Index('my_index')

i.analyzer(my_custom_analyzer)

analyzed_text = [x['token'] for x in es.indices.analyze                 (index='my_index', analyzer='my_html_analyzer', text=text)['tokens']]
print(','.join(analyzed_text))



