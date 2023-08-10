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

settings = {
    "analysis" : {
        "analyzer" : {
            "en_GB" : {
                "tokenizer" : "standard",
                "filter" : [ "lowercase", "en_GB" ]
            }
        },
        "filter" : {
            "en_GB" : {
                "type" : "hunspell",
                "locale" : "en_GB"
            }
        }
    }
}
index.create_my_index(body=settings)

# test with the standard English analyzer
text = "You're right about organizing jack's Über generation of waiters." 
analyzed_text = [x['token'] for x in es.indices.analyze                 (index='my_index', analyzer='english', text=text)['tokens']]
print(','.join(analyzed_text))

analyzed_text = [x['token'] for x in es.indices.analyze                 (index='my_index', analyzer='en_GB', text=text)['tokens']]
print(','.join(analyzed_text))

text = "A generically generally generously generated organized waiter."
# English
analyzed_text = [x['token'] for x in es.indices.analyze                 (index='my_index', analyzer='english', text=text)['tokens']]
print(','.join(analyzed_text))

# en_GB Hunspell:
analyzed_text = [x['token'] for x in es.indices.analyze                 (index='my_index', analyzer='en_GB', text=text)['tokens']]
print(','.join(analyzed_text))

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

# my_english custom analyzer:
analyzed_text = [x['token'] for x in es.indices.analyze                 (index='my_index', analyzer='my_english', text=text)['tokens']]
print(','.join(analyzed_text))

porter_token_filter = {
  "settings": {
    "analysis": {
      "filter": {
        "english_stop": {
          "type":       "stop",
          "stopwords":  "_english_"
        },
        "porter": {
          "type":       "stemmer",
          "language":   "porter" 
        },
        "english_possessive_stemmer": {
          "type":       "stemmer",
          "language":   "possessive_english"
        }
      },
      "analyzer": {
        "my_porter_english": {
          "tokenizer":  "standard",
          "filter": [
            "english_possessive_stemmer",
            "lowercase",
            "english_stop",
            "porter", 
            "asciifolding" 
          ]
        }
      }
    }
  }
}
index.create_my_index(body=porter_token_filter)

# my_english custom analyzer:
analyzed_text = [x['token'] for x in es.indices.analyze                 (index='my_index', analyzer='my_porter_english', text=text)['tokens']]
print(','.join(analyzed_text))

porter2_token_filter = {
  "settings": {
    "analysis": {
      "filter": {
        "english_stop": {
          "type":       "stop",
          "stopwords":  "_english_"
        },
        "porter2": {
          "type":       "stemmer",
          "language":   "porter2" 
        },
        "english_possessive_stemmer": {
          "type":       "stemmer",
          "language":   "possessive_english"
        }
      },
      "analyzer": {
        "my_porter2_english": {
          "tokenizer":  "standard",
          "filter": [
            "english_possessive_stemmer",
            "lowercase",
            "english_stop",
            "porter2", 
            "asciifolding" 
          ]
        }
      }
    }
  }
}
index.create_my_index(body=porter2_token_filter)

# my_english custom analyzer:
analyzed_text = [x['token'] for x in es.indices.analyze                 (index='my_index', analyzer='my_porter2_english', text=text)['tokens']]
print(','.join(analyzed_text))

stem_control_settings = {
  "settings": {
    "analysis": {
      "filter": {
        "no_stem": {
          "type": "keyword_marker",
          "keywords": [ "skies" ] 
        }
      },
      "analyzer": {
        "my_stemmer": {
          "tokenizer": "standard",
          "filter": [
            "lowercase",
            "no_stem",
            "porter_stem"
          ]
        }
      }
    }
  }
}
index.create_my_index(body=stem_control_settings)

# my_stemmer custom analyzer:
text = ['sky skies skiing skis']
analyzed_text = [x['token'] for x in es.indices.analyze                 (index='my_index', analyzer='my_stemmer', text=text)['tokens']]
print(','.join(analyzed_text))

my_stemmer_override = {
  "settings": {
    "analysis": {
      "filter": {
        "custom_stem": {
          "type": "stemmer_override",
          "rules": [ 
            "skies=>sky",
            "mice=>mouse",
            "feet=>foot"
          ]
        }
      },
      "analyzer": {
        "my_stemmer_override": {
          "tokenizer": "standard",
          "filter": [
            "lowercase",
            "custom_stem", 
            "porter_stem"
          ]
        }
      }
    }
  }
}
index.create_my_index(body=my_stemmer_override)
# my_stemmer_override custom analyzer:
text = ['The mice came down from the skies and ran over my feet']
analyzed_text = [x['token'] for x in es.indices.analyze                 (index='my_index', analyzer='my_stemmer_override', text=text)['tokens']]
print(','.join(analyzed_text))

