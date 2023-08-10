get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')

import warnings
warnings.filterwarnings('ignore')

# !pip install pyelasticsearch

from pyelasticsearch import ElasticSearch, bulk_chunks
import pandas as pd

ES_HOST = 'http://localhost:9200/'
INDEX_NAME = "expo2009"
DOC_TYPE = "flight"

# ElasticSearch?

es = ElasticSearch(ES_HOST)

es.count('*')['count']

# init index
try :
    es.delete_index(INDEX_NAME)
    print('Deleting %s'%(INDEX_NAME))
except :
    print('ERROR: Deleting %s failed!'%(INDEX_NAME))
    pass

es.create_index(index=INDEX_NAME)

# https://pyelasticsearch.readthedocs.io/en/latest/api/#pyelasticsearch.ElasticSearch.put_mapping
# https://www.elastic.co/guide/en/elasticsearch/reference/current/null-value.html
mapping = {
    'flight': {
        'properties': {
            'SecurityDelay': {
                'type': 'integer',
                'null_value': -1
            },
            'FlightNum': {
                'type': 'text'
            },
            'Origin': {
                'type': 'keyword'
            },
            'LateAircraftDelay': {
                'type': 'integer',
                'null_value': -1
            },
            'NASDelay': {
                'type': 'integer',
                'null_value': -1
            },
            'ArrTime': {
                'type': 'integer'
            },
            'AirTime': {
                'type': 'integer'
            },
            'DepTime': {
                'type': 'integer'
            },
            'Month': {
                'type': 'string'
            },
            'CRSElapsedTime': {
                'type': 'integer'
            },
            'DayofMonth': {
                'type': 'string'
            },
            'Distance': {
                'type': 'integer'
            },
            'CRSDepTime': {
                'type': 'integer',
            },
            'DayOfWeek': {
                'type': 'keyword'
            },
            'CancellationCode': {
                'type': 'keyword'
            },
            'Dest': {
                'type': 'keyword'
            },
            'DepDelay': {
                'type': 'integer'
            },
            'TaxiIn': {
                'type': 'integer'
            },
            'UniqueCarrier': {
                'type': 'keyword'
            },
            'ArrDelay': {
                'type': 'integer'
            },
            'Cancelled': {
                'type': 'boolean'
            },
            'Diverted': {
                'type': 'boolean'
            },
            'message': {
                'type': 'text'
            },
            'TaxiOut': {
                'type': 'integer'
            },
            'ActualElapsedTime': {
                'type': 'integer'
            },
            'CarrierDelay': {
                'type': 'integer',
                'null_value': -1
            },
            '@timestamp': {
                'format': 'strict_date_optional_time||epoch_millis',
                'type': 'date'
            },
            'Year': {
                'type': 'keyword'
            },
            'WeatherDelay': {
                'type': 'integer',
                'null_value': -1
            },
            'CRSArrTime': {
                'type': 'integer'
            },
            'TailNum': {
                'type': 'text'
            }
        }
    }

}
es.put_mapping(index=INDEX_NAME, doc_type=DOC_TYPE,mapping=mapping )

es.count('*')['count']

# if import fails, we can selectivly remove entries

# GET expo2009/_search
# {
#   "query": {
#     "range": {
#         "@timestamp" : { "gte" : "2002-01-01T00:00:00" }
#     }
#   }
# }

# # https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-delete-by-query.html

# POST expo2009/_delete_by_query
# {
#   "query": { 
#     "range": {
#         "@timestamp" : { "gte" : "2002-01-01T00:00:00" }
#     }
#   }
# }

# curl -XPOST "http://localhost:9200/expo2009/_delete_by_query" -H 'Content-Type: application/json' -d'
# {
#   "query": { 
#     "range": {
#         "@timestamp" : { "gte" : "2002-01-01T00:00:00" }
#     }
#   }
# }'



