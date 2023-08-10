from elasticsearch import Elasticsearch
import pandas as pd
es = Elasticsearch(urls=['localhost'], port=9200)

example_query = {
    'query': {
        'bool': {
            'must': [
                {
                    'match': {
                     'field': 'a'   
                    }
                },
                {
                    'match': {
                     'field': 'b'   
                    }
                },
                {
                    'bool': {
                        'should': [
                            {
                                'match': {
                                    'field': 'c'
                                }
                            },
                            {
                                'match': {
                                    'field': 'd'
                                }
                            }
                        ]
                    }
                }
            ]
        }
    }
}

must_query = {
    'query': {
        'bool': {
            'must': [
                {
                    'match': {
                        'description': 'art'                        
                    }
                },
                {
                    'match': {
                        'description': 'australian'
                    }
                }
            ]
        }
    }
}

should_query = {
    'query': {
        'bool': {
            'should': [
                {
                    'match': {
                        'description': 'art'                        
                    }
                },
                {
                    'match': {
                        'description': 'australian'
                    }
                }
            ]
        }
    }
}

# curl -X GET localhost:9200/goma/_search -d @must_query.json
must_res = es.search(index='goma', body=must_query)
should_res = es.search(index='goma', body=should_query)

pd.DataFrame([must_res['hits']['total'], should_res['hits']['total']], 
             index=['must', 'should'],
             columns=['Number of results'])

