from elasticsearch import Elasticsearch
import pandas as pd
es = Elasticsearch(urls=['localhost'], port=9200)

# This query will retrieve every document in the index.
query = {
    'query': {
        'match_all': {}
    }
}

# Send a search request to Elasticsearch.
# curl -X GET localhost:9200/goma/_search -H 'Content-Type: application/json' -d @query.json
res = es.search(index='goma', body=query)

# The response is a json object, the listing is nested inside it.
# Here we are accessing the first hit in the listing.
res['hits']['hits'][0]

# curl -X GET localhost:9200/goma/event/AV19Sgi4jk6MoKTLfifp/_termvectors?term_statistics&fields=description
res = es.termvectors(index='goma', doc_type='event', id='AV19Sgi4jk6MoKTLfifp', 
                     fields=['description'], term_statistics=True)

# We don't really care that much about the additional info, let's get straight to the point.
tv = res['term_vectors']['description']
tv

pd.DataFrame(tv['field_statistics'], index=['count'])

terms = []
for term in tv['terms']:
    term_info = tv['terms'][term].copy()
    del(term_info['tokens'])
    term_info.update({'term': term})
    terms.append(term_info)
df = pd.DataFrame(terms).set_index('term')
df[0:10]

# Sorted by doc_freq
df.sort_values(by='doc_freq', ascending=False)[0:10]

# Sorted by term_freq
df.sort_values(by='term_freq', ascending=False)[0:10]

# Sorted by ttf
df.sort_values(by='ttf', ascending=False)[0:10]

