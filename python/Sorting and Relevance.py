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

# And a filter-only search
s = Search(using=es)
s = s.filter('term', user_id=1)
res = s.execute()

for hit in res:
    print('Score:{}'.format(hit.meta.score))

res

# Or we can make sure the items have a constant non-zero score
s = Search(using=es).query('constant_score', filter=Q('term', user_id=1))
res = s.execute()

for hit in res:
    print('Score:{} with date of {}'.format(hit.meta.score,hit.date))

s = Search(using=es).query('bool', filter=Q('term', user_id=1))
s = s.sort({ "date": { "order": "desc" }})
res = s.execute()
# Now is date descending order:
for hit in res:
    print('Score:{} with date of {} and sort field:{}'
          .format(hit.meta.score,hit.date,hit.meta.sort))

s = Search(using=es).query('bool', 
                           must=Q('match', tweet='manage text search'),
                           filter=Q('term', user_id=2))
s = s.sort({ "date":   { "order": "desc" }}, { "_score": { "order": "desc" }})
#s = s.sort("date","_score") # sorted by date first
res = s.execute()

for hit in res:
    print('Score:{} with date of {} and sort field:{}'
          .format(hit.meta.score,hit.date,hit.meta.sort))

doc1 = {
    'title': 'How I Met Your Mother',
    'date': '2013-01-01',
    'ratings': [2,3,1,3,4,5,5,5,3,4,2]
}
doc2 = {
    'title': 'Breaking Bad',
    'date': '2013-01-01',
    'ratings': [5,5,4,3,4,5,5,5,3,5,5]
}
es.create(index='shows', doc_type='tv_series', body=doc1, id=1)
es.create(index='shows', doc_type='tv_series', body=doc2, id=2)

s = Search(using=es)
s = s.sort({ "ratings":   { "order": "desc", "mode":"avg" }})
#s = s.sort("date","_score") # sorted by date first
res = s.execute()

for hit in res:
    print(hit.title, hit.meta.sort)

r = index.populate(template=2)

s = Search(using=es).query(Q('match', tweet='elasticsearch'))
s = s.sort("tweet.raw")
res = s.execute()

for hit in res:
    print(hit.meta.sort)

s = Search(using=es).query(Q('match', tweet='honeymoon'))
s = s.extra(explain=True)
res = s.execute()

index.RenderJSON(res['hits']['hits'])

s = Search(using=es).query(Q('match', tweet='honeymoon') & Q('match', _id=12))
s = s.extra(explain=True)
res = s.execute()

index.RenderJSON(res['hits']['hits'])



