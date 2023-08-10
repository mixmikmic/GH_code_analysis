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

res = es.search('_all') # same as es.search()

#from pprint import pprint
#pprint(res)

s = Search(using=es)
response = s.execute()
response

res = es.search('_all', timeout='10ms') # same as es.search(timeout='10ms')

# To see the results, we can iterate:
# Elasticsearch pages the results (to 10 hits)
for hit in s:
    print(hit)

#/_search
#Search all types in all indices
res = es.search('_all')

#/gb/_search
#Search all types in the gb index
res = es.search(index='gb')

#/gb,us/_search
#Search all types in the gb and us indices
res = es.search(index=['gb','us'])

#/g*,u*/_search
#Search all types in any indices beginning with g or beginning with u
res = es.search(index=['g*','u*'])

#/gb/user/_search
#Search type user in the gb index
res = es.search(index='gb', doc_type='user')

#/gb,us/user,tweet/_search
#Search types user and tweet in the gb and us indices
res = es.search(index=['g*','u*'], doc_type=['user', 'tweet'])
print(res['hits']['total'])

#/_all/user,tweet/_search
#Search types user and tweet in all indices
res = es.search(doc_type=['user', 'tweet'])
print(res['hits']['total'])

#/_search
#Search all types in all indices
s = Search(using=es)
response = s.execute()

#/gb/_search
#Search all types in the gb index
s = Search(using=es, index='gb')
response = s.execute()

#/gb,us/_search
#Search all types in the gb and us indices
s = Search(using=es, index=['gb','us'])
response = s.execute()

#/g*,u*/_search
#Search all types in any indices beginning with g or beginning with u
s = Search(using=es, index=['g*','u*'])
response = s.execute()

#/gb/user/_search
#Search type user in the gb index
s = Search(using=es, index=['g*','u*'], doc_type='user')
response = s.execute()


#/gb,us/user,tweet/_search
#Search types user and tweet in the gb and us indices
s = Search(using=es, index=['g*','u*'], doc_type=['user','tweet'])
response = s.execute()

#/_all/user,tweet/_search
#Search types user and tweet in all indices
s = Search(using=es, doc_type=['user','tweet'])
response = s.execute()
print(response.hits.total)
print(len(res['hits']['hits']))

# For search API:
res = es.search(doc_type=['user', 'tweet'], from_=5, size=5)

print(res['hits']['total'])
print(len(res['hits']['hits']))

res = es.search(doc_type=['tweet'], q='tweet:elasticsearch')
print('Total hits:{}\n'.format(res['hits']['total']))
pprint(res['hits']['hits'][0])

s = Search(using=es, doc_type=['tweet'])     .query('match', tweet='elasticsearch')
response = s.execute()
print('Total hits:{}\n'.format(response.hits.total))
pprint(response[0].meta)

for hit in response:
    print(hit.tweet)

res = es.search(q='mary')
print('Total hits:{}\n'.format(res['hits']['total']))
pprint(res['hits']['hits'][0])

s = Search(using=es)     .query('match', _all='mary')
response = s.execute()
print('Total hits:{}\n'.format(response.hits.total))
print(response[0].tweet)

res = es.search(q='+name:(mary john) +date:>2014-09-10 +(aggregations geo)')
print('Total hits:{}\n'.format(res['hits']['total']))
pprint(res['hits']['hits'][0])







