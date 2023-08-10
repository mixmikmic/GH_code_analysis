# Code using neo4j.v1 (Official Python driver for Neo4j) -- Not using this
'''
from neo4j.v1 import GraphDatabase, basic_auth

driver = GraphDatabase.driver("bolt://ec2-52-23-203-124.compute-1.amazonaws.com:7687", auth=basic_auth("neo4j", "neo4j2"))
session = driver.session()

# Neo4j Query
result = session.run('match (p:Physician)-[r]-() where p.zipCode = "' + zipcode + '" RETURN p,r LIMIT 25')

for record in result:
    print (record)
    print 'hello'


# Closing session
session.close()

# Code using %%cypher for directly converting result set into Networkx graph

%load_ext cypher
# Note ipython-cypher can only be run through ipython at the time being

%%cypher http://neo4j:neo4j2@ec2-52-23-203-124.compute-1.amazonaws.com:7474/db/data
match (n) return n limit 1


%%cypher http://neo4j:neo4j2@ec2-52-23-203-124.compute-1.amazonaws.com:7474/db/data
match (p:Physician)-[r:SHARED_PATIENTS]-() where p.stateName = "TX" RETURN p,r limit 4

results = %cypher match (p:Physician)-[r:SHARED_PATIENTS]-() where p.stateName = "TX" RETURN p,r limit 4'''

import operator
import time
from neo4jrestclient.client import GraphDatabase
from neo4jrestclient import client

# Connection using neo4jrestclient
gdb = GraphDatabase("http://ec2-52-23-203-124.compute-1.amazonaws.com:7474/db/data", username="neo4j", password="neo4j2")

import pickle

# Input city here
city = 'SAN FRANCISCO'

# Query 1

q1 = 'MATCH (p:Physician)-[r:SHARED_PATIENTS]->(p2:Physician) WHERE p.city = "'+ city +'"  RETURN distinct p,r,p2'

unique_count = {}


start = time.clock()

results = gdb.query(q1, returns=(client.Node, client.Relationship, client.Node))
for r in results:
    #print r[0]
    if('firstName' in r[0].properties):
        first_node = r[0]
        print first_node.properties['firstName'],r[0].properties['lastName']
        print '----'
        #print r[1]
    
    relationship = r[1]
    print relationship.properties
    print '-----'
    
    if('firstName' in r[2].properties):   
        print r[2].properties['firstName'],r[2].properties['lastName']
        print '---**----' 
    
        if(unique_count.has_key(first_node.id)):
            # Add the value to the count
            unique_count[first_node.id] = unique_count[first_node.id] + int(relationship.properties['unique'])
        else:
            unique_count[first_node.id] =  int(relationship.properties['unique'])
    #print("(%s)-[%s]->(%s)" % (r[0]["name"], r[1], r[2]["name"]))
# The output:
# (Marco)-[likes]->(Punk IPA)
# (Marco)-[likes]->(Hoegaarden Rosee)
print 'Time taken', time.clock() - start

pickle.dump(results, open('San_Francisco_physicians.p', 'wb'))

pickle.dump(unique_count, open('San_Francisco_unique_counts.p','wb'))

top_physician_id = max(unique_count.iteritems(), key=operator.itemgetter(1))[0]

for r in results:
    if(r[0].id == top_physician_id):
        print r[0].properties['firstName'],r[0].properties['lastName']
        break

top_physician_id

for tupl in sorted(unique_count.iteritems(), key = operator.itemgetter(1)):
    print tupl[0]

id_rank_list = []
rank = len(unique_count)
for tupl in sorted(unique_count.iteritems(), key = operator.itemgetter(1)):
    id_rank_list.append((tupl[0], rank))
    rank = rank - 1

id_rank_list

# Query 2 gives the results of interconnected physicians that work in the given city
q2 = 'MATCH (p:Physician)-[r:SHARED_PATIENTS]-(p2:Physician) WHERE p.city = "'+ city +'" and p2.city = "'+ city + '" RETURN p,r,p2'

results = gdb.query(q1, returns=(client.Node, client.Relationship, client.Node))

# results available for conversion into NetworkX graph and further CNN Computation

