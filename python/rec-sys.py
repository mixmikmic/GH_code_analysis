get_ipython().system('pip install py2neo == 2.0.8')
from py2neo import Graph
graph = Graph()

query = """
MATCH (group:Group)-[:HAS_TOPIC]->(topic)
WHERE topic.name CONTAINS "Python" 
RETURN group.name, COLLECT(topic.name) AS topics
"""

result = graph.cypher.execute(query)

for row in result:
    print(row) 

get_ipython().system('pip install python-igraph')
from igraph import Graph as IGraph

query = """
MATCH (topic:Topic)<-[:HAS_TOPIC]-()-[:HAS_TOPIC]->(other:Topic)
WHERE ID(topic) < ID(other)
RETURN topic.name, other.name, COUNT(*) AS weight
ORDER BY weight DESC
LIMIT 10
"""

graph.cypher.execute(query)

query = """
MATCH (topic:Topic)<-[:HAS_TOPIC]-()-[:HAS_TOPIC]->(other:Topic)
WHERE ID(topic) < ID(other)
RETURN topic.name, other.name, COUNT(*) AS weight
"""

ig = IGraph.TupleList(graph.cypher.execute(query), weights=True)
ig

clusters = IGraph.community_walktrap(ig, weights="weight")
clusters = clusters.as_clustering()
len(clusters)

nodes = [node["name"] for node in ig.vs]
nodes = [{"id": x, "label": x} for x in nodes]
nodes[:5]

for node in nodes:
    idx = ig.vs.find(name=node["id"]).index
    node["group"] = clusters.membership[idx]
    
nodes[:5]

query = """
UNWIND {params} AS p 
MATCH (t:Topic {name: p.id}) 
MERGE (cluster:Cluster {name: p.group})
MERGE (t)-[:IN_CLUSTER]->(cluster)
"""

graph.cypher.execute(query, params = nodes)

nodes



