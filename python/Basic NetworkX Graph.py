import networkx as nx
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

G = nx.DiGraph()

G.add_node(1)
G.add_node("Sam")
G.add_node(1.567)

G.nodes()

G.add_edge("Sam", 1.567)
G.add_edge(1, "Sam")

G.edges()

G.node[1]['Number'] = True

G.node[1]

G[1]['Sam']['weight'] = 5

G[1]

G[1]['Sam']

# See number of connections per node
nx.degree(G)

G.degree("Sam")

# Find connectors between nodes and paths between
for path in nx.all_simple_paths(G, source=1, target=1.567):
    print(path)

import random
new_nodes = []

# Add 10 new nodes
for i in range(1, 10):
    G.add_node(i)
    new_nodes.append(i)

# Make sure each node is connected to at least one other node
for i in range(1, 10):
    G.add_edge(i, new_nodes.pop())

paths = list(nx.shortest_simple_paths(G, 1, 9))
print(paths)

# The simplest rendering
nx.draw(G)
plt.show()



