import networkx as nx

get_ipython().magic('matplotlib inline')

G = nx.gnm_random_graph(10, 20)   # 10 nodes and 20 edges

for v in G.nodes():   # Some properties of the graph
    print v, G.degree(v) , nx.clustering(G,v)

list(G.adjacency())  # adjacency list of the graph

nx.draw(G)

