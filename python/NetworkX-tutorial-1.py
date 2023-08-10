import networkx as nx

G = nx.Graph()

G = nx.DiGraph()

G = nx.MultiGraph()

G = nx.MultiDiGraph()

G = nx.Graph()
G.add_node(1)           # Adding one node at a time
nlist = [2, 3]          
G.add_nodes_from(nlist) # Adding a list of nodes

list(G.nodes()) # Accessing nodes

G.add_edge(1, 2)         # Adding one edge at a time
elist = [(2, 3), (1, 3)]
G.add_edges_from(elist)  # Adding a list of edges

list(G.edges()) # Accessing edges

