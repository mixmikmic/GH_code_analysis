import networkx as nx

# Create a directed networkx graph
G = nx.DiGraph()

# Add web pages (graph nodes)
for i in range(5):
   G.add_node(i, label="p" + str(i))

# Add outgoing web links with weights (directed graph edges)
G.add_edge(0, 1, weight=1.0/4.0, label="1/4")
G.add_edge(0, 2, weight=1.0/4.0, label="1/4")
G.add_edge(0, 3, weight=1.0/4.0, label="1/4")
G.add_edge(0, 4, weight=1.0/4.0, label="1/4")


G.add_edge(1, 2, weight=1.0/3.0, label="1/3")
G.add_edge(1, 3, weight=1.0/3.0, label="1/3")
G.add_edge(1, 4, weight=1.0/3.0, label="1/3")

G.add_edge(2, 0, weight=1.0/2.0, label="1/2")
G.add_edge(2, 1, weight=1.0/2.0, label="1/2")

G.add_edge(3, 0, weight=1.0/2.0, label="1/2")
G.add_edge(3, 2, weight=1.0/2.0, label="1/2")

G.add_edge(4, 1, weight=1.0/3.0, label="1/3")
G.add_edge(4, 2, weight=1.0/3.0, label="1/3")
G.add_edge(4, 3, weight=1.0/3.0, label="1/3")

print(nx.__version__)

# To plot graph, convert to a PyGraphviz graph for drawing
Ag = nx.nx_agraph.to_agraph(G)
Ag.layout(prog='dot')
Ag.draw('web.svg')
from IPython.display import SVG
SVG('web.svg')



