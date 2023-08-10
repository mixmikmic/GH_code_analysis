import networkx as nx
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt

G = nx.Graph()

G.add_node('A')
G.add_node('B')
G.add_node('C')
G.add_node('D')
G.add_node('E')
G.add_node('F')
G.add_node('G')
G.add_node('H')

#just a nodes of points, hanging out.
nx.draw(G)

G.add_edge('A','C')
G.add_edge('B','C')
G.add_edge('D','C')
G.add_edge('E','D')
G.add_edge('F','D')
G.add_edge('G','F')
G.add_edge('G','H')

pos=nx.spring_layout(G)
nx.draw_networkx_labels(G,pos)
nx.draw(G, pos, node_color='w')

#Eiganvalue Centrality
#centrality=nx.eigenvector_centrality(G, max_iter=1000)
centrality= nx.eigenvector_centrality_numpy(G)

centrality

for key, val in centrality.items():
   centrality[key]= round(val ,2)


pos=nx.spring_layout(G)
nx.draw_networkx_labels(G,pos,centrality,font_size=16)
nx.draw_networkx_nodes(G,pos,node_size=1800, alpha=.6, node_color='w')
nx.draw_networkx_edges(G, pos)
plt.axis('off')
plt.show()






