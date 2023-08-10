import networkx as nx

nx.__version__

g = nx.Graph()

g.add_edge('a', 'b', weight=0.1)
g.add_edge('b', 'c', weight=1.5)
g.add_edge('a', 'c', weight=1.0)
g.add_edge('c', 'd', weight=2.2) 

g.add_node('e')

get_ipython().magic('matplotlib inline')

nx.draw(g, with_labels=True)

pos = nx.spring_layout(g) #this will create a common layout for nodes; a dict mapping each node to a position
nx.draw(g, pos, with_labels=True) #we can now draw the same graph using these specific positions for nodes
nx.draw_networkx_edge_labels(g,pos,edge_labels=nx.get_edge_attributes(g,'weight')); #and add edge labels using the same positions

nx.shortest_path(g, source='b', target='d')

from networkx import NetworkXNoPath
try:
    print(nx.shortest_path(g, source='b', target='e'))
except NetworkXNoPath:
    print('No path between node b and e.')

nx.shortest_path(g, 'b', 'd', weight='weight')

nx.shortest_path(g, 'b')

nx.shortest_path(g, target='b')

nx.shortest_path(g)

