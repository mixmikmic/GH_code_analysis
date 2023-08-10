# Ignore this; it suppresses system warnings. 

import warnings
warnings.filterwarnings('ignore')

# Include these two lines whenever you want to work with graphs in Python. 
# The first line loads networkx, and gives it the nickname "nx". 
# The second loads matplotlib's plotting library, and gives it the nickname "plt". 
# Pro tip: Save these commands as a text snippet or in a text file for easy access. 

import networkx as nx
import matplotlib.pyplot as plt

# Create from an edge list. 
# Note the nx. and the capital "G". 

g1 = nx.Graph([(0,1), (0,2), (0,3), (1,2), (1,3)])

# Create from a dictionary. 

g2 = nx.Graph({'a': ['b', 'c', 'd', 'e'], 'b': ['c','e'], 'c': ['a', 'd'], 'd': ['e']})

# Counting the number of nodes or edges in a graph: 

g1.number_of_nodes()

g2.number_of_edges()

# Getting a list of the nodes or edges in a graph

g1.nodes()

g2.edges() 

# Getting the degree of a node 

g1.degree(1)

g2.degree('b')

# Getting the degree sequence of a graph: Same command but leave the argument off. 

g1.degree()

g2.degree()

for node in g2.nodes():
    print("The degree of node %s is %d." % (node, g2.degree(node)))

# Generate the complete graph on 7 nodes:

k7 = nx.complete_graph(7)

# Now check to see if it worked: 

print(k7.nodes())
print(k7.number_of_edges())
print(k7.degree())

# Generate the complete bipartite graph K_{3,4}:

k34 = nx.complete_bipartite_graph(3,4)
k34.edges()

# Generate the path graph P_6:

p6 = nx.path_graph(6)
p6.nodes()

# Generate the cycle graph C_{10}:
c10 = nx.cycle_graph(10)
c10.degree()

# Specify number of nodes and edges: This is called a "GMN" graph. 
# Note that we have to go into networkX, then a library for random graphs and pull a function out. 

my_random = nx.random_graphs.gnm_random_graph(5,10)
my_random.edges()

# Specify number of nodes and a probability value: This is called a "GNP" graph. 

another_random = nx.random_graphs.gnp_random_graph(10, 0.55)
another_random.degree()

k5 = nx.complete_graph(5) # Complete graph K_5
k5.edges()

nx.to_dict_of_lists(k5)

nx.adjacency_matrix(k5)

# Example: Drawing the graph g1 from earlier 

nx.draw(g1)
plt.show()

nx.draw(g1, with_labels=True)
plt.show()

# Another example: g2
# We'll add options to change the color of the nodes and the thickness of the edges

nx.draw(g2, with_labels=True, node_color="yellow", width=3)
plt.show()

# Example: Visualizing K_{3,7} with gray nodes and thin dashed edges

k37 = nx.complete_bipartite_graph(3,7)
nx.draw(k37, with_labels=True, node_color="gray", width=0.5, style="dashed")
plt.show()

# Example: Using a "circular" layout to plot K_{3,7}
# This uses the function nx.draw_circular(): 

nx.draw_circular(k37, with_labels=True, node_color="gray", width=0.5, style="dashed")
plt.show()

# Example: Same thing but using g2 from earlier

nx.draw_circular(g2, with_labels=True, node_color="#f442e8", width = 4, style="dotted")
plt.show()

random_g = nx.random_graphs.gnp_random_graph(20, 0.2)
nx.draw(random_g, with_labels=True, node_color="#42f46e", width = 2, node_shape="s")
plt.show()



