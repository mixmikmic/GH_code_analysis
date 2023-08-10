import warnings
warnings.filterwarnings('ignore')

import networkx as nx
import matplotlib.pyplot as plt

g = nx.Graph({0: [1, 3, 4, 5, 6, 7],
 1: [0, 3, 4, 5, 6, 7],
 2: [3, 5, 6, 7],
 3: [0, 1, 2, 4, 6],
 4: [0, 1, 3, 5, 7],
 5: [0, 1, 2, 4, 6, 7],
 6: [0, 1, 2, 3, 5, 7],
 7: [0, 1, 2, 4, 5, 6]})

nx.draw_circular(g, with_labels=True)
plt.show()

nx.coloring.greedy_color(g)

# This is the default greedy algorithm -- the strategy does not 
# actually need to be stated. 
nx.coloring.greedy_color(g, strategy=nx.coloring.strategy_largest_first)

# This one randomly selects vertices one at a time and builds a 
# proper coloring. Notice the result is different here. 
nx.coloring.greedy_color(g, strategy=nx.coloring.strategy_random_sequential)

nx.coloring.greedy_color(g, strategy=nx.coloring.strategy_connected_sequential)

nx.coloring.greedy_color(g, strategy=nx.coloring.strategy_independent_set)

nx.coloring.greedy_color(g, strategy=nx.coloring.strategy_smallest_last)

k44 = nx.complete_bipartite_graph(4,4)
nx.coloring.greedy_color(k44)

nx.coloring.greedy_color(g, strategy=nx.coloring.strategy_random_sequential)

# First let's remind ourselves what the graph looked like

nx.draw_circular(g, with_labels=True)
plt.show()

# And the coloring: 

nx.coloring.greedy_color(g)

# The color classes are {0,2}, {1}, {3,5}, {4,6}, and {7}. 
# Here is some code that will assign those red, blue, yellow, green, and purple. 

# Specify the layout; in this case "circular" 
pos = nx.circular_layout(g)

# Now draw just the nodes, one color class at a time. 
# Here we have to specify the layout (`pos`) and the list of nodes. 
nx.draw_networkx_nodes(g, pos, nodelist=[0,2], node_color='red')
nx.draw_networkx_nodes(g, pos, nodelist=[1], node_color='blue')
nx.draw_networkx_nodes(g, pos, nodelist=[3,5], node_color='yellow')
nx.draw_networkx_nodes(g, pos, nodelist=[4,6], node_color='green')
nx.draw_networkx_nodes(g, pos, nodelist=[7], node_color='purple')

# Now draw the edges and labels: 
nx.draw_networkx_edges(g, pos)
nx.draw_networkx_labels(g, pos)

# Turn off the axes: 
plt.axis('off')

# Then plot the whole thing: 
plt.show()

# In this coloring, nodes 0-3 are one color and 4-7 are another. 

k44 = nx.complete_bipartite_graph(4,4)
nx.coloring.greedy_color(k44)

# Specify the layout; in this case with a "random" layout.
pos = nx.random_layout(k44)

# Now draw just the nodes, one color class at a time. 
# Here we have to specify the layout (`pos`) and the list of nodes. 
nx.draw_networkx_nodes(k44, pos, nodelist=[0,1,2,3], node_color='red')
nx.draw_networkx_nodes(k44, pos, nodelist=[4,5,6,7], node_color='blue')

# Now draw the edges and labels: 
nx.draw_networkx_edges(k44, pos)
nx.draw_networkx_labels(k44, pos)

# Turn off the axes: 
plt.axis('off')

# Then plot the whole thing: 
plt.show()



