import networkx as nx
import matplotlib.pyplot as plt

k6 = nx.complete_graph(6)
nx.draw_circular(k6, with_labels=True)
plt.show()

p6 = nx.path_graph(6)
nx.draw_circular(p6, with_labels=True)
plt.show()

g = nx.Graph([(0, 1), (0, 3), (0, 7), (1, 2), (2, 3), (3, 4),
 (4, 5), (4, 7), (5, 5), (5, 6), (6, 7)])

nx.draw_circular(g, with_labels = True)
plt.show()

g = nx.Graph([(0, 1), (0, 3), (0, 7), (1, 2), (2, 3), (3, 4),
 (4, 5), (4, 7), (5, 5), (5, 6), (6, 7)])

nx.diameter(g)

nx.diameter(nx.complete_graph(10))

nx.average_shortest_path_length(g)



