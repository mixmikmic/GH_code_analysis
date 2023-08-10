import networkx
import numpy

import cncp

import matplotlib as mpl
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import seaborn

g = networkx.Graph()
g.add_nodes_from([1, 2, 3])
g.add_edges_from([(1, 2), (1, 2), (2, 3)])

g.edges(1)

h = networkx.MultiGraph()
h.add_nodes_from([1, 2, 3])
h.add_edges_from([(1, 2), (1, 2), (2, 3)])
h.edges(1)

fig = plt.figure(figsize = (3, 3))
plt.gca().set_axis_off()

networkx.draw_networkx(h, node_size = 400)

_ = plt.show()

g2 = networkx.Graph()
g2.add_nodes_from([1, 2, 3])
g2.add_edges_from([(1, 2), (1, 1), (2, 3), (2, 2)])   # includes two self-loops

# extract the edges intersecting node 1
g2.edges(1)

g2.selfloop_edges()

print "Network has {n} self-adjacent nodes labelled {ns}".format(n = g2.number_of_selfloops(),
                                                                 ns = g2.nodes_with_selfloops())

g2.remove_edges_from(g2.selfloop_edges())
print "Network now has {n} self-loops".format(n = g2.number_of_selfloops())

