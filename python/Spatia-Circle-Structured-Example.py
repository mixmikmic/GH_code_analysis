# Python 3 
get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import random

# package for creation and visuliazation of networks
import networkx as nx 

import axelrod as axl

strategies = [s() for s in axl.ordinary_strategies]

players = random.sample(strategies, 10)

G = nx.cycle_graph(len(players))

pos = nx.circular_layout(G)
# for the nodes 
nx.draw_networkx_nodes(G,pos,
                       node_color='r',
                       node_size=100
                       )
# for the edges
nx.draw_networkx_edges(G,pos,
                       width=5, alpha = 0.5)
# using labels
labels={}
for i in range(len(players)):
    labels[i]= '%s' % (players[i])

nx.draw_networkx_labels(G,pos,labels,font_size=10)
plt.show()

edges = G.edges()
tournament = axl.Tournament(players, edges=G.edges(), repetitions=1)
results = tournament.play(processes=1)

results.ranked_names

plot = axl.Plot(results)
plot.boxplot();

