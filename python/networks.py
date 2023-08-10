get_ipython().magic('matplotlib qt4')
from __future__ import division

from collections import OrderedDict, defaultdict

from models import tools, optimize, models, filters
from models.tests import PerformanceTest

import scipy as sp
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

data = tools.load_data(limit=1000000, offset=1000000)

def plot_network(G, offset=23):
    fig = plt.figure(num=None, figsize=(20, 15), dpi=80)

    pos = nx.graphviz_layout(G)
    nx.draw_networkx(
        G,
        pos=pos,
        width=4,
        edge_vmin=0, edge_vmax=1,
        with_labels=False,
        node_size=300,
        node_color='orange',
        edge_color=[G[e[0]][e[1]]['weight'] for e in G.edges()],
        edge_cmap=plt.cm.Blues,
    )

    for p in pos:  # raise text positions
        x, y = pos[p]
        pos[p] = (x, y + offset)
    labels = nx.draw_networkx_labels(G, pos, font_size=16)

    plt.axis('off')
    plt.tight_layout()

places = tools.load_places().T.to_dict()
n = lambda v: tools.to_place_name(v, places=places)

d_corr = data[filters.asian_countries(data)]
print len(d_corr)

pfae = models.PFAExt(models.EloModel())
pfae.train(d_corr)

place_users = {}
correlations = {}
for place_id in pfae.prior.places:
    place_users[place_id] = {
        item.user.id for index, item in pfae.items.items()
        if place_id == index[1]
    }
for i, place_i in enumerate(pfae.prior.places):
    for place_j in pfae.prior.places:
        d = []
        for user_id in place_users[place_i]:
            if user_id in place_users[place_j]:
                d += [(pfae.items[user_id, place_i].knowledge,
                       pfae.items[user_id, place_j].knowledge)]
        correlation = sp.stats.spearmanr(d)
        correlations[place_i, place_j] = correlation
    tools.echo('{}/{}'.format(i+1, len(place_users)))

edges = OrderedDict()
min_c = 0.84
max_c = max(correlation for correlation, pvalue in correlations.values())

for (v1, v2), (correlation, pvalue) in correlations.items():
    if pvalue < 0.05 and v1 != v2 and (v2, v1) not in edges and correlation > min_c:
        edges[v1, v2] = (correlation - min_c) / (max_c - min_c)
nodes = list({e[0] for e in edges} | {e[1] for e in edges})

G = nx.Graph()

for (v1, v2), weight in edges.items():
    G.add_edge(n(v1), n(v2), weight=weight)
for v in nodes:
    G.add_node(n(v))

plot_network(G, offset=15)

d = data[filters.european_countries(data) & (data['is_correct'] == 0)]
places_answered = defaultdict(list)
for _, row in d.T.iteritems():
    if np.isfinite(row.place_answered):
        places_answered[int(row.place_id)].append(int(row.place_answered))

G = nx.Graph()

for v1, answeres in places_answered.iteritems():
    for v2 in set(answeres):
        weight = answeres.count(v2) / len(answeres)
        e = (n(v1), n(v2))
        if tuple(reversed(e)) in G.edges():
            weight += G[e[0]][e[1]]['weight']
        if weight <= 0.14:
            continue
        G.add_edge(e[0], e[1], weight=weight)
    G.add_node(n(v1))

plot_network(G)



