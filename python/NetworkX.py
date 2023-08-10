get_ipython().magic('matplotlib inline')

from operator import itemgetter
import networkx as nx

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
from io import StringIO
import pydotplus
from IPython.display import SVG, display

sns.set_context("poster")
sns.set_style("ticks")

DATA_DIR="../data"
INPUT_NETWORK=os.path.join(DATA_DIR, "lesmis","lesmis.gml")
INPUT_NETWORK

G = nx.read_gml(INPUT_NETWORK)
#nx.write_gml(G, "../data/lesmis/lesmis.paj.gml")

df_node_degree = pd.DataFrame(list(dict(G.degree()).items()), columns=["node_name", "degree"])

df_node_degree.sort_values("degree", ascending=False).head(10)

print("radius: {:d}\n".format(nx.radius(G)))
print("diameter: {:d}\n".format(nx.diameter(G)))
print("eccentricity: {}\n".format(nx.eccentricity(G)))
print("center: {}\n".format(nx.center(G)))
print("periphery: {}\n".format(nx.periphery(G)))
print("density: {:f}".format(nx.density(G)))

connected_components = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
print("{} connected components found.".format(len(connected_components)))

nx.draw(G)

fig, ax = plt.subplots(1,1, figsize=(16,16))
nx.draw_networkx(
    G, with_labels=True,
    node_size=[x[1]*10 for x in G.degree_iter()],
    pos=nx.spring_layout(G),
    node_color="g",
    font_size=8,
    ax=ax)
ax.axis("off")

def show_graph(G, file_path):
    dotfile = StringIO()
    nx.drawing.nx_pydot.write_dot(G, dotfile)
    pydotplus.graph_from_dot_data(dotfile.getvalue()).write_svg(file_path)
    display(SVG(file_path))

show_graph(G, "../output/lesmis.svg")

