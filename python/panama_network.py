# load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import random

get_ipython().magic('matplotlib inline')
import matplotlib as mpl
mpl.style.use("ggplot")

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from pputils import *

# load the raw data into dataframes and cleans up some of the strings
adds = pd.read_csv("data/Addresses.csv", low_memory=False)

ents = pd.read_csv("data/Entities.csv", low_memory=False)
ents["name"] = ents.name.apply(normalise)

inter = pd.read_csv("data/Intermediaries.csv", low_memory=False)
inter["name"] = inter.name.apply(normalise)

offi = pd.read_csv("data/Officers.csv", low_memory=False)
offi["name"] = offi.name.apply(normalise)

edges = pd.read_csv("data/all_edges.csv", low_memory=False)

# create graph

G = nx.DiGraph()

for n,row in adds.iterrows():
    G.add_node(row.node_id, node_type="address", details=row.to_dict())
    
for n,row in ents.iterrows():
    G.add_node(row.node_id, node_type="entities", details=row.to_dict())
    
for n,row in inter.iterrows():
    G.add_node(row.node_id, node_type="intermediates", details=row.to_dict())
    
for n,row in offi.iterrows():
    G.add_node(row.node_id, node_type="officers", details=row.to_dict())
    
for n,row in edges.iterrows():
    G.add_edge(row.node_1, row.node_2, rel_type=row.rel_type, details={})

# store locally to allow faster loading
nx.write_adjlist(G,"pp_graph.adjlist")

# G = nx.read_adjlist("pp_graph.adjlist")

print(G.number_of_nodes())
print(G.number_of_edges())

get_ipython().magic('time merge_similar_names(G)')

print(G.number_of_nodes())
print(G.number_of_edges())

subgraphs = [g for g in nx.connected_component_subgraphs(G.to_undirected())]

subgraphs = sorted(subgraphs, key=lambda x: x.number_of_nodes(), reverse=True)
print([s.number_of_nodes() for s in subgraphs[:10]])

plot_graph(subgraphs[134])

plot_graph(subgraphs[206], figsize=(8,8))

# grab the largest subgraph
g = subgraphs[0]

# look at node degree
nodes = g.nodes()
g_degree = g.degree()
types = [g.node[n]["node_type"] for n in nodes]
degrees = [g_degree[n] for n in nodes]
names = [get_node_label(g.node[n]) for n in nodes]
node_degree = pd.DataFrame(data={"node_type":types, "degree":degrees, "name": names}, index=nodes)

# how many by node_type
node_degree.groupby("node_type").agg(["count", "mean", "median"])

node_degree.sort_values("degree", ascending=False)[0:15]

def plot_path(g, path):
    plot_graph(g.subgraph(path), label_edges=True)

path = nx.shortest_path(g, source=54662, target=298333)
plot_path(G, path)

plot_graph(G.subgraph(nx.ego_graph(g, 24663, radius=1).nodes()), label_edges=True)

path = nx.shortest_path(g,11011863, 298293)
plot_path(G, path)

max_bin = max(degrees)
n_bins = 20
log_bins = [10 ** ((i/n_bins) * np.log10(max_bin)) for i in range(0,n_bins)]
fig, ax = plt.subplots()
node_degree.degree.value_counts().hist(bins=log_bins,log=True)
ax.set_xscale('log')

plt.xlabel("Number of Nodes")
plt.ylabel("Number of Degrees")
plt.title("Distribution of Degree");

get_ipython().magic('time pr = nx.pagerank_scipy(g)')

node_degree["page_rank"] = node_degree.index.map(lambda x: pr[x])

node_degree.sort_values("page_rank", ascending=False)[0:15]

node_degree[node_degree.node_type == "entities"].sort_values("page_rank", ascending=False)[0:10]

t = nx.ego_graph(g, 10165699, radius=1)
plot_graph(t, label_edges=True)

get_ipython().magic('time cl = nx.clustering(g)')

node_degree["clustering_coefficient"] = node_degree.index.map(lambda x: cl[x])

node_degree.clustering_coefficient.hist()

node_degree.sort_values(["clustering_coefficient", "degree"], ascending=False)[0:10]

t = nx.ego_graph(g, 122762, radius=1)
plot_graph(G.subgraph(t), label_edges=True)

owner_rels = set({
    'shareholder of',
    'Shareholder of',
    'Director / Shareholder of',
    'Director of',
    'Director (Rami Makhlouf) of',
    'Power of Attorney of',
    'Director / Shareholder / Beneficial Owner of',
    'Member / Shareholder of',
    'Owner of',
    'Beneficial Owner of',
    'Power of attorney of',
    'Owner, director and shareholder of',
    'President - Director of',
    'Sole shareholder of',
    'President and director of',
    'Director / Beneficial Owner of',
    'Power of Attorney / Shareholder of',
    'Director and shareholder of',
    'beneficiary of',
    'President of',
    'Member of Foundation Council of',
    'Beneficial owner of',
    'Sole signatory of',
    'Sole signatory / Beneficial owner of',
    'Principal beneficiary of',
    'Protector of',
    'Beneficiary, shareholder and director of',
    'Beneficiary of',
    'Shareholder (through Julex Foundation) of',
    'First beneficiary of',
    'Authorised Person / Signatory of',
    'Successor Protector of',
    'Register of Shareholder of',
    'Reserve Director of',
    'Resident Director of',
    'Alternate Director of',
    'Nominated Person of',
    'Register of Director of',
    'Trustee of Trust of',
    'Personal Directorship of',
    'Unit Trust Register of',
    'Chairman of',
    'Board Representative of',
    'Custodian of',
    'Nominee Shareholder of',
    'Nominee Director of',
    'Nominee Protector of',
    'Nominee Investment Advisor of',
    'Nominee Trust Settlor of',
    'Nominee Beneficiary of',
    'Nominee Secretary of',
    'Nominee Beneficial Owner of'
});

# copy main graph
g2 = G.copy()

# remove non-ownership edges
for e in g2.edges(data=True):
    if e[2]["rel_type"] not in owner_rels:
        g2.remove_edge(e[0], e[1])
        
# get all subgraphs
subgraphs = [sg for sg in nx.connected_component_subgraphs(g2.to_undirected())]
subgraphs = sorted(subgraphs, key=lambda x: x.number_of_nodes(), reverse=True)
len(subgraphs)

g2.number_of_edges()

tt = subgraphs[1000].nodes()
plot_graph(g2.subgraph(tt), label_edges=True)

avg_deg = pd.Series(data=[np.median(list(sg.degree().values())) for sg in subgraphs],
                    index=range(0,len(subgraphs)))

avg_deg.sort_values(ascending=False)[0:10]

tt = subgraphs[582].nodes()
plot_graph(g2.subgraph(tt))

lp = nx.dag_longest_path(g2)
print("The longest path is {} nodes long.".format(len(lp)))
plot_graph(g2.subgraph(lp), label_edges=True)

