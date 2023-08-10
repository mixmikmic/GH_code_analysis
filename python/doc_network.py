import itertools
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
get_ipython().magic('matplotlib inline')

doc = pd.read_csv('../Data(net)/doc_sample.csv')
doc_net = pd.read_csv('../Data(net)/doc_network.csv')

# Graph G1 with nodes as doctors and edges as Hospitals
G1 = nx.Graph()
G1.name = 'G1 (Node:Doctor, Edge:Hospital)'

# making a list of all hospitals for which each hospital is associated

grp_list = list(doc.groupby('DOC_ID'))
total_doc_ids = 6789
doc_hosp = []

for doc_id in range(total_doc_ids+1):
    listing = list(grp_list[doc_id][1]['HOSPITAL'])
    doc_hosp.append(listing)

#Creating Nodes

nodes_attr = []
for row in range(len(doc)):
    attr = list(zip(doc.columns[1:], doc.iloc[row,1:]))
    attr = dict(attr)
    nodes_attr.append(attr)

nodes = list(zip(doc['DOC_ID'], nodes_attr))
G1.add_nodes_from(nodes)

#modifying hospital attribute of nodes to add all the hospitals
for node in G1.nodes():
    G1.node[node]['HOSPITAL'] = doc_hosp[node]

print('Total Doctors(with repeatition): %d\nTotal Doctors(without repeatition): %d' %(len(doc), len(G1.nodes())))

G1.node[12]

#Creating Edges

for i,group in doc_net.groupby('HOSP_ID')['DOC_ID']:
    for u,v in itertools.combinations(group, 2):
        set_u = set(G1.node[u]['HOSPITAL'])
        set_v = set(G1.node[v]['HOSPITAL'])
        common_hosp = list(set_u.intersection(set_v))
        G1.add_edge(u, v, attr_dict={'HOSPITAL':common_hosp})

#Summary of Graph G1
print(nx.info(G1))

# nx.write_edgelist(G1, 'G1.edges')

random_doctor_id = np.random.randint(len(G1.nodes()))
G1.node[random_doctor_id]

# closeness centrality
closeness_centrality = nx.closeness_centrality(G1)[random_doctor_id]
#avg_distance = 1/closeness_centrality
#avg_distance

k = len(G1.edges())/len(G1.nodes())
k

larg_conn_comp = max(nx.connected_component_subgraphs(G1), key=len)
L_actual = nx.average_shortest_path_length(larg_conn_comp)
L_actual

n = len(G1.nodes())
edge_creation_prob = np.log(n)/np.log(k)
#random_larg_conn_comp = nx.fast_gnp_random_graph(n, edge_creation_prob)
#L_random = nx.average_shortest_path_length(random_larg_conn_comp)
L_random = edge_creation_prob
L_random

C_actual = nx.average_clustering(larg_conn_comp, weight='weight')
C_actual

edge_creation_prob = k/n
#random_larg_conn_comp = nx.fast_gnp_random_graph(n, edge_creation_prob)
#C_random = nx.average_clustering(random_larg_conn_comp, weight='weight')
C_random = edge_creation_prob
C_random

sw = (C_actual/L_actual)*(L_random/C_random)
sw

# top 5 doctors sorted according to their degrees
sorted(G1.degree().items(), key=lambda x:x[1], reverse=True)[:5]

degree_centrality = nx.degree_centrality(G1)
print(sorted(degree_centrality.items(), key=lambda x:x[1], reverse=True)[:5])
G1.node[17]

closeness_centrality = nx.closeness_centrality(G1)
sorted(closeness_centrality.items(), key=lambda x:x[1], reverse=True)[:5]    #top5 

betweeness_centrality = nx.betweenness_centrality(G1)
sorted(betweeness_centrality.items(), key=lambda x:x[1], reverse=True)[:5]    #top5 



