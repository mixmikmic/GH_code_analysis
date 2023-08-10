import sys

sys.path.append('../../code/')
import os
import json
from datetime import datetime
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

import networkx as nx

from load_data import load_citation_network, case_info
from helper_functions import *

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

data_dir = '../../data/'
court_name = 'scotus'

court_adj_mat = pd.read_csv(data_dir + 'clean/jurisdictions_adj_mat.csv', index_col='Unnamed: 0')
court_adj_mat.index = [j + '_ing' for j in court_adj_mat.index]
court_adj_mat.columns= [j + '_ed' for j in court_adj_mat.columns]

fed_appellate = ['ca' + str(i+1) for i in range(11)]
fed_appellate.append('cafc')
fed_appellate.append('cadc')

fed_appellate_ing = [j + '_ing' for j in fed_appellate]
fed_appellate_ed = [j + '_ed' for j in fed_appellate]

fed_appellate_network = court_adj_mat.loc[fed_appellate_ing, fed_appellate_ed]

fed_appellate_network

inter_juris_rankings = pd.DataFrame(columns = fed_appellate)

for court in fed_appellate:

    # grab the inter court citations for court
    citing_count = fed_appellate_network.loc[court + '_ing']
    
    # rank the courts by number of citations
    court_ranking = citing_count.sort_values(inplace=False, ascending=False).index.tolist()
    inter_juris_rankings[court] = [j.split('_')[0] for j in court_ranking]
    
inter_juris_rankings.index = [i + 1 for i in inter_juris_rankings.index]
inter_juris_rankings.index.name = 'rank'

inter_juris_rankings

import igraph as ig

court_adj_mat.as_matrix() + court_adj_mat.transpose().as_matrix()

num_jurisdictions = court_adj_mat.shape[0]

courts = [j.split('_')[0] for j in court_adj_mat.index.tolist()]






G = ig.Graph(n=num_jurisdictions)
G.vs['name'] = courts

# list of all edges
edges = get_pairs(jurisdictions)
for e in edges:
    ct1 = e[0]
    ct2 = e[1]

    weight = court_adj_mat.loc[ct1 + '_ing', ct2 + '_ed'] + court_adj_mat.loc[ct2 + '_ing', ct1 + '_ed']
    
    if weight > 0:
        v1 = G.vs.select(name=ct1)[0]
        v2 = G.vs.select(name=ct2)[0]
        
        G.add_edge(v1, v2, weight=weight)

G.summary()

visual_style = {}
visual_style["layout"] = G.layout("kk")

visual_style["vertex_size"] = 20
visual_style["vertex_label"] = G.vs["name"]
visual_style["label_size"] = .5

visual_style["edge_width"] = [1 for e in G.es]


visual_style["bbox"] = (600, 600)
visual_style["margin"] = 20


# ig.plot(G, **visual_style)



