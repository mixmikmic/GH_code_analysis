import sys

sys.path.append('../../code/')
import os
import json
from datetime import datetime
import time
from math import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

import igraph as ig
import networkx as nx

from load_data import load_citation_network, case_info

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

data_dir = '../../data/'
court_name = 'scotus'

case_metadata = pd.read_csv(data_dir + 'clean/case_metadata_master.csv')
edgelist = pd.read_csv(data_dir + 'clean/edgelist_master.csv')

# net_dir = data_dir + 'clean/' + court_name + '/'
# case_metadata = pd.read_csv(net_dir + 'case_metadata.csv')

# edgelist = pd.read_csv(net_dir + 'edgelist.csv')
# edgelist.drop('Unnamed: 0', inplace=True, axis=1)


start = time.time()
# create graph and add metadata
G = nx.DiGraph()
G.add_nodes_from(case_metadata.index.tolist())
nx.set_node_attributes(G, 'date', case_metadata['date'].to_dict())
for index, edge in edgelist.iterrows():
    ing = edge['citing']
    ed = edge['cited']
    G.add_edge(ing, ed)
end = time.time()

print 'pandas took %d seconds to go though %d edges using iterrows'  % (end - start, edgelist.shape[0])

# go through edglist using itertuples

start = time.time()
# create graph and add metadata
G = nx.DiGraph()
G.add_nodes_from(case_metadata.index.tolist())
nx.set_node_attributes(G, 'date', case_metadata['date'].to_dict())
for row in edgelist.itertuples():
    ing = row[1]
    ed = row[2]
    G.add_edge(ing, ed)
end = time.time()

print 'pandas took %d seconds to go though %d edges using itertuples'  % (end - start, edgelist.shape[0])

# create a dictonary that maps court listener ids to igraph ids
cl_to_ig_id = {}
cl_ids = case_metadata['id'].tolist()
for i in range(case_metadata['id'].size):
    cl_to_ig_id[cl_ids[i]] = i

start = time.time()
V = case_metadata.shape[0]

g = ig.Graph(n=V, directed=True)
g.vs['date'] = case_metadata['date'].tolist()
g.vs['name'] = case_metadata['id'].tolist()

ig_edgelist = []
missing_cases = 0
start = time.time()
# i = 1
for row in edgelist.itertuples():
#     if log(i, 2) == int(log(i, 2)):
#         print 'edge %d' % i
#     i += 1

    cl_ing = row[1]
    cl_ed = row[2]

    if (cl_ing in cl_to_ig_id.keys()) and (cl_ed in cl_to_ig_id.keys()):
        ing = cl_to_ig_id[cl_ing]
        ed = cl_to_ig_id[cl_ed]
    else:
        missing_cases += 0
    
    ig_edgelist.append((ing, ed))
intermediate = time.time()

g.add_edges(ig_edgelist)
end = time.time()

print 'itertuples took %d seconds to go through %d edges'  % (intermediate - start, edgelist.shape[0])
print 'igraph took %d seconds to add %d edges'  % (end - start, edgelist.shape[0])

start = time.time()
R = 1000
for i in range(R):
    g.vs.find(name='92891')
end = time.time()
print 'g.vs.find took %E seconds per lookup' % ((end - start)/R)

start = time.time()
R = 1000
for i in range(R):
    g.vs.select(name='92891')
end = time.time()
print 'g.vs.select took %E seconds per lookup' % ((end - start)/R)

start = time.time()
R = 1000
for i in range(R):
    cl_to_ig_id[92891]
end = time.time()
print 'pandas df lookup took %E seconds per lookup' % ((end - start)/R)

