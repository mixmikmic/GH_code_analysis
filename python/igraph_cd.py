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

from load_data import load_citation_network, case_info

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

data_dir = '../../data/'
court_name = 'scotus'

start = time.time()
if court_name == 'all':
    case_metadata = pd.read_csv(data_dir + 'clean/case_metadata_master.csv')

    edgelist = pd.read_csv(data_dir + 'clean/edgelist_master.csv')
else:
    net_dir = data_dir + 'clean/' + court_name + '/'
    if not os.path.exists(net_dir):
        os.makedirs(net_dir)
        make_court_subnetwork(court_name, data_dir)

    case_metadata = pd.read_csv(net_dir + 'case_metadata.csv')

    edgelist = pd.read_csv(net_dir + 'edgelist.csv')
    edgelist.drop('Unnamed: 0', inplace=True, axis=1)

# create a dictonary that maps court listener ids to igraph ids
cl_to_ig_id = {}
cl_ids = case_metadata['id'].tolist()
for i in range(case_metadata['id'].size):
    cl_to_ig_id[cl_ids[i]] = i

# add nodes
V = case_metadata.shape[0]
g = ig.Graph(n=V, directed=True)
# g.vs['date'] = case_metadata['date'].tolist()
g.vs['name'] = case_metadata['id'].tolist()

# create igraph edgelist
cases_w_metadata = set(cl_to_ig_id.keys())
ig_edgelist = []
missing_cases = 0
start = time.time()
for row in edgelist.itertuples():

    cl_ing = row[1]
    cl_ed = row[2]

    if (cl_ing in cases_w_metadata) and (cl_ed in cases_w_metadata):
        ing = cl_to_ig_id[cl_ing]
        ed = cl_to_ig_id[cl_ed]
    else:
        missing_cases += 0
    
    ig_edgelist.append((ing, ed))

# add edges to graph
g.add_edges(ig_edgelist)


# add vertex attributes
g.vs['court'] =  case_metadata['court'].tolist()
g.vs['year'] = [int(d.split('-')[0]) for d in case_metadata['date'].tolist()]

end = time.time()

print '%d seconds for %d edges' % (end - start, len(g.es))

g.summary()

# make graph undirected
gu = g.copy().as_undirected() 

start = time.time()
mod_clusters = gu.community_fastgreedy().as_clustering()
end = time.time()
print 'fastgreedy modularity took %d seconds with %d nodes and %d edges' % (end-start, len(g.vs), len(g.es))

mod_cl_sizes = mod_clusters.sizes()
{s: mod_cl_sizes.count(s) for s in set(mod_cl_sizes)}

# start = time.time()
# walktrap = gu.community_walktrap(steps=4)
# end = time.time()
# print 'walktrap took %d seconds with %d nodes and %d edges' % (end-start, len(g.vs), len(g.es))

# walktrap_clusters = walktrap.as_clustering()

walktrap_cl_sizes = walktrap_clusters.sizes()
{s: walktrap_cl_sizes.count(s) for s in set(walktrap_cl_sizes)}

mod_cl_sizes = mod_clusters.sizes()
{s: mod_cl_sizes.count(s) for s in set(mod_cl_sizes)}

x = 2

x



