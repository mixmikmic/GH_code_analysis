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

import igraph as ig

from load_data import load_citation_network_igraph, case_info

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

data_dir = '../../data/'
court_name = 'scotus'

# this will be a little slow the first time you run it
G = load_citation_network_igraph(data_dir, court_name)

print 'loaded %s network with %d cases and %d edges' % (court_name, len(G.vs), len(G.es))

desired_num_samples = 1000

all_indices = range(len(G.vs))

nonexistant_edge_list = []

start_time = time.time()
while len(nonexistant_edge_list) < desired_num_samples:
    # randomly select a pair of vertices
    rand_pair = np.random.choice(all_indices, size=2, replace=False)
    
    # check if there is currently an edge between the two vertices
    edge_check = G.es.select(_between=([rand_pair[0]], [rand_pair[1]]))
    
    # if edge does not exist add it to the list
    if len(edge_check) == 0: 
       
        # order the vertices by time
        if G.vs[rand_pair[0]]['year'] <= G.vs[rand_pair[1]]['year']:
            ing_id = rand_pair[1]
            ed_id = rand_pair[0]
        else:
            ing_id = rand_pair[0]
            ed_id = rand_pair[1]
            
        nonexistant_edge_list.append((ing_id, ed_id))
total_runtime = time.time() - start_time

print 'total_runtime %1.5f' % (total_runtime/desired_num_samples)

print 'len nonexistant_edge_list %d' % len(nonexistant_edge_list)

print 'estimated time to get to 500000 samples: %1.5f min' % (((total_runtime/desired_num_samples) * 500000)/60)



