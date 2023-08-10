

top_directory = '/Users/iaincarmichael/Dropbox/Research/law/law-net/'

from __future__ import division

import os
import sys
import time
from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
import copy

# our code
sys.path.append(top_directory + 'code/')
from load_data import load_and_clean_graph, case_info

from make_snapshots import *
from make_edge_df import *
from attachment_model_inference import *
from compute_ranking_metrics import *
from pipeline_helper_functions import *


# directory set up
data_dir = top_directory + 'data/'
experiment_data_dir = top_directory + 'explore/vertex_metrics_experiment/experiment_data/'

court_name = 'scotus'

# jupyter notebook settings
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

G = load_and_clean_graph(data_dir, court_name)

year_interval = 10
snapshot_year_list = np.array([year for year in range(1760, 2021) if year % year_interval == 0])
metrics = ['indegree','pagerank']



run_transform_snaphots(experiment_data_dir)



# get all present edges
edgelist_to_add = G.get_edgelist()
num_edges = len(edgelist_to_add)

# get the citing year of each edge
ing_years = [G.vs[edge[0]]['year'] for edge in edgelist_to_add]

# map the citing year to the snapshot year
snap_ing_years = [get_snapshot_year(y, snapshot_year_list) for y in ing_years]

# dict that organizes edges by ing snapshot year
edges_by_ing_snap_year_dict = {y: [] for y in snapshot_year_list}
for i in range(num_edges):
    sn_year = snap_ing_years[i]
    edge = edgelist_to_add[i]
    
    edges_by_ing_snap_year_dict[sn_year].append(edge)






# 
edge_data = pd.DataFrame(columns=columns_to_use)
for sn_year in snapshot_year_list:
    # vertex metrcs in snapshot year
    ing_snap_df = snapshots_dict['vertex_metrics_' + str(sn_year)]
    
    # edges to add whos ing year is in the snapshot year
    edges = edges_by_ing_snap_year_dict[sn_year]
    
    sn_num_edges = len(edges)
    
    # CL ids of ed cases (indexes the snap_df rows)
    ed_CLids = [G.vs[edge[1]]['name'] for edge in edges]
    ing_CLids = [G.vs[edge[0]]['name'] for edge in edges]
    
    # ages 
    ed_year = np.array([G.vs[edge[1]]['year'] for edge in edges])
    ing_year = np.array([G.vs[edge[0]]['year'] for edge in edges])
    
    
    # get case similarities
    similarities = [0] * sn_num_edges
    for i in range(sn_num_edges):
        # similarities[i] = similarity_matrix.ix[ing_CLids[i], ed_CLids[i]]
        similarities[i] = 0
    
    
    # ed metrics in ing year 
    ed_metrics = ing_snap_df.loc[ed_CLids]
    
    # create edge data frame 
    sn_edge_data = pd.DataFrame()
    sn_edge_data['indegree'] = ed_metrics['indegree'].tolist()
    sn_edge_data['l_pagerank'] = ed_metrics['l_pagerank'].tolist()
    
    sn_edge_data['age'] = ing_year - ed_year
    sn_edge_data['similarity'] = similarities
    
    

    sn_edge_data.index = [str(edge[0]) + '_' + str(edge[1]) for edge in edges]
    sn_edge_data.index.name = 'CLids'
    
    # edge_data = pd.concat([edge_data, sn_edge_data], axis=1)
    edge_data = edge_data.append(sn_edge_data)













# load snapshot dataframes
snapshots_dict = load_snapshots(experiment_data_dir, train=True)

# similarity_matrix = pd.read_csv(experiment_data_dir + 'similarity_matrix.csv', index_col=0)
similarity_matrix = 0

# initialize edge data frame
colnames = copy.deepcopy(columns_to_use)
colnames.append('is_edge')
edge_data = pd.DataFrame(columns=colnames)

# get all present edges
present_edgelist = G.get_edgelist()

# organize edges by ing snapshot year
edges_by_ing_snap_year_dict =  get_edges_by_snapshot_dict(G, present_edgelist, snapshot_year_list)

# add present edge data
for sn_year in snapshot_year_list:
    # vertex metrcs in snapshot year
    snapshot_df = snapshots_dict['vertex_metrics_' + str(sn_year)]

    # edges to add whos ing year is in the snapshot year
    edges = edges_by_ing_snap_year_dict[sn_year]

    sn_edge_data = populate_edge_df(G, edges, snapshot_df, similarity_matrix, edge_status='present')
    edge_data.append(sn_edge_data)

edge_data

sn_edge_data





















def make_edge_df(G, experiment_data_dir, snapshot_year_list, num_non_edges_to_add, columns_to_use, seed=None):
    
    # load snapshot dataframes
    snapshots_dict = load_snapshots(experiment_data_dir, train=True)
    
    # similarity_matrix = pd.read_csv(experiment_data_dir + 'similarity_matrix.csv', index_col=0)
    similarity_matrix = 0
    
    # initialize edge data frame
    colnames = copy.deepcopy(columns_to_use)
    colnames.append('is_edge')
    edge_data = pd.DataFrame(columns=colnames)
    
    # get all present edges
    present_edgelist = G.get_edgelist()

    # organize edges by ing snapshot year
    edges_by_ing_snap_year_dict =  get_edges_by_snapshot_dict(G, present_edgelist, snapshot_year_list)
    
    # add present edge data
    for sn_year in snapshot_year_list:
        # vertex metrcs in snapshot year
        snapshot_df = snapshots_dict['vertex_metrics_' + str(sn_year)]
        
        # edges to add whos ing year is in the snapshot year
        edges = edges_by_ing_snap_year_dict[sn_year]
    
        sn_edge_data = populate_edge_df(G, edges, snapshot_df, similarity_matrix, edge_status='present')
        edge_data = edge_data.append(sn_edge_data)
        
        
    # get a sample of non-present edges
    absent_edgelist = sample_non_edges(G, year_interval, num_non_edges_to_add,
                                       seed=seed)
    
    # organize edges by ing snapshot year
    edges_by_ing_snap_year_dict =  get_edges_by_snapshot_dict(G, absent_edgelist, snapshot_year_list)
    
    # add absent edge data
    for sn_year in snapshot_year_list:
        # vertex metrcs in snapshot year
        snapshot_df = snapshots_dict['vertex_metrics_' + str(sn_year)]
        
        # edges to add whos ing year is in the snapshot year
        edges = edges_by_ing_snap_year_dict[sn_year]
    
        sn_edge_data = populate_edge_df(G, edges, snapshot_df, similarity_matrix, edge_status='absent')
        edge_data = edge_data.append(sn_edge_data)
        
        
    # edge_data.to_csv(experiment_data_dir + 'edge_data.csv')
    
    return edge_data
    





















def get_edges_by_snapshot_dict(G, edgelist, snapshot_year_list):
    """
    Organizes edges by ing snapshot year
    
    """
    
    num_edges = len(edgelist)

     # get the citing year of each edge
    ing_years = [G.vs[edge[0]]['year'] for edge in edgelist]
    
    # map the citing year to the snapshot year
    snap_ing_years = [get_snapshot_year(y, snapshot_year_list) for y in ing_years]
    
    
     # dict that organizes edges by ing snapshot year
    edges_by_ing_snap_year_dict = {y: [] for y in snapshot_year_list}
    for i in range(num_edges):
        sn_year = snap_ing_years[i]
        edge = edgelist[i]

        edges_by_ing_snap_year_dict[sn_year].append(edge)
    
    return edges_by_ing_snap_year_dict

def get_edge_data(G, edges, snapshot_df, similarity_matrix, edge_status=None):

    
    num_edges = len(edges)
    
    # CL ids of ed cases (indexes the snap_df rows)
    ed_CLids = [G.vs[edge[1]]['name'] for edge in edges]
    ing_CLids = [G.vs[edge[0]]['name'] for edge in edges]
    
    # ages 
    ed_year = np.array([G.vs[edge[1]]['year'] for edge in edges])
    ing_year = np.array([G.vs[edge[0]]['year'] for edge in edges])
    
    
    # get case similarities
    similarities = [0] * num_edges
    for i in range(num_edges):
        # similarities[i] = similarity_matrix.ix[ing_CLids[i], ed_CLids[i]]
        similarities[i] = 0
    
   
    # ed metrics in ing year 
    ed_metrics = snapshot_df.loc[ed_CLids]
    
    # create edge data frame 
    edge_data = pd.DataFrame()
    edge_data['indegree'] = ed_metrics['indegree'].tolist()
    edge_data['l_pagerank'] = ed_metrics['l_pagerank'].tolist()
    
    edge_data['age'] = ing_year - ed_year
    edge_data['similarity'] = similarities
    
    # add edge status
    if edge_status == 'present':
        is_edge = [1] *num_edges
    elif edge_status == 'absent':
        is_edge = [0] *num_edges
    else:
        # TODO: check if edge is present
        is_edge = [-999] * num_edges
    
    edge_data['is_edge'] = is_edge
    

    edge_data.index = [str(edge[0]) + '_' + str(edge[1]) for edge in edges]
    edge_data.index.name = 'CLids'
    
    return edge_data
  



columns_to_use = ['indegree', 'l_pagerank', 'age', 'similarity']
num_non_edges_to_add = 10 # len(G.es())
snapshot_year_list = np.array([year for year in range(1760, 2021) if year % 10 == 0])


make_edge_df(G, experiment_data_dir, snapshot_year_list, num_non_edges_to_add, columns_to_use, seed=None)

df = pd.read_csv(experiment_data_dir + 'edge_data.csv', index_col=0)



