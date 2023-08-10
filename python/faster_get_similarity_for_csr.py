top_directory = '/Users/iaincarmichael/Dropbox/Research/law/law-net/'

from __future__ import division

import os
import sys
import time
from math import *
import copy
import cPickle as pickle

# data
import numpy as np
import pandas as pd

# viz
import matplotlib.pyplot as plt


# graph
import igraph as ig


# NLP
from nltk.corpus import stopwords


# our code
sys.path.append(top_directory + 'code/')
from load_data import load_and_clean_graph, case_info
from pipeline.download_data import download_bulk_resource
from pipeline.make_clean_data import *
from viz import print_describe


sys.path.append(top_directory + 'explore/vertex_metrics_experiment/code/')
from make_snapshots import *
from make_edge_df import *
from attachment_model_inference import *
from compute_ranking_metrics import *
from pipeline_helper_functions import *
from make_case_text_files import *
from bag_of_words import *
from similarity_matrix import *

# directory set up
data_dir = top_directory + 'data/'
experiment_data_dir = data_dir + 'vertex_metrics_experiment/'

court_name = 'scotus'

# jupyter notebook settings
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

G = load_and_clean_graph(data_dir, court_name)

active_years = range(1900, 2015 + 1)

similarity_matrix, CLid_to_index = load_similarity_matrix(experiment_data_dir)

clids_ing = CLid_to_index.keys()[:5]
clids_ing.append(CLid_to_index.keys()[0])
clids_ing.append('0')
clids_ing.append(CLid_to_index.keys()[423])

clids_ed = CLid_to_index.keys()[5:10]
clids_ed.append(CLid_to_index.keys()[35])
clids_ed.append('0')
clids_ed.append(CLid_to_index.keys()[34])

Clid_pairs = zip(clids_ing, clids_ed)

# def get_similarity_index(CLid_pair, CLid_to_index):
#     """
#     Workhorse function for get_similarities
#     """
#     try:
#         ida = CLid_to_index[CLid_pair[0]]
#         idb = CLid_to_index[CLid_pair[1]]

#         return (ida, idb)
#     except KeyError:
#         return (np.nan, np.nan)
    
# def get_similarity_indices(Clid_pairs, CLid_to_index):
#     return zip(*[get_similarity_index(pair, CLid_to_index) for pair in Clid_pairs])

# this ends up being faseter for csr matrices
def get_similarity_index(clid, CLid_to_index):
    """
    Workhorse function for get_similarities
    """
    try:
        return CLid_to_index[clid]
    except KeyError:
        return np.nan
    
def get_similarity_indices(CLid_pairs, CLid_to_index):
    
    # get lists of CL ids
    CLids = zip(*CLid_pairs)
    citing_clids = list(CLids[0])
    cited_clids = list(CLids[1])

    # get similarity matrix indices
    idA = np.array([get_similarity_index(clid, CLid_to_index) for clid in citing_clids])
    idB = np.array([get_similarity_index(clid, CLid_to_index) for clid in cited_clids])

    # which indices don't have nans
    not_nan_indices = np.where(~(np.isnan(idB) | np.isnan(idA)))[0]

    similarities = np.array([np.nan]*len(idA))
    similarities[not_nan_indices] = 0.0
    
    # row indices should be smaller set
    if len(set(idA[not_nan_indices])) <= len(set(idB[not_nan_indices])):
        row_indices = idA[not_nan_indices].astype(int)
        col_indices = idB[not_nan_indices].astype(int)
    else:
        col_indices = idA[not_nan_indices].astype(int)
        row_indices = idB[not_nan_indices].astype(int)
    
    return row_indices, col_indices, similarities


def get_similarities2(similarity_matrix, CLid_pairs, CLid_to_index):
    # row/column indices of similarity matrix 
    row_indices, col_indices, similarities = get_similarity_indices(CLid_pairs, CLid_to_index)

    # the rows we want to get from the similarity matrix
    rows_to_get = list(set(row_indices))

    # get row subsetted similarity matrix
    row_subsetted_matrix = similarity_matrix[rows_to_get, :]

    # map the row indices from original matrix to row indices in row subsetting matrix
    row_indices_subseted = [np.where(rows_to_get == i)[0][0] for i in row_indices]

    # get the similarities that we actually have
    if type(row_subsetted_matrix) == np.ndarray:
        sims = row_subsetted_matrix[row_indices_subseted, col_indices]
    else:
        sims = row_subsetted_matrix.toarray()[row_indices_subseted, col_indices]

    
    # update similarities
    similarities[~np.isnan(similarities)] = sims
    
    return similarities.tolist()

def get_similarity(similarity_matrix, CLid_pair, CLid_to_index):
    """
    Workhorse function for get_similarities
    """
    try:
        ida = CLid_to_index[CLid_pair[0]]
        idb = CLid_to_index[CLid_pair[1]]

        return similarity_matrix[ida, idb]
    except KeyError:
        return np.nan


def get_similarities1(similarity_matrix, CLid_pairs, CLid_to_index):
    """
    Returns the similarities for cases index by CL ids as a list from
    precomuted similarity matrix

    Parameters
    ----------
    similarity_matrix: precomputed similarity matrix

    CLid_pair: lists of CL id pairs whose similarities we want

    CLid_to_index: dict that maps CL ids to similarity_matrix indices
    """
    return [get_similarity(similarity_matrix, pair, CLid_to_index) for pair in CLid_pairs]

time1 = 0
time2 = 0

seed = 243

R = 100

# compute ranking metrics function

# get list of test cases
test_vertices = get_test_cases(G, active_years, R, seed)

# load snapshots
snapshots_dict = load_snapshots(experiment_data_dir)

similarity_matrix, CLid_to_index = load_similarity_matrix(experiment_data_dir)



# run until we get R test cases (some cases might not have any citations)
for i in range(R):


    # randomly select a case
    test_case = test_vertices[i]

    # converted ig index to CL id
    cited_cases = get_cited_cases(G, test_case)


    # get vetex metrics in year before citing year
    snapshot_year = test_case['year'] - 1

    # grab data frame of vertex metrics for test case's snapshot
    snapshot_df = snapshots_dict['vertex_metrics_' +
                                 str(int(snapshot_year))]

    # restrict ourselves to ancestors of ing
    # case strictly before ing year
    ancentors = [v.index for v in G.vs.select(year_le=snapshot_year)]

    # all edges from ing case to previous cases
    edgelist = zip([test_case.index] * len(ancentors), ancentors)

    # get edge data function

    ed_CLids = [G.vs[edge[1]]['name'] for edge in edgelist]
    ing_CLids = [G.vs[edge[0]]['name'] for edge in edgelist]
    
    
    start = time.time()
    sims1 = get_similarities1(similarity_matrix, zip(ing_CLids, ed_CLids), CLid_to_index)
    time1 += (time.time() - start)
    
    start = time.time()
    sims2 = get_similarities2(similarity_matrix, zip(ing_CLids, ed_CLids), CLid_to_index)
    time2 += (time.time() - start)

print 'matrix time1: %d seconds ' % time1
print 'matrix time2: %d seconds ' % time2

