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

def compute_ranking_metrics_LR_group(G,
                               LogReg,
                               columns_to_use,
                               experiment_data_dir,
                               active_years,
                               R,
                               year_floor=1900,
                               seed=None,
                               print_progress=False):
    '''
    Computes the rank score metric for a given logistic regression object.

    Sample R test cases that have at least one citation. For each test case
    rank test case's ancestors then compute rank score for test cases actual
    citations.

    Parameters
    ------------
    G: network (so we can get each cases' ancestor network)

    LogReg: a logistic regression object
    (i.e. the output of fit_logistic_regression)

    columns_to_use: list of column names of edge metrics data frame that we
    should use to fit logistic regression

    path_to_vertex_metrics_folder: we will need these for prediciton

    year_interval: the year interval between each vertex metric .csv file

    R: how many cases to compute ranking metrics for

    year_floor: sample only cases after this year

    seed: random seed for selecting cases whose ancsetry to score

    Output
    -------
    The average ranking score over all R cases we tested
    '''

    # ranking scores for each test case
    test_case_rank_scores = []
    
    # get list of test cases
    test_vertices = get_test_cases(G, active_years, R, seed=seed)

    # load snapshots
    snapshots_dict = load_snapshots(experiment_data_dir)

    # mabye load the similarities
    if 'similarity' in columns_to_use:
        similarity_matrix, CLid_to_index = load_similarity_matrix(experiment_data_dir)
    else:
        similarity_matrix = None
        CLid_to_index = None
        
        
    # organize edges by ing snapshot year
    case_dict = get_test_cases_by_snapshot_dict(G, test_vertices, active_years)
    
    
    for year in case_dict.keys():
        
        # get vetex metrics in year before citing year
        snapshot_year = year - 1

        # grab data frame of vertex metrics for test case's snapshot
        snapshot_df = snapshots_dict['vertex_metrics_' +
                                     str(int(snapshot_year))]

        # build edgelist for all cases in given year
        edgelist = get_combined_edgelist(G, case_dict[year], snapshot_year)

        # grab edge data
        edge_data = get_edge_data(G, edgelist, snapshot_df, columns_to_use,
                                  similarity_matrix, CLid_to_index,
                                  edge_status=None)

            
        for test_case in case_dict[year]:

            # indices of edge_data 
            df_indices = [test_case['name'] + '_' + v['name']
                          for v in G.vs.select(year_le=snapshot_year)]

            # grab test case edges
            case_edge_data = edge_data.loc[df_indices]

            # rank ancestors
            ancestor_ranking = get_case_ranking_logreg(case_edge_data,
                                                       LogReg, columns_to_use)

            # get cited cases
            cited_cases = get_cited_cases(G, test_case)


            # compute rank score for cited cases
            score = score_ranking(cited_cases, ancestor_ranking)
            
            test_case_rank_scores.append(score)

    # return test_case_rank_scores, case_ranks, test_cases
    return test_case_rank_scores

def get_cited_cases(G, citing_vertex):
    """
    Returns the ciations of a cases whose cited year is strictly less than citing year
    
    Parameters
    ----------
    G: igraph object
    
    citing_vertex: igraph vertex
    
    Output
    ------
    list of CL ids of cited cases
    """
    
    # get neighbors first as ig index
    all_citations = G.neighbors(citing_vertex.index, mode='OUT')

    # return CL indices of cases
    # only return cited cases whose year is stictly less than citing year
    return [G.vs[ig_id]['name'] for ig_id in all_citations
            if G.vs[ig_id]['year'] < citing_vertex['year']]

def get_test_cases_by_snapshot_dict(G, test_cases, active_years):
    """
    Organizes test cases by year

    list is igraph indices
    """
    # get the citing year of each edge
    case_years = [case['year'] for case in test_cases]

    # dict that organizes edges by ing snapshot year
    case_dict = {y: [] for y in active_years}
    for i in range(len(test_cases)):
        case_dict[case_years[i]].append(test_cases[i])

    # only return years with at least one case
    return {k : case_dict[k] for k in case_dict.keys() if len(case_dict[k]) > 1}

def contact_lists(LOL):
    """
    Concatonates a list of lists
    """
    if len(LOL) > 1:
        return LOL[0] +  contact_lists(LOL[1:])
    else:
        return LOL[0]

def get_combined_edgelist(G, test_cases, snapshot_year):
    
    # build edgelist for all cases in given year
    edgelists = []
    for test_case in test_cases:

        # restrict ourselves to ancestors of ing
        # case strictly before ing year
        ancentors = [v.index for v in G.vs.select(year_le=snapshot_year)]

        # append test cases edgelist to edgelist
        edgelists.append(zip([test_case.index] * len(ancentors), ancentors))

    return contact_lists(edgelists)

columns_to_use = ['indegree', 'similarity']

R = 1000
seed_ranking = 3424

LogReg = fit_logistic_regression(experiment_data_dir, columns_to_use)

start = time.time()
compute_ranking_metrics_LR(G, LogReg, columns_to_use, experiment_data_dir,
                            active_years, R, seed=seed_ranking,print_progress=True)

print 'new function took %d seconds for %d test cases' % (time.time() - start, R)

start = time.time()
compute_ranking_metrics_LR_group(G, LogReg, columns_to_use, experiment_data_dir,
                            active_years, R, seed=seed_ranking,print_progress=True)

print 'new and improved function took %d seconds for %d test cases' % (time.time() - start, R)



