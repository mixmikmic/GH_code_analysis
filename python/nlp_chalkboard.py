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

sys.path.append(top_directory + 'explore/vertex_metrics_experiment/code/')
from make_snapshots import *
from make_edge_df import *
from attachment_model_inference import *
from compute_ranking_metrics import *
from pipeline_helper_functions import *
from make_case_text_files import *
from bag_of_words import *

# directory set up
data_dir = top_directory + 'data/'
experiment_data_dir = data_dir + 'vertex_metrics_experiment/'

court_name = 'scotus'

# jupyter notebook settings
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

G = load_and_clean_graph(data_dir, court_name)

similarity_matrix = load_sparse_csr(filename=experiment_data_dir + 'cosine_sims.npz')

with open(experiment_data_dir + 'CLid_to_index.p', 'rb') as f:
    CLid_to_index = pickle.load(f)

def get_similarities(similarity_matrix, CLid_A, CLid_B, CLid_to_index):
    """
    Returns the similarities for cases index by CL ids as a list
    
    Parameters
    ----------
    similarity_matrix: precomputed similarity matrix
    
    CLid_A, CLid_B: two lists of CL ids whose similarities we want
    
    CLid_to_index: dict that maps CL ids to similarity_matrix indices
    """
    
    if len(CLid_A) != len(CLid_B):
        raise ValueError('lists not the same length')
    else:
        N = len(CLid_A)
    
    # list to return
    similarities = [0] * N

    # grab each entry
    for i in range(N):
        
        try:
            # convet CL id to matrix index
            idA = CLid_to_index[CLid_A[i]]
            idB = CLid_to_index[CLid_B[i]]

            similarities[i] = similarity_matrix[idA, idB]
        except KeyError:
            # if one of the CLid's is not in the similarity matrix return nan
            similarities[i] = np.nan

    return similarities

def save_similarity_matrix(experiment_data_dir, similarity_matrix, CLid_to_index):
    """
    saves similarity matrix and CLid_to_index dict
    """
    
    # save similarity matrix
    save_sparse_csr(filename=experiment_data_dir + 'cosine_sims',
                    array=S)

    # save clid to index map
    with open(experiment_data_dir + 'CLid_to_index.p', 'wb') as fp:
        pickle.dump(CLid_to_index, fp)
        
        
def load_similarity_matrix(experiment_data_dir):
    """
    Load similarity matrix and CLid_to_index dict
    
    Parameters
    ----------
    experiment_data_dir:
    
    Output
    ------
    similarity_matrix, CLid_to_index
    """
    
    similarity_matrix = load_sparse_csr(filename=experiment_data_dir + 'cosine_sims.npz')

    with open(experiment_data_dir + 'CLid_to_index.p', 'rb') as f:
        CLid_to_index = pickle.load(f) 
        
        
    return similarity_matrix, CLid_to_index

CLid_ing = []
CLid_ed = []
for e in G.es:
    
    CLid_ing.append(G.vs[e.source]['name'])
    CLid_ed.append(G.vs[e.target]['name'])

start = time.time()
sims = get_similarities(S, CLid_ing, CLid_ed, CLid_to_index)
runtime = time.time() - start

len(CLid_to_index.keys())
map_clids = CLid_to_index.keys()

print 'there are %d keys' % len(CLid_to_index.keys())

len(G.vs)

G_clids = G.vs['name']

print 'there are %d vertices in the graph' % len(G.vs)

set(G_clids).difference(set(map_clids))

len(os.listdir(experiment_data_dir + 'textfiles/'))



