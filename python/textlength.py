repo_directory = '/Users/iaincarmichael/Dropbox/Research/law/law-net/'
data_dir = '/Users/iaincarmichael/data/courtlistener/'

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import cPickle as pickle
from collections import Counter
import pandas as pd

# graph package
import igraph as ig

# our code
sys.path.append(repo_directory + 'code/')
from setup_data_dir import setup_data_dir, make_subnetwork_directory
from pipeline.download_data import download_bulk_resource, download_master_edgelist, download_scdb
from helpful_functions import case_info
from viz import print_describe
from stats.linear_model import *

sys.path.append(repo_directory + 'vertex_metrics_experiment/code/')

from custom_vertex_metrics import *


# which network to download data for
network_name = 'scotus' # 'federal', 'ca1', etc


# some sub directories that get used
raw_dir = data_dir + 'raw/'
subnet_dir = data_dir + network_name + '/'
text_dir = subnet_dir + 'textfiles/'


# jupyter notebook settings
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

G = ig.Graph.Read_GraphML(subnet_dir + network_name +'_network.graphml')

num_words = np.array(G.vs['num_words'])

outdegrees = np.array(G.outdegree())

indegrees = G.indegree()

years = G.vs['year']

plt.figure(figsize=[12, 4])

plt.subplot(1,3,1)
plt.scatter(num_words, outdegrees)
plt.xlabel('num words')
plt.ylabel('outdegree')

plt.subplot(1,3,2)
plt.scatter(num_words, indegrees)
plt.xlabel('num words')
plt.ylabel('indegree')

plt.subplot(1,3,3)
plt.scatter(years, num_words)
plt.ylabel('year')
plt.ylabel('num words')

get_SLR(num_words, outdegrees, xlabel='num words', ylabel='outdegree')

# remove cases with extremes
out_deg_upper = np.percentile(outdegrees, 99)
out_deg_lower = np.percentile(outdegrees, 0)

num_words_upper = np.percentile(num_words, 99)
num_words_lower = np.percentile(num_words, 0)

od_to_keep = (out_deg_lower <= outdegrees) & (outdegrees <= out_deg_upper)
nw_to_keep = (num_words_lower <= num_words) & (num_words <= num_words_upper)
to_keep =  od_to_keep & nw_to_keep

# remove cases that have zero out-degree
get_SLR(num_words[to_keep], outdegrees[to_keep], xlabel='num words', ylabel='outdegree')

get_SLR(num_words, indegrees)



plt.scatter(years, num_words)

get_SLR(years, num_words)

def get_year_aggregate(years, x, fcn):
    by_year = {y: [] for y in set(years)}
    for i in range(len(years)):
        by_year[years[i]].append(x[i])
    
    year_agg_dict = {y: fcn(by_year[y]) for y in by_year.keys()}
    return pd.Series(year_agg_dict)

in_year_median = get_year_aggregate(years, indegrees, np.median)

nw_year_median = get_year_aggregate(years, num_words, np.median)

od_year_median = get_year_aggregate(years, outdegrees, np.median)

plt.figure(figsize=[8, 4])
plt.subplot(1,2,1)
plt.plot(nw_year_median.index, nw_year_median/1000, label='num words')
plt.plot(od_year_median.index, od_year_median, label='out degree')
plt.ylabel('mean')
plt.xlabel('year')
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.plot(nw_year_median.index, nw_year_median/1000, label='num words')
plt.plot(od_year_median.index, od_year_median, label='out degree')

plt.ylabel('median')
plt.xlabel('year')
plt.legend(loc='upper right')

plt.figure(figsize=[8, 8])
plt.scatter(nw_year_median.index, nw_year_median/1000,
            label='num words/1000', color='blue', marker='*')
plt.scatter(od_year_median.index, od_year_median,
            label='out degree',  color='red', marker='s')
plt.scatter(in_year_median.index, in_year_median,
            label='indegree degree',  color='green', marker='o')
plt.ylabel('median')
plt.xlabel('year')
plt.legend(loc='upper right')
plt.xlim([1800, 2017])
plt.ylim([0, 30])

plt.figure(figsize=[8, 8])
plt.plot(nw_year_median.index, nw_year_median/1000,
            label='num words/1000', color='black', marker='$n$', alpha=.7, linewidth=1, linestyle=':')
plt.plot(od_year_median.index, od_year_median,
            label='out degree',  color='black', marker='$o$', alpha=.7, linestyle=':')
plt.plot(in_year_median.index, in_year_median,
            label='indegree degree',  color='black', marker='$i$', alpha=.7, linestyle=':')
plt.ylabel('median')
plt.xlabel('year')
plt.legend(loc='upper right')
plt.xlim([1800, 2017])
plt.ylim([0, 30])

plt.figure(figsize=[6, 9])
plt.subplot(3,1,1)
plt.plot(nw_year_median.index, nw_year_median/1000,
         color='black', marker='.', linestyle=':')
plt.axvline(1953, color='black', alpha=.5)
plt.axvline(1969, color='black', alpha=.5)
plt.ylabel('median text length')
plt.xlim([1800, 2017])
plt.ylim([0, 30])

plt.subplot(3,1,2)
plt.plot(od_year_median.index, od_year_median,
         color='black', marker='.', linestyle=':')
plt.axvline(1953, color='black', alpha=.5)
plt.axvline(1969, color='black', alpha=.5)
plt.ylabel('median outdegree')
plt.xlim([1800, 2017])
plt.ylim([0, 30])

plt.subplot(3,1,3)
plt.plot(in_year_median.index, in_year_median,
         color='black', marker='.', linestyle=':')
plt.axvline(1953, color='black', alpha=.5)
plt.axvline(1969, color='black', alpha=.5)
plt.ylabel('median indegree')
plt.xlabel('year')
plt.xlim([1800, 2017])
plt.ylim([0, 30])



get_ipython().magic('pinfo plt.scatter')

import networkx as nx

Gnx = nx.read_graphml(subnet_dir + network_name +'_network.graphml')

get_ipython().run_cell_magic('time', '', 'katz = nx.katz_centrality(Gnx)')



