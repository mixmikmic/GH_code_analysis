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


from seaborn.apionly import color_palette

from load_data import load_citation_network_igraph, case_info
from helper_functions import rankdata_reverse

from dim_reduction import *
from viz import *
from color_palettes import *



get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

data_dir = '../../data/'
court_name = 'scotus'

g_d = load_citation_network_igraph(data_dir, 'scotus', directed=True)
# g_u = load_citation_network_igraph(data_dir, 'scotus', directed=False)

case_metrics = pd.DataFrame(index=range(len(g_d.vs)))
case_metrics['id'] = g_d.vs['name']
case_metrics['year'] = g_d.vs['year']


# run on directed graph
case_metrics['authority_d'] = g_d.authority_score()
case_metrics['indegree_d'] = g_d.indegree()
case_metrics['outdegree_d'] = g_d.outdegree()
case_metrics['hubs_d'] = g_d.hub_score()
case_metrics['betweenness_d'] = g_d.betweenness()
case_metrics['eigen_d'] = g_d.eigenvector_centrality()
# case_metrics['closeness_d'] = g_d.closeness()
case_metrics['pagerank_d'] = g_d.pagerank()

# # run on undirected graph
# case_metrics['authority_u'] = g_u.authority_score()
# case_metrics['indegree_u'] = g_u.indegree()
# case_metrics['outdegree_u'] = g_u.outdegree()
# case_metrics['hubs_u'] = g_u.hub_score()
# case_metrics['betweenness_u'] = g_u.betweenness()
# case_metrics['eigen_u'] = g_u.eigenvector_centrality()
# case_metrics['closeness_u'] = g_u.closeness()
# case_metrics['pagerank_u'] = g_u.pagerank()

case_metrics

# put metrics in data frame

metrics = case_metrics.columns.tolist()[2:]
X = case_metrics[metrics]

# PCA of metrics

U, D, V = get_PCA(X, scale=True)
scores = np.dot(U, np.diag(D))

# cases colored by year

case_years = case_metrics['year'].tolist() # case years
start_year = min(case_years)
years0 = [y -  start_year for y in case_years] # case years beginning at zero

year_palette =  color_palette("PuBu", max(years0) +1 )

case_year_colors = [year_palette[y] for y in years0]

plot_scores(scores, 
            start=1,
            n_comp=5, 
            palette = case_year_colors,
            title='PCA of vertex metrics')

plt.figure(figsize=[10, 10])
plt.title('loadings plot of vertex metrics')
d = len(metrics)
for k in range(d):
    plt.plot(range(d),
             V[k],
             marker='o',
             color=color_palette("PuBu", d)[d-k-1],
             label='loading %d' % k)
    
             # alpha = 1 - (k + 0.0)/d)
plt.axhline(0, ls='--', color='red', alpha=.5)
plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))

plt.xlabel('metric')
plt.ylabel('loading')

print metrics

y = case_metrics['pagerank_d']
years = case_metrics['year']
plt.scatter(years,
            y,
            color=case_year_colors)
plt.ylim([0, max(y)])
plt.xlim([min(years), max(years)])

plot_scatter_matrix(case_metrics)



