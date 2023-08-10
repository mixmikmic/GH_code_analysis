# import some useful packages

import numpy as np
import pandas as pd
import networkx as nx
import community
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

dataDE = pd.read_csv('DE_data/example_DE_data.tsv',sep='\t')
print(dataDE.head())

# genes in dataDE
gene_list = list(dataDE['IDENTIFIER'])

# only use the average fold-change (because there are multiple entries for some genes
dataDE_mean = dataDE.DiffExp.groupby(dataDE['IDENTIFIER']).mean()

# load the gene-gene interactions (from genemania)
filename = 'DE_data/DE_experiment_interactions.txt'

DE_network = pd.read_csv(filename, sep='\t', header=6)
DE_network.columns = ['Entity 1','Entity 2', 'Weight','Network_group','Networks']

# create the graph, and add some edges (and nodes)
G_DE = nx.Graph()
idxCE = DE_network['Network_group']=='Co-expression'
edge_list = zip(list(DE_network['Entity 1'][idxCE]),list(DE_network['Entity 2'][idxCE]))
G_DE.add_edges_from(edge_list)

print('number of edges = ' + str(len(G_DE.edges())))
print('number of nodes = '+ str(len(G_DE.nodes())))

# create version with weighted edges
G_DE_w = nx.Graph()
edge_list_w = zip(list(DE_network['Entity 1']),list(DE_network['Entity 2']),list(DE_network['Weight']))
    
G_DE_w.add_weighted_edges_from(edge_list_w)

import imp
import plot_network
imp.reload(plot_network)

from IPython.html.widgets import interact
from IPython.html import widgets
import matplotlib.colorbar as cb
import seaborn as sns
import community

# import network plotting module
from plot_network import *

# temporary graph variable
Gtest = nx.Graph()

# check whether you have differential expression data
diff_exp_analysis=True

# replace G_DE_w with G_DE in these two lines if unweighted version is desired
Gtest.add_nodes_from(G_DE_w.nodes())  
Gtest.add_edges_from(G_DE_w.edges(data=True))

# prep border colors
nodes = Gtest.nodes()
#gene_list = gene_list

if diff_exp_analysis:
    diff_exp = dataDE_mean
    genes_intersect = np.intersect1d(gene_list,nodes)
    border_cols = Series(index=nodes)

    for i in genes_intersect:
        if diff_exp[i]=='Unmeasured':
            border_cols[i] = np.nan
        else:
            border_cols[i] = diff_exp[i] 
else:  # if no differential expression data
    border_cols = [None]


numnodes = len(Gtest)

# make these three global to feed into widget
global Gtest

global boder_cols  

global DE_network

def plot_network_shell(focal_node_name,edge_thresh=.5,network_algo='spl', map_degree=True,
                       plot_border_col=False, draw_shortest_paths=True,
                       coexpression=True, colocalization=True, other=False,physical_interactions=False,
                       predicted_interactions=False,shared_protein_domain=False):
    
    # this is the main plotting function, called from plot_network module
    fig = plot_network(Gtest, border_cols, DE_network,
                 focal_node_name, edge_thresh, network_algo, map_degree, plot_border_col, draw_shortest_paths,
                 coexpression, colocalization, other, physical_interactions, predicted_interactions, shared_protein_domain)


    return fig

# threshold slider parameters
min_thresh = np.min(DE_network['Weight'])
max_thresh = np.max(DE_network['Weight']/10)
thresh_step = (max_thresh-min_thresh)/1000.0

interact(plot_network_shell, focal_node_name=list(np.sort(nodes)),
         edge_thresh=widgets.FloatSliderWidget(min=min_thresh,max=max_thresh,step=thresh_step,value=min_thresh,description='edge threshold'),
         network_algo = ['community','clustering_coefficient','pagerank','spl']);









