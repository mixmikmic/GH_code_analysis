# imports and plotting defaults
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.style.use('ggplot')
from copy import deepcopy

# use clustergrammer module to load/process (source code in clustergrammer directory)
from clustergrammer import Network

def plot_cl_boxplot_with_missing_data(inst_df, ylim=5):
    '''
    Make a box plot of the cell lines where the cell lines are ranked based 
    on their average PTM levels
    '''
    print(inst_df.shape)
    # get the order of the cell lines based on their mean 
    sorter = inst_df.mean().sort_values().index.tolist()
    # reorder based on ascending mean values
    sort_df = inst_df[sorter]
    # box plot of PTM values ordered based on increasing mean 
    sort_df.plot(kind='box', figsize=(10,5), rot=90, ylim=(-ylim, ylim))

filename = '../lung_cellline_3_1_16/lung_cl_all_ptm/all_ptm_ratios.tsv'
net = deepcopy(Network())
net.load_file(filename)
net.normalize(axis='row', norm_type='zscore')
tmp_df = net.dat_to_df()
inst_df = tmp_df['mat']

plot_cl_boxplot_with_missing_data(inst_df, 3)

filename = '../lung_cellline_3_1_16/lung_cl_all_ptm/all_ptm_ratios.tsv'
net = deepcopy(Network())
net.load_file(filename)
net.filter_threshold('row', threshold=0, num_occur=45)
net.normalize(axis='row', norm_type='zscore')
tmp_df = net.dat_to_df()
inst_df = tmp_df['mat']

plot_cl_boxplot_with_missing_data(inst_df, 3)



