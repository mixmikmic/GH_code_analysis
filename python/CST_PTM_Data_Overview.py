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

# load data data and export as pandas dataframe: inst_df
def load_data(filename):
    ''' 
    load data using clustergrammer and export as pandas dataframe
    '''
    net = deepcopy(Network())
    net.load_file(filename)
    tmp_df = net.dat_to_df()
    inst_df = tmp_df['mat']

    
    # simplify column names (remove categories)
    col_names = inst_df.columns.tolist()
#     simple_col_names = []
#     for inst_name in col_names:
#         simple_col_names.append(inst_name[0])

    inst_df.columns = col_names

    print(inst_df.shape)
    
    ini_rows = inst_df.index.tolist()
    unique_rows = list(set(ini_rows))
    
    if len(ini_rows) > len(unique_rows):
        print('found duplicate PTMs')
    else:
        print('did not find duplicate PTMs')
    
    return inst_df

filename = '../lung_cellline_3_1_16/lung_cellline_phospho/' + 'lung_cellline_TMT_phospho_combined_ratios.tsv'
inst_df = load_data(filename)

inst_df.count().sort_values().plot(kind='bar', figsize=(10,2))
print(type(inst_df))

def plot_cl_boxplot_with_missing_data(inst_df):
    '''
    Make a box plot of the cell lines where the cell lines are ranked based 
    on their average PTM levels
    '''
    
    # get the order of the cell lines based on their mean 
    sorter = inst_df.mean().sort_values().index.tolist()
    # reorder based on ascending mean values
    sort_df = inst_df[sorter]
    # box plot of PTM values ordered based on increasing mean 
    sort_df.plot(kind='box', figsize=(10,3), rot=90, ylim=(-8,8))

plot_cl_boxplot_with_missing_data(inst_df)

def plot_cl_boxplot_no_missing_data(inst_df):
    # get the order of the cell lines based on their mean 
    sorter = inst_df.mean().sort_values().index.tolist()
    # reorder based on ascending mean values
    sort_df = inst_df[sorter]

    # transpose to get PTMs as columns 
    tmp_df = sort_df.transpose()

    # keep only PTMs that are measured in all cell lines
    ptm_num_meas = tmp_df.count()
    ptm_all_meas = ptm_num_meas[ptm_num_meas == 45]
    ptm_all_meas = ptm_all_meas.index.tolist()

    print('There are ' + str(len(ptm_all_meas)) + ' PTMs measured in all cell lines')
    
    # only keep ptms that are measured in all cell lines 
    # I will call this full_df as in no missing measurements
    full_df = tmp_df[ptm_all_meas]

    # transpose back to PTMs as rows
    full_df = full_df.transpose()

    full_df.plot(kind='box', figsize=(10,3), rot=90, ylim=(-8,8))
    num_ptm_all_meas = len(ptm_all_meas)

plot_cl_boxplot_no_missing_data(inst_df)

filename = '../lung_cellline_3_1_16/lung_cellline_Ack/' +     'lung_cellline_TMT_Ack_combined_ratios.tsv'
inst_df = load_data(filename)

inst_df.count().sort_values().plot(kind='bar', figsize=(10,2))

plot_cl_boxplot_with_missing_data(inst_df)

plot_cl_boxplot_no_missing_data(inst_df)

filename = '../lung_cellline_3_1_16/lung_cellline_Rme1/' + 'lung_cellline_TMT_Rme1_combined_ratios.tsv'
inst_df = load_data(filename)

inst_df.count().sort_values().plot(kind='bar', figsize=(10,2))

plot_cl_boxplot_with_missing_data(inst_df)

plot_cl_boxplot_no_missing_data(inst_df)

filename = '../lung_cellline_3_1_16/lung_cellline_Kme1/' + 'lung_cellline_TMT_Kme1_combined_ratios.tsv'
inst_df = load_data(filename)

inst_df.count().sort_values().plot(kind='bar', figsize=(10,2))

plot_cl_boxplot_with_missing_data(inst_df)

plot_cl_boxplot_no_missing_data(inst_df)



