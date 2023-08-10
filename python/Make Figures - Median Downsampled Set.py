import networkx as nx
import figures as fg
import custom_funcs as cf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from collections import defaultdict

get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

seaborn.set_context("paper")
seaborn.set_style('white')

# Load the graphs into memory.
Gs = []

for i in range(100):
    g = nx.read_gpickle('md_final_graphs/20150902_all_ird Final Graph {0}.pkl'.format(i))
    g = cf.impute_reassortant_status(g)
    g = cf.impute_weights(g)
    g = cf.remove_zero_weighted_edges(g)  
    Gs.append(g)

len(Gs)

# Compile the total counts of number of nodes in each graph.
# i.e. plot representation

h_subtype_data = defaultdict(list)
c_subtype_data = defaultdict(list)
s_subtype_data = defaultdict(list)
for G in Gs:
    # Human-human, chicken-chicken, and swine-swine
    hh_nodes = fg.same_host_descent(G, 'Human')
    cc_nodes = fg.same_host_descent(G, 'Chicken')
    ss_nodes = fg.same_host_descent(G, 'Swine')
    
    hh_subtypes = fg.subtype_counts(hh_nodes, G, log=True)
    ss_subtypes = fg.subtype_counts(ss_nodes, G, log=True)
    cc_subtypes = fg.subtype_counts(cc_nodes, G, log=True)
    
    def add_to_compiled(subtype_data, subtypes):
        for subtype, counts in subtypes.items():
            subtype_data[subtype].append(counts)
            
    add_to_compiled(h_subtype_data, hh_subtypes)
    add_to_compiled(s_subtype_data, ss_subtypes)
    add_to_compiled(c_subtype_data, cc_subtypes)

s_subtype_data

# Plot the data
def summarized_data(subtype_data):
    means = dict()
    stds = dict()
    for k, v in subtype_data.items():
        means[k] = np.mean(v)
        stds[k] = np.std(v)
    return means, stds

h_means, h_stds = summarized_data(h_subtype_data)
# c_means, c_stds = summarized_data(c_subtype_data)
s_means, s_stds = summarized_data(s_subtype_data)

fig = plt.figure(figsize=(2,6))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

def plot_subtype_counts(means, stds, ax, color='blue'):
    data = pd.DataFrame()
    data['means'] = pd.Series(means)
    data['stds'] = pd.Series(stds)

    data['means'].plot(kind='bar', ax=ax, yerr=data['stds'], color=color)

plot_subtype_counts(h_means, h_stds, ax1, color='blue')
# plot_subtype_counts(c_means, c_stds, ax2)
plot_subtype_counts(s_means, s_stds, ax3, color='red')

data_all_props = defaultdict(list)
null_all_props = defaultdict(list)


for G in Gs:
    G = cf.clean_host_species_names(G)
    data_props = cf.edge_proportion_reassortant(G, attr='host_species', exclusions=['Unknown'])
    null_props = fg.null_distribution_proportion_reassortant(G, equally=True)
    
    print(data_props, null_props)
    # Append the data to the compilation
    for k, v in data_props.items():
        data_all_props[k].append(v)
        
    for k, v in null_props.items():
        null_all_props[k].append(v)

def summarize_prop_reassortant(all_props):
    means = dict()
    stds = dict()
    
    for k, v in all_props.items():
        means[k] = np.mean(v)
        stds[k] = np.std(v)

    return means, stds

data_prop_mean, data_prop_std = summarize_prop_reassortant(data_all_props)
null_prop_mean, null_prop_std = summarize_prop_reassortant(null_all_props)

pd.Series(null_all_props)

means = pd.DataFrame()
means['data'] = pd.Series(data_prop_mean)
means['null'] = pd.Series(null_prop_mean)

stds = pd.DataFrame()
stds['data'] = pd.Series(data_prop_std)
stds['null'] = pd.Series(null_prop_std)

means.plot(kind='bar', yerr=stds)



