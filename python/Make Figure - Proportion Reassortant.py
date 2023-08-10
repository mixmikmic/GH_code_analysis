import networkx as nx
import custom_funcs as cf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
seaborn.set_context('paper')
seaborn.set_style('white')


from tqdm import tqdm

from numba import jit, autojit
from time import time
from collections import defaultdict, Counter
from joblib import Parallel, delayed
from Bio import SeqIO
from Levenshtein import distance
from random import shuffle
from scipy.stats import percentileofscore

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('matplotlib inline')

# Read in the graph data and clean it.
G = nx.read_gpickle('20150902_all_ird Final Graph.pkl')
G = cf.clean_host_species_names(G)
G = cf.impute_reassortant_status(G)
G = cf.impute_weights(G)
G = cf.remove_zero_weighted_edges(G)

all_hosts = set([d['host_species'] for n, d in G.nodes(data=True)])
len(all_hosts)

len(G.edges())

cf.count_edges(G, attr='host_species')

# Proportion reassortant in data
data_props = cf.edge_proportion_reassortant(G, attr='host_species')
data_props

## Note: I know that this is bad practice, to have mixed variable scopes. Bear with me for a while...
excluded_hosts = ['Aquatic Bird', 'Avian', 'Bird', 'Duck', 'Environment', 'Mallard-Black Duck Hybrid', 'Sea Mammal', 
                  'Unknown', 'Waterfowl']

def null_distribution_proportion_reassortant(G, equally=False):
    G_shuffled = cf.shuffle_node_attribute_label(G, 'host_species', equally)
    props = cf.edge_proportion_reassortant(G_shuffled, 'host_species', exclusions=excluded_hosts)
    return props

# Proportion reassortant under null
start = time()
props_null = Parallel(n_jobs=-1)(delayed(null_distribution_proportion_reassortant)(G, equally=True) for i in range(100))
end = time()
print(end - start)

null = pd.DataFrame(props_null)
null.mean()

null.quantile(0.995)

data_samediff = pd.DataFrame([data_props])
data_samediff.mean()

seaborn.set_context("paper")
seaborn.set_style('white')

fig1 = plt.figure(figsize=(2,2))
ax1 = fig1.add_subplot(1,1,1)
ind = np.arange(2)
width = 0.35

ax1.bar(ind, null.mean(), width, 
        color='blue', label='Null', alpha=0.3, 
        yerr=[null.mean() - null.quantile(0.005),
              null.quantile(0.995) - null.mean(),])
        # yerr=null.std()*3)
ax1.bar(ind+width, data_samediff.mean(), width, color='blue', label='Data')
ax1.set_xticks(ind+width)
ax1.set_xticklabels(['Different', 'Same'])
ax1.set_xlabel('Hosts Pair')
ax1.set_ylabel('Proportion Reassortant')
ax1.legend()
ax1.annotate('A', ha='left', va='top', xy=(0,1), xycoords='figure fraction')
plt.subplots_adjust(left=0.25, bottom=0.2, right=0.98)
plt.savefig('figures/Proportion Reassortant.pdf')

# Compute exact stats
percentileofscore(a=null['same_attr'], score=data_samediff['same_attr'].values)

null['same_attr'].mean() - data_samediff['same_attr']

null['diff_attr'].mean() - data_samediff['diff_attr']

# Open the bird vs. mammal CSV file

host_classes = pd.read_csv('supp_data/host_groups.csv').drop_duplicates()
# I suspected there may have been duplicates, so I added the drop_duplicates() function call.
host_classes.set_index(['Country', 'Species'], inplace=True)
host_classes.ix['Canada'].ix['American Black Duck']

# Impute each node with the appropriate metadata. We will create a new node field: "host_class".
# The try/except coding pattern allows me to iteratively identify things for which we have no labels yet.

for n, d in G.nodes(data=True):
    c = d['country']
    s = d['host_species']
    try:
        hc = host_classes.ix[c].ix[s]
        habt = hc['Habitat/setting']
        host = hc['Bird, Mammal, Human']

        host_class = habt + ' ' + host
        G.node[n]['host_class'] = host_class
    except:
        print(c,s)

#### THIS IS WHERE THE CODE SHOULD GO ####
# Read in the bird surveillance data
birds = pd.read_csv('supp_data/20160210_avian_surveillance.tsv', sep='\t')
birds = birds[['Strain Name', 'Bird Behavior']]
birds['Strain Name'] = birds['Strain Name'].str.split('(').str[0]
birds.columns = ['Strain Name', 'Behavior']

# Read in the mammal surveillance data
mammals = pd.read_csv('supp_data/20160210_mammal_surveillance.tsv', sep='\t')
mammals = mammals[['Strain Name', 'Behavior']]
mammals['Strain Name'] = mammals['Strain Name'].str.split('(').str[0]

# Join the two data tables together
behavior = birds.append(mammals)
behavior.reset_index(drop=True, inplace=True)
behavior = behavior.drop_duplicates()
behavior.head()  # only show the last 5 entries

# Count the number of nodes that will be re-checked.
strains = list(behavior['Strain Name'])
counter = 0
for n, d in G.nodes(data=True):
    if n in strains:
        counter += 1
        
        
print("There are {0} strains that will be re-checked".format(counter))

# Do the actual changing of host class
num_checked = 0
num_changed = 0
for r, d in behavior.iterrows():
    strain_name = d['Strain Name']
    behavior = d['Behavior']

    if strain_name in G.nodes():
        host_class = G.node[strain_name]['host_class']
        animal_behavior = host_class.split(' ')[0]  # i.e. the "Domestic" in "Domestic Bird"
        animal_type = host_class.split(' ')[1]  # i.e. the "Bird" in "Domestic Bird"
        num_checked += 1
        if behavior != animal_behavior:  # this is where we do the overwrite.
            host_class = behavior + ' ' + animal_type
            G.node[strain_name]['host_class'] = host_class

            num_changed += 1
    
print('Number checked: {0}'.format(num_checked))
print('Number changed: {0}'.format(num_changed))

data_props = cf.edge_proportion_reassortant(G, attr='host_class')
data_props

# A function that computes the proportion reassortant across different node attributes.

def counts_reassortant_domain_graph(G, node_attr):
    """
    Computes the weighted counts of reassortant edges when going between different node attributes.
    
    Returns a "domain graph" with counts of weighted reasosrtant edges and sum weighted edges.
    """
    hg_graph = nx.DiGraph()

    for n, node_d in G.nodes(data=True):

        in_edges = G.in_edges(n, data=True)
        total_edges = len(in_edges)
        is_reassortant = node_d['reassortant']

        sk_hg = G.node[n][node_attr]
        if sk_hg not in hg_graph.nodes():
            hg_graph.add_node(sk_hg)

        for sc, _, edge_d in in_edges:
            sc_hg = G.node[sc][node_attr]
            if sc_hg not in hg_graph.nodes():
                hg_graph.add_node(sc_hg)

            if (sc_hg, sk_hg) not in hg_graph.edges():
                hg_graph.add_edge(sc_hg, sk_hg, total=edge_d['weight'], reassortant=0)
            if (sc_hg, sk_hg) in hg_graph.edges():
                hg_graph.edge[sc_hg][sk_hg]['total'] += edge_d['weight']

            if is_reassortant:
                hg_graph.edge[sc_hg][sk_hg]['reassortant'] += edge_d['weight']

    for sc, sk, d in hg_graph.edges(data=True):
        hg_graph.edge[sc][sk]['p_reassortant'] = d['reassortant'] / d['total']
        
    return hg_graph

# Compute the proportion reassortant across different host class pairs.
hg = counts_reassortant_domain_graph(G, 'host_class')
# We will ignore edges that involve "unknown" hosts.
for n in hg.nodes():
    if 'Unknown' in n:
        hg.remove_node(n)
# hg.edges(data=True)
hg.edges(data=True)

# A helper function for computing the null distribution.
def null_proportion_domain_graph_reassortant(G, node_attr, equally=False):
    
    G_shuffled = cf.shuffle_node_attribute_label(G, node_attr, equally)
    hg_graph_shuf = counts_reassortant_domain_graph(G_shuffled, node_attr)
    
    return hg_graph_shuf

# Compute the null distribution.
### CAUTION! YOU WILL HAVE TO WAIT 3 MINUTES FOR THIS TO FINISH!
start = time()
results = Parallel(n_jobs=-1)(delayed(null_proportion_domain_graph_reassortant)(G, 'host_class', equally=True) for i in range(100))
len(results)
end = time()
print(end - start)

# Summarize the proportion reassortant distribution under null.
def distr_null_p_reassortant(list_of_hg_graphs):
    hg_graph = nx.DiGraph()
    for g in tqdm(list_of_hg_graphs):
        hg_graph.add_nodes_from(g.nodes())
        for sc, sk, d in g.edges(data=True):
            if (sc, sk) not in hg_graph.edges():
                hg_graph.add_edge(sc, sk, p_reassortant=[d['p_reassortant']])
            else:
                hg_graph.edge[sc][sk]['p_reassortant'].append(d['p_reassortant'])
    return hg_graph

summaryG = distr_null_p_reassortant(results)

# Remove "unknowns" from consideration
for n, d in summaryG.nodes(data=True):
    if 'Unknown' in n:
        summaryG.remove_node(n)
summaryG.nodes(data=True)

# Grab out the "null" model statistics.
means = []  # mean of the distribution under null.
stds  = []  # standard deviation of distribution under null.
names = []  # names
# grab out the 1st, 5th, 95th and 99th percentile of null distribution
percs = dict()
for p in [0.5, 5, 95, 99.5]:
    percs[p] = []

name_map = {'Domestic Bird': 'DB',
            'Wild Bird': 'WB',
            'Human Human': 'H',
            'Domestic Mammal': 'DM',
            'Wild Mammal': 'WM'}

# Reverse name_map for convenience
key_map = {v:k for k, v in name_map.items()}

for sc, sk, d in sorted(summaryG.edges(data=True), key=lambda x:(x[0], x[1])):
    mean = np.mean(d['p_reassortant'])
    std = np.std(d['p_reassortant'])

    names.append('{0}:{1}'.format(name_map[sc], name_map[sk]))
    means.append(mean)
    stds.append(std)
    for p in [0.5, 5, 95, 99.5]:
        percs[p].append(np.percentile(d['p_reassortant'], p))

summaryG.edges(data=True)

percs[99.5]

# Compile the "data" statistics.
data = []
names_data = []
log10weights = []
log10reassort = []
log10clonal = []
for sc, sk, d in sorted(hg.edges(data=True), key=lambda x:(x[0], x[1])):
    if sc == 'Unknown' or sk == 'Unknown':
        pass
    else:
        names_data.append('{0}:{1}'.format(name_map[sc], name_map[sk]))
        data.append(d['p_reassortant'])
        log10weights.append(np.log10(d['total']))
        log10reassort.append(np.log10(d['reassortant']))
        log10clonal.append(np.log10(d['total'] - d['reassortant']))
data

# Plot data vs. null model.
fig = plt.figure(figsize=(7,3))

ind = np.arange(len(means))
width = 0.35

ax = fig.add_subplot(1,1,1)
ax.bar(ind, means, width=width, 
       color='blue', 
       label='Null', 
       #yerr=np.array(stds)*3, 
       yerr=[np.array(means) - percs[0.5],
             percs[99.5] - np.array(means)],
       alpha=0.3)
ax.bar(ind+width, data, width=width, color='blue', label='Data')
ax.set_xticks(ind+width)
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_ylabel('Proportion Reassortant')
ax.set_xlabel('Host Group Transition')

for i, label in enumerate(ax.get_xaxis().get_ticklabels()):
    if log10weights[i] > 3 or log10reassort[i] > 1:
        label.set_weight('bold')

ax2 = ax.twinx()
ax2.scatter(ind+width, log10weights, color='orange', label='Total', alpha=0.3)
ax2.scatter(ind+width, log10reassort, color='green', label='Reassortant', alpha=0.3)
ax2.set_ylabel('log10(Num. Events)')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

ax2.axhline(y=1, color='green', alpha=0.3, linestyle='--')
ax2.axhline(y=3, color='orange', alpha=0.3, linestyle='--')

ax.annotate('B', xy=(0,1), xycoords='figure fraction', va='top', ha='left')
# plt.legend()
plt.subplots_adjust(left=0.10, right=0.92, bottom=0.23)
plt.savefig('figures/Proportion Reassortant Ecological.pdf')

get_ipython().getoutput(' ls supp_data')

# Open the host COI data
host_coi = [s for s in SeqIO.parse('supp_data/sequences/host_coi_trimmed.fasta', 'fasta')]
# Modify the SeqRecord ID such that it only contains the host species name.
for s in host_coi:
    s.id = s.id.split('.')[0].replace('_', ' ')
host_coi = SeqIO.to_dict(host_coi)
host_coi

# Figure out which sequences have gaps. I suspect that the alignment quality might be compromised by those sequences.
ignored = []
for k, sequence in host_coi.items():
    if '-' in str(sequence.seq):
        ignored.append(k)
        print(sequence.seq)
        print(sequence.id)
ignored

# Impute the COI sequence based on the host species name, ignoring the sequences that have gaps in them. 
for n, d in G.nodes(data=True):    
    try:
        hs = d['host_species']
        assert hs not in ignored
        G.node[n]['coi_sequence'] = host_coi[hs]
    except:
        G.node[n]['coi_sequence'] = None

# Compute proportion under data. 

# Load the patristic distance data.
pds = nx.read_gpickle('supp_data/patristic_distances.pkl')
def proportion_reassortant_coi(G, dist_metric, pds=None):
    """
    Parameters
    - G: the network data, either shuffled or not shuffled.
    - dist_metric: (str) the distance metric.
        - hamming: uses hamming distance between 
    """
    
    assert dist_metric in ['hamming', 'evolutionary']
    
    data = dict()

    if dist_metric == 'hamming':
        bins = np.arange(0,0.3,0.05)  # note: this has been hard-coded
    if dist_metric == 'evolutionary':
        assert isinstance(pds, nx.Graph)
        bins = np.arange(0,1.4,0.1)
        
    for b in bins:
        data[b] = dict()
        data[b]['reassortant'] = 0
        data[b]['clonal'] = 0

    for sc, sk, d in G.edges(data=True):
            
        # Compute the appropriate distance metric.
        if dist_metric == 'hamming':
            sc_coi = G.node[sc]['coi_sequence']
            sk_coi = G.node[sk]['coi_sequence']
            if sc_coi and sk_coi:
                dist = distance(str(sc_coi.seq), str(sk_coi.seq)) / len(str(sc_coi.seq))
                b = bins[int(np.digitize(dist, bins)) - 1] 
                if d['edge_type'] == 'reassortant':
                    data[b]['reassortant'] += d['weight']
                else:
                    data[b]['clonal'] += d['weight']  

        if dist_metric == 'evolutionary':
            sc_host = G.node[sc]['host_species']
            sk_host = G.node[sk]['host_species']
            if pds.has_node(sc_host) and pds.has_node(sk_host):
                dist = pds[sc_host][sk_host]['pd']
                b = bins[int(np.digitize(dist, bins)) - 1]
                if d['edge_type'] == 'reassortant':
                    data[b]['reassortant'] += d['weight']
                else:
                    data[b]['clonal'] += d['weight']  
    return data

coi_data = proportion_reassortant_coi(G, dist_metric='hamming', pds=pds)

coi_data

# Temporary cell
prop_reassortant = dict()
for k, v in coi_data.items():
    try:
        prop_reassortant[k] = v['reassortant'] / (v['clonal'] + v['reassortant'])
    except:
        prop_reassortant[k] = 0
prop_reassortant

# Computes proportion under null: shuffle host COIs
def null_proportion_reassortant_coi(G, dist_metric, pds):
    if dist_metric == 'hamming':
        trait = 'coi_sequence'
    if dist_metric == 'evolutionary':
        trait = 'host_species'
    
    host_trait = [d[trait] for n, d in G.nodes(data=True)]
    shuffle(host_trait)

    G_shuffled = G.copy()
    for i, (n, d) in enumerate(G_shuffled.nodes(data=True)):
        G_shuffled.node[n][trait] = host_trait[i]    

    return proportion_reassortant_coi(G_shuffled, dist_metric, pds)    

start = time()
results = Parallel(n_jobs=-1)(delayed(null_proportion_reassortant_coi)(G, dist_metric='hamming', pds=pds) for i in range(500))
end = time()
print(end - start)

# View the results of permutation test simulation
results

# Aggregate the results into clonal and reassortant

agg = dict()
agg['clonal'] = defaultdict(list)
agg['reassortant'] = defaultdict(list)

for result in results:
    for dist, d in result.items():
        agg['clonal'][dist].append(d['clonal'])
        agg['reassortant'][dist].append(d['reassortant'])

agg['clonal']

agg['reassortant']

# We want to compute the proportion reassortant under the null.
# Iterate over the clonal. 

prop_null = defaultdict(list)

for d, cts in agg['reassortant'].items():
    for i, ct in enumerate(cts):
        try:
            prop_null[d].append(ct / (ct + agg['clonal'][d][i]))
        except:
            prop_null[d].append(np.nan)
prop_null



null_mean = pd.DataFrame(prop_null).mean()
null_mean = null_mean.reset_index()
null_mean.columns = ['Host COI Hamming Distance', 'Proportion Reassortant']
null_mean

null_std = pd.DataFrame(prop_null).std()
null_std = null_std.reset_index()
null_std.columns = ['Host COI Hamming Distance', 'Proportion Reassortant']

# Null buttom error range = 0.005
null_b = pd.DataFrame(prop_null).quantile(0.005)
null_b = null_b.reset_index()
null_b.columns = ['Host COI Hamming Distance', 'Proportion Reassortant']

# Null top error range = 0.995
null_t = pd.DataFrame(prop_null).quantile(0.995)
null_t = null_t.reset_index()
null_t.columns = ['Host COI Hamming Distance', 'Proportion Reassortant']

# Plot this data

## 1 November 2015:
## TODO: Rather than plot std*3 or std*5, plot 95% interval from the null. 
## Use the axes.errorbar API: http://matplotlib.org/api/axes_api.html
## [X] DONE



df = pd.DataFrame(coi_data).T
df['prop'] = df['reassortant'] / (df['clonal'] + df['reassortant'])

proportions = df['prop'].dropna().reset_index()
proportions.columns = ['Host COI Hamming Distance', 'Proportion Reassortant']

ixcol = 'Host COI Hamming Distance'
summary = proportions.set_index(ixcol).join(null_mean.set_index(ixcol), rsuffix='r').join(null_std.set_index(ixcol), rsuffix='s')
summary = summary.join(null_b.set_index(ixcol), rsuffix='r').join(null_t.set_index(ixcol), rsuffix='rr')
summary.columns = ['data', 'null', 'std', 'low', 'upp']
summary['low'] = summary['null'] - summary['low']
summary['upp'] = summary['upp'] - summary['null']

fig = plt.figure(figsize=(3,2))
ax = fig.add_subplot(111)
d_handle = ax.errorbar(fmt='^', 
                       x=np.array(summary.index), 
                       y=summary['null'], 
                       yerr=[summary['low'], summary['upp']], 
                       color='blue', 
                       alpha=0.3, 
                       label='Null')
summary['null'].iloc[:-1].plot(color='blue', alpha=0.3)

n_handle = ax.errorbar(fmt='^', 
                       x=np.array(summary.index), 
                       y=summary['data'], 
                       color='blue', 
                       label='Data')
summary['data'].iloc[:-1].plot(color='blue')

total_counts = df[['reassortant', 'clonal']].apply(lambda x:np.log10(x)).reset_index()
total_counts.columns = ['Host COI Hamming Distance', 'Reassortant', 'Clonal']
total_counts.replace(-np.inf, np.nan, inplace=True)
total_counts.dropna(inplace=True)
total_counts.set_index('Host COI Hamming Distance', inplace=True)

ax2 = ax.twinx()
# Plot the points version of total
c_xs = np.array(total_counts.index)
c_ys = total_counts['Clonal'] + total_counts['Reassortant']
c_handle = ax2.scatter(x=c_xs, 
                       y=c_ys, 
                       color='orange',
                       alpha=0.3)
# Plot the line
# c_ys.plot(color='olive', alpha=1, linestyle=':')

# Plot the points version
r_xs = np.array(total_counts.index)
r_ys = total_counts['Reassortant']
r_handle = ax2.scatter(x=r_xs, 
                       y=r_ys, 
                       color='green', 
                       alpha=0.3,
                       label='Reassortant')

# Plot the line version
# r_ys.plot(color='red', alpha=1, linestyle=':')

# Set axes parameters
ax.set_xlim(-0.015, 0.235)
ax.set_ylabel('Proportion Reassortant')
ax2.set_ylabel('log10(Num Events)')
ax.set_xlabel('Percentage COI Distance')
ax.annotate('C', xy=(0,1), xycoords='figure fraction', va='top', ha='left')

# Draw the horizontal lines
ax2.axhline(y=1, color='green', linestyle='--', alpha=0.3)
ax2.axhline(y=3, color='orange', linestyle='--', alpha=0.3)

# Final adjustments on the plot
plt.subplots_adjust(left=0.2, right=0.85, bottom=0.2)

# Save figure.
plt.savefig('figures/Proportion Reassortant COI.pdf')



# To get a sense of COIs occur in each bin, let's put together a list of COI distances.
from itertools import product
coi_distances = []
for (k1, s1), (k2, s2) in product(host_coi.items(), host_coi.items()):
    if k1 not in ignored and k2 not in ignored:
        s1 = str(s1.seq)
        s2 = str(s2.seq)
        d = distance(s1, s2) / len(s1)
        coi_distances.append((k1, k2, d))
    

# These are all pairwise host COI distances greater than some number.
coi_distance_df = pd.DataFrame(coi_distances)
coi_distance_df[coi_distance_df[2] >= 0.20]

host_orders = pd.read_csv('supp_data/20160214_animal_order.csv', na_values=['Unknown']).dropna()
host_orders = host_orders[['Order', 'Family', 'Species']]
host_orders['Order'] = host_orders['Order'].str.strip(' ')
host_orders['Family'] = host_orders['Family'].str.strip(' ')
host_orders.head()

# Convert the dataframe to a dictionary, where key = species, and value = order

host_order_map = dict()

for r, d in host_orders.iterrows():
    host_order_map[d['Species']] = d['Family']
    
host_order_map['Mallard']

len(set(host_orders['Order'])) ** 2

# Impute host order into the graph.

for n, d in G.nodes(data=True):
    if d['host_species'] in host_order_map.keys():
        G.node[n]['host_order'] = host_order_map[d['host_species']]
    else:
        G.node[n]['host_order'] = 'Unknown'

G.nodes(data=True)[0]

# Construct the reassortment domain graph based on "host_species"
host_orderG = counts_reassortant_domain_graph(G, 'host_order')
host_orderG.edges(data=True)

# What edges have sufficient data? 
# They should have at least 10 reassortant events, or 1000 total events.

for u, v, d in host_orderG.edges(data=True):
    if d['reassortant'] < 10 and d['total'] < 1000:
        host_orderG.remove_edge(u, v)

host_orderG.remove_node('Unknown')
host_orderG.edges(data=True)

good_host_orders = set([i[0] for i in host_orderG.edges()])
good_host_orders = good_host_orders.union([i[1] for i in host_orderG.edges()])
good_host_orders

# There are only enough observed data genome transfer between Anseriformes, Galliformes and Charadriiformes.
# Therefore, we will answer the question at this scale, rather than at the generic "mallard and gull".
# A shortcut is to re-label all of the nodes' host groups to "unknown" if they are not in one of the above.

for n, d in G.nodes(data=True):
    if d['host_order'] not in good_host_orders:
        G.node[n]['host_order'] = 'Unknown'

start = time()
null_proportion_domain_graph_reassortant(G, 'host_order', equally=True)
end = time()
print('{0} seconds'.format(end - start))



# This cell takes ~281 seconds to run.
start = time()
results = Parallel(n_jobs=-1)(delayed(null_proportion_domain_graph_reassortant)(G, 'host_order', equally=True) for i in range(100))
end = time()
print(end - start)

# # Pickle the graphs to disk, so that we don't have to run the cell above.
# import pickle as pkl

# with open('supp_data/20160214_host_species_graphs.pkl', 'wb') as f:
#     pkl.dump(results, f)

# # Load the pickled data from disk
# with open('supp_data/20160214_host_species_graphs.pkl', 'rb') as f:
#     summary_graphs = pkl.load(f)

# Compute the mean and std. dev.
# Also takes a while, so we have to pickle to disk as well.
# distr_null_p_reassortant is not parallelizable in its current form.
summaryG = distr_null_p_reassortant(results)
for u, v in summaryG.edges():
    if (u, v) not in host_orderG.edges():
        summaryG.remove_edge(u, v)



# nx.write_gpickle(summaryG, 'supp_data/20160214_host_species_summary_graph-random.pkl')

# summaryG = nx.read_gpickle('supp_data/20160214_host_species_summary_graph-random.pkl')

np.percentile(summaryG.edges(data=True)[0][2]['p_reassortant'], [2.5, 97.5])

summaryG.remove_node('Unknown')
summaryG.edges()

# Grab out the "null" model statistics.
means = []  # mean of the distribution under null.
stds  = []  # standard deviation of distribution under null.
names = []  # names
# grab out the 1st, 5th, 95th and 99th percentile of null distribution
percs = dict()
for p in [0.5, 2.5, 97.5, 99.5]:
    percs[p] = []

for sc, sk, d in sorted(summaryG.edges(data=True), key=lambda x:(x[0], x[1])):
    mean = np.mean(d['p_reassortant'])
    std = np.std(d['p_reassortant'])

    names.append('{0}:{1}'.format(sc, sk))
    means.append(mean)
    stds.append(std)
    for p in [0.5, 2.5, 97.5, 99.5]:
        percs[p].append(np.percentile(d['p_reassortant'], p))

percs[0.5]

summaryG.edges(data=True)

# Compile the "data" statistics.
data = []
names_data = []
log10weights = []
log10reassort = []
log10clonal = []
for sc, sk, d in sorted(host_orderG.edges(data=True), key=lambda x:(x[0], x[1])):
    names_data.append('{0}:{1}'.format(sc, sk))
    data.append(d['p_reassortant'])
    log10weights.append(np.log10(d['total']))
    log10reassort.append(np.log10(d['reassortant']))
    log10clonal.append(np.log10(d['total'] - d['reassortant']))
data

# Plot data vs. null model.
fig = plt.figure(figsize=(6,3))

ind = np.arange(len(means))
width = 0.35

ax = fig.add_subplot(1,1,1)
ax.bar(ind, means, width=width, 
       color='blue', 
       label='Null', 
       #yerr=np.array(stds)*3, 
       yerr=[np.array(means) - percs[0.5],
             percs[99.5] - np.array(means)],
       alpha=0.3)
ax.bar(ind+width, data, width=width, color='blue', label='Data')
ax.set_xticks(ind+width)
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_ylabel('Proportion Reassortant')
ax.set_xlabel('Bird Family Transition')

for i, label in enumerate(ax.get_xaxis().get_ticklabels()):
    if log10weights[i] >= 3 or log10reassort[i] >= 1:
        label.set_weight('bold')

ax2 = ax.twinx()
ax2.scatter(ind+width, log10weights, color='orange', label='Total', alpha=0.3)
ax2.scatter(ind+width, log10reassort, color='green', label='Reassortant', alpha=0.3)
ax2.set_ylabel('log10(Num. Events)')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax2.set_ylim(0,4)

ax2.axhline(y=1, color='green', alpha=0.3, linestyle='--')
ax2.axhline(y=3, color='orange', alpha=0.3, linestyle='--')

# ax.annotate('B', xy=(0,1), xycoords='figure fraction', va='top', ha='left')
# plt.legend()
# plt.subplots_adjust(left=0.10, right=0.92, bottom=0.23)
plt.savefig('figures/Proportion Reassortant Host Family.pdf', bbox_inches='tight')

len(data)



