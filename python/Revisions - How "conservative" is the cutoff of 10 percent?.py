import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

# Read the graphs into memory
finalG = nx.read_gpickle('20150902_all_ird Final Graph.pkl')
fullG = nx.read_gpickle('20150902_all_ird Full Complement Graph.pkl')

# Get the cut-off value for re-evaluation of clonal descent edges
weights = [d['pwi'] for _, _, d in fullG.edges(data=True)]
cutoff = np.percentile(weights, 10)
cutoff

# Identify the sink nodes that are associated with edges that are below the cutoff.

sinks = dict()
for sc, sk, d in fullG.edges(data=True):
    if d['pwi'] < cutoff:
        sinks[sk] = d['pwi']
        
        
len(sinks)

# How many of those sinks were identified as reassortant?
# First, identify the reassortant viruses in the finalG
reassortants = dict()
for sc, sk, d in finalG.edges(data=True):
    if d['edge_type'] == 'reassortant':
        reassortants[sk] = d['pwi']

# How many sinks were re-identified as reassortant viruses instead?
len(set(reassortants.keys()).intersection(sinks.keys()))

# What is the distribution of PWI improvements?
improvements = dict()
for k, d in sinks.items():
    if k in reassortants.keys():
        improvements[k] = reassortants[k] - d

# Here's the histogram
plt.hist(list(improvements.values()))

# Min, median, max of PWI improvements
np.percentile(list(improvements.values()), [0, 50, 100])



