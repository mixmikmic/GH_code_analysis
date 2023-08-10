import os
import time
from collections import defaultdict, Counter

import pybel
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

pybel.__version__

time.asctime()

gene_orthology_url = 'http://resources.openbel.org/belframework/20150611/resource/gene-orthology.bel'

get_ipython().run_cell_magic('time', '', "# Download, parse, and cache if not already avaliable as a gpickle\npath = os.path.join(pybel.constants.PYBEL_DIR, 'gene-orthology.gpickle')\n\nif not os.path.exists(path):\n    orthology = pybel.from_url(gene_orthology_url)\n    pybel.to_pickle(orthology, path)\nelse:\n    orthology = pybel.from_pickle(path)")

named_complexes_url = 'http://resources.openbel.org/belframework/20150611/resource/named-complexes.bel'
protein_families_url = 'http://resources.openbel.org/belframework/20150611/resource/protein-families.bel'

orthology_undirected = orthology.to_undirected()

index2component = {}
member2index = {}
index2mgi = {}

for i, component in enumerate(nx.connected_components(orthology_undirected)):
    index2component[i] = component

    for function, namespace, name in component:
        member2index[function, namespace, name] = i

        if 'MGI' == namespace:
            index2mgi[i] = function, namespace, name

mapping = {}

for function, namepace, name in orthology_undirected:
    if (function, namepace, name) not in member2index:
        continue
        
    index = member2index[function, namepace, name]
    
    if index not in index2mgi:
        continue
        
    mapping[function, namepace, name] = index2mgi[index]

g = pybel.get_large_corpus()

before_counter = Counter(node[1] for node in g.nodes_iter() if g.node[node]['type'] == 'Gene')

before_df = pd.DataFrame.from_dict(before_counter, orient='index')
before_df.sort_values(0, ascending=False).plot(kind='barh')
plt.show()

# Map node data
for name, data in g.nodes_iter(data=True):
    if data['type'] in ('Gene','RNA','Protein') and name in mapping:
        g.node[name].update(orthology_undirected.node[mapping[name]])

# Map node labels
g_relabeled = nx.relabel_nodes(g, lambda n: mapping[n] if n in mapping else n, copy=False)

after_counter = Counter(node[1] for node in g_relabeled.nodes_iter() if g_relabeled.node[node]['type'] == 'Gene')

after_df = pd.DataFrame.from_dict(after_counter, orient='index')
after_df.sort_values(0, ascending=False).plot(kind='barh')
plt.show()

# cache the resulting graph as a gpickle for later
path = os.path.join(pybel.constants.PYBEL_DIR, 'large_corpus_mgi.gpickle')
pybel.to_pickle(g_relabeled, path)

