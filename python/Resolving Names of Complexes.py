import itertools as itt
import math
import time
import os

import pybel
import networkx as nx
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

pybel.__version__

time.asctime()

url = 'http://resource.belframework.org/belframework/20150611/resource/named-complexes.bel'
path = os.path.join(pybel.constants.PYBEL_DIR, 'named-complexes.gpickle')

if not os.path.exists(path):
    complexes = pybel.from_url(url)
    pybel.to_pickle(complexes, path)
else:
    complexes = pybel.from_pickle(path)

plt.title('Size Rank Distribution of Named Complexes')
plt.xlabel('Rank')
plt.ylabel('Size')
plt.plot(sorted([len(complexes.adj[n]) for n in complexes.nodes_iter(type='Complex')], reverse=True), '-r')
plt.show()

# build mapping
t2n = {}
relabel_mapping = {}
for n, d in complexes.nodes_iter(type='Complex', data=True):
    members = complexes.edge[n]
    members = [member for member in members if complexes.node[member]['namespace'] == 'HGNC']
    t = ('Complex',) + tuple(sorted(members))
    t2n[t] = d
    relabel_mapping[t] = n

test_bel = """SET DOCUMENT Name = "PyBEL Test Document"
SET DOCUMENT Description = "Made for testing PyBEL parsing"
SET DOCUMENT Version = "1.6"
SET DOCUMENT Copyright = "Copyright (c) Charles Tapley Hoyt. All Rights Reserved."
SET DOCUMENT Authors = "Charles Tapley Hoyt"
SET DOCUMENT Licenses = "Other / Proprietary"
SET DOCUMENT ContactInfo = "charles.hoyt@scai.fraunhofer.de"
DEFINE NAMESPACE ChEBI AS URL "http://resource.belframework.org/belframework/1.0/namespace/chebi-names.belns"
DEFINE NAMESPACE HGNC AS URL "http://resource.belframework.org/belframework/1.0/namespace/hgnc-approved-symbols.belns"
SET Citation = {"Test Source", "Test Title", "123456"}
SET Evidence = "Evidence 1"
complex(p(HGNC:HUS1), p(HGNC:RAD1), p(HGNC:RAD9A)) increases a(ChEBI:"oxygen radical")
complex(p(HGNC:SFN), p(HGNC:YWHAB)) cnc a(ChEBI:"oxygen radical")"""

g = pybel.BELGraph(test_bel.split('\n'))

plt.axis('off')
pos = nx.spring_layout(g, iterations=1000)
nx.draw_networkx(g, 
                 pos=pos, 
                 with_labels=True, 
                 nodes=5, 
                 labels={n:('{}'.format(n[2]) if len(n) == 3 else n) for n in g}, 
                 font_size=9)
plt.show()

# fix node data
for com in g.nodes_iter(type="Complex"):
    if com in t2n:
        g.node[com].update(t2n[com]) 
        
# fix node names
gr = nx.relabel_nodes(g, lambda n: relabel_mapping[n] if n in relabel_mapping else n, copy=False)

plt.axis('off')
pos = nx.spring_layout(gr, iterations=1000)
nx.draw_networkx(gr, 
                 pos=pos, 
                 with_labels=True, 
                 nodes=5, 
                 labels={n:('{}'.format(n[2]) if len(n) == 3 else n) for n in g}, 
                 font_size=9)
plt.show()

