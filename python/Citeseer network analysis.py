import networkx as nx
import numpy as np
import pickle as p
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

data_loc = './../data/raw/citeseer/'  # 'citeseer.cites', 'citeseer.content'

graph_file = open(data_loc+'citeseer.cites', 'r')

for _ in range(5): print(repr(graph_file.readline()))

graph_file.seek(0)
iid = {}  # Integer id conversion dict
idx = 0
citeseer_edgelist = []
for line in graph_file.readlines():
    i, j = line.split()
    if i not in iid:
        iid[i] = idx
        idx += 1
    if j not in iid:
        iid[j] = idx
        idx += 1
    citeseer_edgelist.append((iid[j],iid[i]))  # Correct direction of links

print("Number of edges:", len(citeseer_edgelist))

citeseer = nx.DiGraph(citeseer_edgelist)

print("Number of nodes:", len(citeseer))

len(iid)

graph_file.close()

# Prepare data arrays and labels lookup table
citeseer_labels = np.ndarray(shape=(len(iid)), dtype=int)
citeseer_features = np.ndarray(shape=(len(iid), 3703), dtype=int)
labels = {'Agents': 0, 'AI': 1, 'DB': 2, 'IR': 3, 'ML': 4, 'HCI': 5}
no_labels = set(citeseer.nodes())

# Read data
with open(data_loc+'citeseer.content', 'r') as f:
    for line in f.readlines():
        oid, *data, label = line.split()
        citeseer_labels[iid[oid]] = labels[label]
        citeseer_features[iid[oid],:] = list(map(int, data))
        no_labels.remove(iid[oid])
        
for i in no_labels:
    citeseer_labels[i] = -1
    citeseer_features[i,:] = np.zeros(3703)
    
# Validation
with open(data_loc+'citeseer.content', 'r') as f:
    for line in f.readlines():
        oid, *data, label = line.split()
        assert citeseer_labels[iid[oid]] == labels[label]
        assert citeseer_labels[iid[oid]] < 6
        assert sum(citeseer_features[iid[oid]]) == sum(map(int, data))
    print("Validation for `citeseer_labels` and `citeseer_features` passes.")
        

print("Feature shape: ", citeseer_features.shape)
print("Label shape: ", citeseer_labels.shape)

from scipy.sparse import csr_matrix
citeseer_csr_features = csr_matrix(citeseer_features)
citeseer_dataset = {'NXGraph': citeseer, 'Labels': citeseer_labels, 
                    'CSRFeatures': citeseer_csr_features}
with open('./../data/citeseer.data', 'wb') as f:
    p.dump(citeseer_dataset, f, protocol=2)

nx.write_edgelist(citeseer, path='./../data/citeseer.edges')  # delimiter is a white space

len(citeseer_labels)

max(citeseer_labels)

np.unique(citeseer_labels)

no_labels



