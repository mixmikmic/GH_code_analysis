import networkx as nx
from graph_tool.all import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

with open('citeseer.data', 'rb') as f:
    citeseer = pickle.load(f)

citeseer.keys()

from motifwalk.utils.Graph import GraphContainer

from motifwalk.utils import find_meta, set_dataloc, get_metadata

set_dataloc(path_to_data=os.path.abspath('./'))

metadata = get_metadata()

citeseer_meta = find_meta('citeseer')

citeseer_pack = GraphContainer(citeseer_meta, dataloc=os.path.abspath('./'))

citeseer_pack.get_labels()

citeseer_gt = citeseer_pack.get_gt_graph()

graph_draw(citeseer_gt, vertex_fill_color=label, output="citeseer_with_labels.pdf")

citeseer_nx = citeseer_pack.get_graph()

label = citeseer_gt.new_vertex_property("int")

classes = np.argmax(citeseer_pack.get_labels(), axis=1)

for i in range(classes.size):
    label[i] = classes[i]

bc_meta = find_meta('blogcatalog')

bc_pack = GraphContainer(bc_meta, dataloc=os.path.abspath('./'))

bc_gt = bc_pack.get_gt_graph()

label = bc_gt.new_vertex_property("int")

classes = np.argmax(bc_pack.get_labels().toarray(), axis=1)

for i in range(classes.size):
    label[i] = classes[i]

from time import time
t = time()
graph_draw(bc_gt, vertex_fill_color=label, output="blogcatalog_with_labels.pdf", output_size=(1200, 1200))
print(time()-t)

cora_meta = find_meta('cora')

cora_pack = GraphContainer(metadata=cora_meta, dataloc=os.path.abspath('./'))

labels = cora_pack.get_labels()

cora_gt = cora_pack.get_gt_graph()

classes = np.argmax(cora_pack.get_labels(), axis=1)

label = cora_gt.new_vertex_property("int")

for i in range(classes.size):
    label[i] = classes[i]

graph_draw(cora_gt, vertex_fill_color=label, output="cora_with_labels.pdf")

aloc = "/home/gear/Dropbox/CompletedProjects/motifwalk/data/raw/amazon_copurchasing"

with open(aloc+'/com-amazon.top5000.cmty.txt') as f:
    top5k = f.read()

top5k = top5k.split('\n')

top5klist = [i.split('\t') for i in top5k]

len(max(top5klist, key=len))

top5klist[0]

amazon_nx = nx.read_edgelist('./raw/amazon_copurchasing/com-amazon.ungraph.txt')

amazon_nx.is_directed()

amazon_nx.size()

len(amazon_nx.nodes())

sorted_nodes_amazon = sorted(amazon_nx.nodes(), key=int)

map_amazon = {}
for i, node_id in enumerate(sorted_nodes_amazon):
    map_amazon[node_id] = i

len(map_amazon)

max(amazon_nx.nodes(), key=int)

map_amazon['548551']

amazon_nx

def amazon_type_map(s):
    return map_amazon[s]

amazon_nx = nx.read_edgelist('./raw/amazon_copurchasing/com-amazon.ungraph.txt', nodetype=amazon_type_map)

amazon_nx[0]

max(amazon_nx.nodes())

from scipy.sparse import csr_matrix

label_amazon = np.zeros(shape=(334863, 5000), dtype=np.int8)

for cmty, nodelist in enumerate(top5klist[:-1]):
    for node in nodelist:
        label_amazon[map_amazon[node]][cmty] = 1 

label_amazon = csr_matrix(label_amazon, dtype=np.int8)

label_amazon

label_amazon[4]

map_amazon['164985']

label_amazon[100150]

np.nonzero(label_amazon[100150])

amazon = {}

amazon['Labels'] = label_amazon

amazon['NXGraph'] = amazon_nx

with open('amazon.data', 'wb') as f:
    pickle.dump(amazon, f)

amazon_meta = find_meta('amazon')

amazon_pack = GraphContainer(metadata=amazon_meta, dataloc=os.path.abspath('./'))

amazon_gt = amazon_pack.get_gt_graph()

amazon_gt

get_ipython().magic('time graph_draw(amazon_gt, output="amazon_graph.pdf", output_size=(1200, 1200))')

with open('/home/gear/Dropbox/CompletedProjects/motifwalk/data/raw/pubmed/Pubmed-Diabetes.DIRECTED.cites.tab') as f:
    pubmed = f.read()

pubmed[0:100]

edges = pubmed.split('\n')

edges = edges[2:]

edges[0]

pubmed_graph = nx.DiGraph()

tuples = []

edges[-1]

ix, src, _ ,dst = edges[-2].split('\t')

edges[-2]

ix

src

dst

src_id = src.split(':')[-1]

src_id

for e in edges[:-1]:
    idx, src, _, dst = e.split('\t')
    src_id = src.split(':')[-1]
    dst_id = dst.split(':')[-1]
    tuples.append((src_id, dst_id))

len(tuples)

pubmed_graph.add_edges_from(tuples)

pubmed_graph.is_directed()

with open('/home/gear/Dropbox/CompletedProjects/motifwalk/data/raw/pubmed/Pubmed-Diabetes.NODE.paper.tab') as f:
    pubmed = f.read()

data = pubmed.split('\n')

data[0]

data[2]

template = data[1].split('\t')

template

kw_id_map = {}

i = 0
for words in template[1:-1]:
    _, word, _ = words.split(':')
    kw_id_map[word] = i
    i += 1

kw_id_map['w-use']

pubmed_graph.nodes()[:10]

all_pubmed_nodes = sorted(pubmed_graph.nodes(), key=int)

all_pubmed_nodes[:5]

all_pubmed_nodes[-1]

len(all_pubmed_nodes)

map_pubmed = {}
for i, node_id in enumerate(all_pubmed_nodes):
    map_pubmed[node_id] = i

map_pubmed['29094']

def pubmed_type(node_id):
    return map_pubmed[node_id]

len(kw_id_map)

len(map_pubmed)

pubmed_features = np.zeros(shape=(19717, 500), dtype=np.float32)

pubmed_labels = np.zeros(shape=(19717, 3), dtype=np.uint8)

data[-1]

test = data[2]

node_id, *features_vec, _ = test.split('\t')

node_id

features_vec

for d in data[2:-1]:
    node_id, label, *feature_vec, summary = d.split('\t')
    int_id = map_pubmed[node_id]
    label = int(label.split('=')[-1]) - 1
    pubmed_labels[int_id][label] = 1
    for f in feature_vec:
        word, val = f.split('=')
        feature_id = kw_id_map[word]
        pubmed_features[int_id][feature_id] = float(val)

map_pubmed["12187484"]

pubmed_labels[11943]

test_labels = np.sum(pubmed_labels, axis=1)

np.count_nonzero(test_labels)

len(edges)

pubmed = nx.DiGraph()

for t in pubmed_graph.edges():
    s, d = map(pubmed_type, t)
    pubmed.add_edge(s,d)

pubmed.size()

pubmed.number_of_nodes()

pubmed.number_of_edges()

pubmed.is_directed()

pubmed.edge[11943]

pubmed.in_edges(11943)

pubmed.out_edges(11943)

all_pubmed = {}
all_pubmed['NXGraph'] = pubmed
all_pubmed['Labels'] = pubmed_labels
all_pubmed['CSRFeatures'] = csr_matrix(pubmed_features)

all_pubmed['CSRFeatures']

with open('./pubmed.data', 'wb') as f:
    pickle.dump(all_pubmed, f)

pubmed_meta = find_meta('pubmed')

pubmed = GraphContainer(pubmed_meta, dataloc='.')

pubmed_gt = pubmed.get_gt_graph()

node_color = pubmed_gt.new_vertex_property("int")
for i, l in enumerate(pubmed.get_labels()):
    node_color[i] = np.argmax(l)

get_ipython().magic('time graph_draw(pubmed_gt, vertex_fill_color=node_color, output="pubmed_with_labels.png", output_size=(1200,1200))')

from motifwalk.motifs import all_3

for m in all_3:
    motif = m.gt_motif
    text = motif.new_vertex_property("string")
    for n in motif.vertices():
        text[n] = str(n)
    graph_draw(m.gt_motif, vertex_text=text, output_size=(80,80))

feed_forward = all_3[9]

motif = feed_forward.gt_motif
text = motif.new_vertex_property("string")
for n in motif.vertices():
    text[n] = str(n)
graph_draw(motif, vertex_text=text, output_size=(100,100))

feed_forward.anchors = {1,2}

from motifwalk.motifs.analysis import construct_motif_graph

ff_pubmed = construct_motif_graph(pubmed, feed_forward)

ff_pubmed

get_ipython().magic('time graph_draw(ff_pubmed, output="pubmed_with_labels_feedforward.png", output_size=(1200,1200))')

vfilt = ff_pubmed.new_vertex_property('bool');
for i in ff_pubmed.vertices():
    v = ff_pubmed.vertex(i)
    if v.out_degree() > 0:
        vfilt[i] = True
    else:
        vfilt[i] = False

ff_pubmed_filtered = GraphView(ff_pubmed, vfilt)

get_ipython().magic('time graph_draw(ff_pubmed_filtered, output="pubmed_with_labels_feedforward_filtered.png", output_size=(1200,1200))')

node_color = pubmed_gt.new_vertex_property("int")
for i, l in enumerate(pubmed.get_labels()):
    node_color[i] = np.argmax(l)

get_ipython().magic('time graph_draw(ff_pubmed_filtered, output="pubmed_with_labels_feedforward_filtered.png", vertex_fill_color=node_color, output_size=(1200,1200))')



