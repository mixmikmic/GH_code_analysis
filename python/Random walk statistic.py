from sys import path
path.append('./../src/')  # import prototype modules
from constrains import R, UTriangle, UWedge
from walks import WalkGenerator
import networkx as nx
import numpy as np
import sklearn as sk
from matplotlib import pyplot as plt
import pickle
from collections import defaultdict as dd
import time
get_ipython().magic('matplotlib inline')

data_root = './../data/'

amazon = pickle.load(open(data_root+'amazon.graph', 'rb'))
assert isinstance(amazon, nx.Graph)
assert amazon.size() == 925872

random_walker = WalkGenerator(graph=amazon, constrain=R())

def get_graph_context(random_walker, num_walk=20000, walk_length=80):
    """Return randomly generated graph context"""
    t0 = t1 = time.time()
    random_context = [i[:] for i in random_walker._gen(num_walk, walk_length)]
    node_count = dd(int)
    for node_list in random_context:
        for i in node_list:
            node_count[i] += 1
    sorted_ids = sorted(node_count,
                         key=lambda i: node_count[i],
                         reverse=True)
    t1 = time.time()
    print("Time elapsed: {}".format(t1-t0))
    return random_context, node_count, sorted_ids

amazon_random_context, amazon_random_node_count, amazon_sorted_ids = get_graph_context(random_walker)

def plot_freq_dist(shorted_ids, node_count):
    fig, (ax, ax_log) = plt.subplots(2,1)
    x = np.arange(0, len(node_count), 1, dtype=int)
    y = [node_count[shorted_ids[i]] for i in x]
    ax.plot(x, y)
    ax_log.loglog(x, y)
    plt.show()
    
plot_freq_dist(amazon_sorted_ids, amazon_random_node_count)

triangle_walker = WalkGenerator(graph=amazon, constrain=UTriangle())
amazon_triangle_context, amazon_triangle_node_count, amazon_triangle_sorted_ids = get_graph_context(triangle_walker)

plot_freq_dist(amazon_triangle_sorted_ids, amazon_triangle_node_count)

amazon_triangle_sorted_ids[:10]

amazon_sorted_ids[:10]

cora = pickle.load(open(data_root+'cora.graph', 'rb'))

ucora = cora.to_undirected()  # Convert to undirected graph

random_ucora_walker = WalkGenerator(graph=ucora, constrain=R())

cora_random_context, cora_random_node_count, cora_sorted_ids = get_graph_context(random_ucora_walker)

plot_freq_dist(cora_sorted_ids, cora_random_node_count)

triangle_ucora_walker = WalkGenerator(graph=ucora, constrain=UTriangle())

cora_triangle_context, cora_triangle_node_count, cora_triangle_sorted_ids = get_graph_context(triangle_ucora_walker)

plot_freq_dist(cora_triangle_sorted_ids, cora_triangle_node_count)

wedge_ucora_walker = WalkGenerator(graph=ucora, constrain=UWedge())
cora_wedge_context, cora_wedge_node_count, cora_wedge_sorted_ids = get_graph_context(wedge_ucora_walker)
plot_freq_dist(cora_wedge_sorted_ids, cora_wedge_node_count)



