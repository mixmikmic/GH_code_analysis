# this line makes the code compatible with Python 2 and 3
from __future__ import print_function, division

# this line makes Jupyter show figures in the notebook
get_ipython().magic('matplotlib inline')

from collections import deque

def bfs(G, start):
    """A simple version of BFS that just computes distances.
    
    G: Graph
    start: int start node
    
    returns: map from node to distance
    """
    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()         
        for child in G.neighbors(node):
            if child not in dist:                    
                dist[child] = dist[node] + 1
                queue.append(child)
    return dist

from networkx import DiGraph

def bfs(G, start):
    """A simple version of BFS that computes distances and
        paths back to start.
    
    G: Graph
    start: int start node
    
    returns: (map from node to distance,
        DiGraph containing paths from each node back to start)
    """
    dist = {start: 0}
    tree = DiGraph()
    queue = deque([start])
    while queue:
        node = queue.popleft()         
        for child in G.neighbors(node):
            if child not in dist:                    
                dist[child] = dist[node] + 1
                tree.add_edge(child, node) 
                queue.append(child)
    return dist, tree

from numpy.random import random

def flip(p):
    return random() < p

def random_pairs(nodes, p):
    """Generate random pairs of nodes."""
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i<j and flip(p):
                yield u, v

import networkx as nx

def make_random_graph(n, p):
    """Generate a random graph.
    
    n: number of nodes
    p: probability of an edge between any pair of nodes
    
    returns: Graph
    """
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(random_pairs(nodes, p))
    return G

random_graph = make_random_graph(10, 0.3)
len(random_graph.edges())

import seaborn as sns
COLORS = sns.color_palette()

nx.draw_circular(random_graph, 
                 node_color=COLORS[2], 
                 node_size=1000, 
                 with_labels=True)

dist, tree = bfs(random_graph, 0)
dist

nx.draw_circular(tree, 
                 node_color=COLORS[2], 
                 node_size=1000, 
                 with_labels=True)

def fft(ys):
    N = len(ys)
    if N == 1:
        return ys
    
    He = fft(ys[::2])
    Ho = fft(ys[1::2])
    
    ns = np.arange(N)
    W = np.exp(-1j * PI2 * ns / N)
    
    return np.tile(He, 2) + W * np.tile(Ho, 2)



