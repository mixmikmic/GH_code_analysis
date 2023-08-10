import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

from collections import defaultdict
from datetime import datetime, date
from random import randint
from networkx.readwrite.json_graph import node_link_data

get_ipython().magic('matplotlib inline')

G = nx.read_gpickle('20150902_all_ird Final Graph.pkl')

G.nodes(data=True)[0]

pH1N1s = [n for n, d in G.nodes(data=True)           if d['reassortant']           and d['subtype'] == 'H1N1'           and d['collection_date'].year >= 2009           and d['host_species'] in ['Human', 'Swine']           and len(G.predecessors(n)) > 0]
len(pH1N1s)

pH1N1s[0:5]

def get_predecessors(nodes, num_degrees):
    """
    Gets the predecessors of the nodes, up to num_degrees specified.
    """
    assert isinstance(num_degrees, int), "num_degrees must be an integer."
    
    ancestors = defaultdict(list)  # a dictionary of number of degrees up and a list of nodes.
    
    degree = 0
    
    while degree <= num_degrees:
        degree += 1
        if degree == 1:
            for n in nodes:
                ancestors[degree].extend(G.predecessors(n))
        else:
            for n in ancestors[degree - 1]:
                ancestors[degree].extend(G.predecessors(n))
    
    return ancestors
    
ancestors = get_predecessors(pH1N1s, 3)

ancestors_subtypes = defaultdict(set)

for deg, parents in ancestors.items():
    for parent in parents:
        ancestors_subtypes[deg].add(G.node[parent]['subtype'])
    
ancestors_subtypes

def collate_nodes_of_interest(nodes, ancestors_dict):
    """
    Given a starting list of nodes and a dictionary of its ancestors and their degrees of separation
    from the starting list of nodes, return a subgraph comprising of those nodes.
    """
    nodes_of_interest = []
    nodes_of_interest.extend(nodes)
    for k in ancestors_dict.keys():
        nodes_of_interest.extend(ancestors[k])
    G_sub = G.subgraph(nodes_of_interest)

    return G_sub

G_sub = collate_nodes_of_interest(pH1N1s, ancestors,)

def serialize_and_write_to_disk(graph, handle):
    """
    Correctly serializes the datetime objects in a graph's edges.
    
    Then, write the graph to disk.
    """
    # Serialize timestamp for JSON compatibility
    date_handler = lambda obj: (
        obj.isoformat()
        if isinstance(obj, datetime)
        or isinstance(obj, date)
        else None
    )

    for n, d in graph.nodes(data=True):
        graph.node[n]['collection_date'] = date_handler(graph.node[n]['collection_date'])

    # Serialize the data to disk as a JSON file
    data = node_link_data(graph)
    s = json.dumps(data)

    with open(handle, 'w+') as f:
        f.write(s)

serialize_and_write_to_disk(G_sub, 'supp_data/viz/H1N1_graph.json')

h7n9s = [n for n, d in G.nodes(data=True)          if d['subtype'] == 'H7N9'          and d['host_species'] == 'Human'          and d['collection_date'].year == 2013]

ancestors = get_predecessors(h7n9s, 3)
G_sub = collate_nodes_of_interest(h7n9s, ancestors,)
serialize_and_write_to_disk(G_sub, 'supp_data/viz/H7N9_graph.json')

# Visualize the data
# First, start the HTPP server
get_ipython().system(' python -m http.server 8002')
# Next, load "localhost:80000/supp_data/viz/h1n1.html"





