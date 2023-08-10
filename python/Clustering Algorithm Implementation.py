import numpy as np
import random

import matplotlib.pyplot as plt

import sys

#Add the src folder to the sys.path list
sys.path.append('../src/')
import data_config as dc

import networkx as nx

connectome = dc.connectome_networkx.data()



not_visited = sensory_neurons
visited = {}
nodes = network.nodes()

while not_visited:
    node = nodes.pop(0)
    edges = network.edge(node)
    for s,e in edges:
        if not e in visited: 
            not_visited.append(e)



