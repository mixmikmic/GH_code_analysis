# notebook to simulate a jupyter working environment
# the python module gets imported here...
from ipysig.core import IPySig
import networkx as nx

ctrl = IPySig('./app') # note this should point to the root folder of the express app

import pickle
import os

pkl_g = os.path.abspath(os.path.join('.','IPySig_demo_data', 'stars_test_graph.pkl'))
stars_graph = pickle.load(open(pkl_g, "rb" )) 

stars_graph.nodes(data=True)[:5]

stars_graph.edges(data=True)[:10]

ctrl.connect(stars_graph,'stars')




G = nx.random_graphs.newman_watts_strogatz_graph(80,15,0.2)
ctrl.connect(G, 'newman_watts')

ctrl.kill_express_process()











