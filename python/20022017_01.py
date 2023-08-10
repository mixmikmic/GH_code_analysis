import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import os
from os import getcwd
from os.path import dirname
from time import gmtime, strftime
import scipy as sp
import cvxpy as cvx
from sklearn.linear_model import lars_path
get_ipython().magic('matplotlib inline')
import sys

cwd = getcwd()
dir_root = dirname(cwd)
filepath = os.path.join(dir_root, 'src')
sys.path.append(filepath) #('/home/tianpei/Dropbox/Codes/Python/LatNet/src/')
print(filepath)
get_ipython().magic('load_ext Cython')

from simulation import SigBeliefNet

G0 = nx.balanced_tree(2,4, create_using = nx.DiGraph())

seed = 100
np.random.seed(seed)
nx.set_node_attributes(G0, 'bias', dict(zip(G0.nodes(), 3*np.random.rand(len(G0)))))

nx.set_edge_attributes(G0,  'weight', dict(zip(G0.edges(), 1.5*np.random.randn(G0.number_of_edges()))))

fig= plt.figure(1)
pos = nx.nx_pydot.graphviz_layout(G0)
nx.draw(G0, pos=pos, arrows=True, with_labels=True, fontsize=6)
plt.show()

sbn = SigBeliefNet(G0)

sbn.roots

sbn.sample(iter=4000)

view_angle = [25,25]
sbn.make_animate(G0, view_angle, False)

pm.Matplot.plot(sbn)



























