import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import *

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

N = int(10)
M = 1


# uniform attachment
ua = ig.Graph.Growing_Random(n=N, m=M,
                             directed=True, citation=True)



g = ua.copy()
g.vs['indeg'] = g.indegree()

g.vs.select(_indegree_gt = 5)

# subgraph of vertices who have indegree at least one
dsg = g.subgraph(g.vs.select(_indegree_ge=1))

# vertices who have at least one citer whose degree is at least one
dsg.vs.select(indeg_ge = 1)



