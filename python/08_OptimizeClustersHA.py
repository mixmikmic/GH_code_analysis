get_ipython().magic('matplotlib inline')
import seaborn
import matplotlib.pyplot as plt
from kmeans import kmeans, randomsample
import numpy as np
import pandas as pd
import random
import os
from BorealWeights import BorealWeightedProblem
from pyomo.opt import SolverFactory
from gradutil import *
seed = 3

get_ipython().run_cell_magic('time', '', "random.seed(seed)\nnp.random.seed(seed)\n_,carbon,_,ha = init_boreal()\nX = normalize(ha.values)\nrandomcenters = randomsample(X, 50)\ncenters, xtoc, dist = kmeans(X,\n                             randomcenters,\n                             delta=.00001,\n                             maxiter=100,\n                             metric='cosine',\n                             verbose=1)")

get_ipython().run_cell_magic('time', '', 'C = centers.copy()\nweights = np.array([sum(xtoc==i) for i in range(len(C))])')

get_ipython().run_cell_magic('time', '', "clustProblemHA = BorealWeightedProblem(C,weights)\nopt = SolverFactory('glpk')\nresClustHA = opt.solve(clustProblemHA.model, False)")

HASurrogateList = res_to_list(clustProblemHA.model)
resultSurrogateHA = cluster_to_value(C, HASurrogateList, weights)
print("(iv) Combined Habitat {:.0f}".format(resultSurrogateHA))

resultOriginHA = clusters_to_origin(X, xtoc, HASurrogateList)
print("(iv) Combined Habitat {:.0f}".format(resultOriginHA))

orig_hc_x = np.concatenate((ha, carbon), axis=1)
clust_hc_x = normalize(orig_hc_x)
no_nan_hc_x = orig_hc_x.copy()
hc_inds = np.where(np.isnan(no_nan_hc_x))
no_nan_hc_x[hc_inds] = np.take(np.nanmin(no_nan_hc_x, axis=0) - np.nanmax(no_nan_hc_x, axis=0), hc_inds[1])

get_ipython().run_cell_magic('time', '', "random.seed(seed)\nnp.random.seed(seed)\nrandomcenters = randomsample(clust_hc_x, 50)\nhc_centers, hc_xtoc, hc_dist = kmeans(clust_hc_x,\n                             randomcenters,\n                             delta=.00001,\n                             maxiter=100,\n                             metric='cosine',\n                             verbose=1)")

get_ipython().run_cell_magic('time', '', 'hc_C = np.array([no_nan_hc_x[hc_xtoc == i].mean(axis=0) for i in range(len(hc_centers))])\n\nhc_weights = np.array([sum(hc_xtoc==i) for i in range(len(hc_C))])')

get_ipython().run_cell_magic('time', '', "clustProblem_hc_ha = BorealWeightedProblem(hc_C[:,:7],hc_weights)\nopt = SolverFactory('glpk')\nresClust_hc_ha = opt.solve(clustProblem_hc_ha.model, False)")

hc_HASurrogateList = res_to_list(clustProblem_hc_ha.model)
hc_resultSurrogateHA = cluster_to_value(hc_C[:,:7], hc_HASurrogateList, hc_weights)
print("(iv) Combined Habitat {:.0f}".format(hc_resultSurrogateHA))

hc_resultOriginHA = clusters_to_origin(orig_hc_x[:,:7], hc_xtoc, hc_HASurrogateList)
print("(iv) Combined Habitat {:.0f}".format(hc_resultOriginHA))

