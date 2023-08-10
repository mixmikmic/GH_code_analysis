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

get_ipython().run_cell_magic('time', '', "seed = 2\nrandom.seed(seed)\nnp.random.seed(seed)\n_,carbon,_,_ = init_boreal()\nX = carbon.values\nX[carbon.isnull()] = np.nanmin(carbon) - np.nanmax(carbon)\nrandomcenters = randomsample(X, 50)\ncenters, xtoc, dist = kmeans(X,\n                             randomcenters,\n                             delta=.00001,\n                             maxiter=100,\n                             metric='cosine',\n                             verbose=0)")

get_ipython().run_cell_magic('time', '', 'C = centers.copy()\nweights = np.array([sum(xtoc==i) for i in range(0,len(C))])')

get_ipython().run_cell_magic('time', '', "ClustProblem = BorealWeightedProblem(C,weights)\nopt = SolverFactory('glpk')\nres = opt.solve(ClustProblem.model, False)")

def res_to_list(model):
    resdict = model.x.get_values()
    reslist = np.zeros(model.n.value)
    for i,j in resdict.keys():
        if resdict[i,j] == 1.:
            reslist[i] = j
    return reslist

reslist = res_to_list(ClustProblem.model)

optim_result_surrogate = sum([C[ind,int(reslist[ind])]*weights[ind] for ind in range(len(reslist))])
optim_result_surrogate

optim_result_surrogate_origin = sum([sum(X[xtoc==ind][:,int(reslist[ind])]) for ind in range(len(reslist))])
optim_result_surrogate_origin

(optim_result_surrogate - optim_result_surrogate_origin)/optim_result_surrogate_origin

optim_result_orig = ideal()['carbon']
optim_result_orig

(optim_result_orig - optim_result_surrogate_origin) / optim_result_orig

(optim_result_orig - optim_result_surrogate) / optim_result_orig

