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
seedn = 2

revenue, carbon, deadwood, ha = init_boreal()

X = np.concatenate((carbon.values, ha.values, deadwood.values, revenue.values), axis=1)
np.shape(X)

Y = np.concatenate((carbon.values, ha.values, deadwood.values, revenue.values), axis=1)

np.nanmin(ha.values)

np.nanmax(ha.values)

np.nanmin(X[:,7:14])

get_ipython().run_cell_magic('time', '', "random.seed(seedn)\nnp.random.seed(seedn)\n# preprocessing to add penalty for Nan values\nX[np.isnan(X)] = np.nanmin(X) - np.nanmax(X)\nrandomcenters = randomsample(X, 50)\ncenters, xtoc, dist = kmeans(X,\n                             randomcenters,\n                             delta=.00001,\n                             maxiter=100,\n                             metric='cosine',\n                             verbose=1)")

get_ipython().run_cell_magic('time', '', 'C = centers.copy()\nnvar = len(X)\nweights = np.array([sum(xtoc==i) for i in range(len(C))])/nvar')

Ccarbon = C[:,0:7]
Cha = C[:,7:14]
Cdeadwood = C[:,14:21]
Crevenue = C[:,21:]

get_ipython().run_cell_magic('time', '', "opt = SolverFactory('glpk')\n\nclustProblemCarbon = BorealWeightedProblem(Ccarbon,weights,nvar)\nresCarbon = opt.solve(clustProblemCarbon.model, False)\n\nclustProblemHa = BorealWeightedProblem(Cha,weights,nvar)\nresHA = opt.solve(clustProblemHa.model, False)\n\nclustProblemDeadwood = BorealWeightedProblem(Cdeadwood,weights,nvar)\nresDeadwood = opt.solve(clustProblemDeadwood.model, False)\n\nclustProblemRevenue = BorealWeightedProblem(Crevenue,weights,nvar)\nresRevenue = opt.solve(clustProblemRevenue.model, False)")

carbonSurrogateList = res_to_list(clustProblemCarbon.model)
haSurrogateList = res_to_list(clustProblemHa.model)
deadwoodSurrogateList = res_to_list(clustProblemDeadwood.model)
revenueSurrogateList = res_to_list(clustProblemRevenue.model)

resultSurrogateCarbon = cluster_to_value(Ccarbon, carbonSurrogateList, weights)
resultSurrogateHa = cluster_to_value(Cha, haSurrogateList, weights)
resultSurrogateDeadwood = cluster_to_value(Cdeadwood, deadwoodSurrogateList, weights)
resultSurrogateRev = cluster_to_value(Crevenue, revenueSurrogateList, weights)

print('Results straight from the surrogate values:')
print("(i) Harvest revenues {:.0f} M€".format(resultSurrogateRev/1000000))
print("(ii) Carbon storage {:.0f} x 10³ MgC".format(resultSurrogateCarbon/1e+3))
print("(iii) Deadwood index {:.0f} m³".format(resultSurrogateDeadwood))
print("(iv) Combined Habitat {:.0f}".format(resultSurrogateHa))

print('Results straight from the surrogate values:')
print("(i) Harvest revenues {:.0f} M€".format(res_value(resRevenue)/1000000))
print("(ii) Carbon storage {:.0f} x 10³ MgC".format(res_value(resCarbon)/1e+3))
print("(iii) Deadwood index {:.0f} m³".format(res_value(resDeadwood)))
print("(iv) Combined Habitat {:.0f}".format(res_value(resHA)))

resultOriginCarbon = clusters_to_origin(X[:,:7], xtoc, carbonSurrogateList)
resultOriginHa = clusters_to_origin(X[:,7:14], xtoc, haSurrogateList)
resultOriginDeadwood = clusters_to_origin(X[:,14:21], xtoc, deadwoodSurrogateList)
resultOriginRev = clusters_to_origin(X[:,21:], xtoc, revenueSurrogateList)

print('Results when surrogate mapped to real values:')
print("(i) Harvest revenues {:.0f} M€".format(resultOriginRev/1000000))
print("(ii) Carbon storage {:.0f} x 100 MgC".format(resultOriginCarbon/100))
print("(iii) Deadwood index {:.0f} m3".format(resultOriginDeadwood))

print("(iv) Combined Habitat {:.0f}".format(resultOriginHa))

