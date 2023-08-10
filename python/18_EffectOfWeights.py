get_ipython().magic('matplotlib inline')
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ASF import ASF
from gradutil import *
from pyomo.opt import SolverFactory
seedn = 1

get_ipython().run_cell_magic('time', '', "revenue, carbon, deadwood, ha = init_boreal()\nn_revenue = nan_to_bau(revenue)\nn_carbon= nan_to_bau(carbon)\nn_deadwood = nan_to_bau(deadwood)\nn_ha = nan_to_bau(ha)\nide = ideal(False)\nnad = nadir(False)\nopt = SolverFactory('cplex')")

x = pd.concat((n_revenue, n_carbon, n_deadwood, n_ha), axis=1)
x_stack = np.dstack((n_revenue, n_carbon, n_deadwood, n_ha))
#Normalize all the columns in 0-1 scale
x_norm = normalize(x.values)
x_norm_stack = normalize(x_stack)

get_ipython().run_cell_magic('time', '', 'nclust1 = 200\nc, xtoc, dist = cluster(x_norm, nclust1, seedn, verbose=1)')

rng = range(nclust1)
total_weight = len(x_norm)
w_orig = np.array([sum(xtoc == i) for i in rng])
w_scale = np.array([sum(xtoc == i)/total_weight for i in rng])

c_close = np.array([x_norm_stack[np.argmin(dist[xtoc == i])] for i in range(nclust1)])

ref = np.array((ide[0], nad[1]+1, nad[2]+1, nad[3]+1))
ASF_lambda = lambda x: ASF(ide, nad, ref, c_close, weights=x[0], scalarization=x[1])

orig_asf   = ASF_lambda((w_orig, 'asf'));   res_orig_asf = opt.solve(orig_asf.model)
orig_stom  = ASF_lambda((w_orig, 'stom'));  res_orig_stom = opt.solve(orig_stom.model)
orig_guess = ASF_lambda((w_orig, 'guess')); res_orig_stom = opt.solve(orig_guess.model)

scale_asf   = ASF_lambda((w_scale, 'asf'));   res_scale_asf = opt.solve(scale_asf.model)
scale_stom  = ASF_lambda((w_scale, 'stom'));  res_scale_stom = opt.solve(scale_stom.model)
scale_guess = ASF_lambda((w_scale, 'guess')); res_scale_stom = opt.solve(scale_guess.model)

model_to_real_values(x_stack, scale_asf.model, xtoc) - model_to_real_values(x_stack, orig_asf.model, xtoc)

model_to_real_values(x_stack, scale_stom.model, xtoc) - model_to_real_values(x_stack, orig_stom.model, xtoc)

model_to_real_values(x_stack, scale_guess.model, xtoc) - model_to_real_values(x_stack, orig_guess.model, xtoc)

