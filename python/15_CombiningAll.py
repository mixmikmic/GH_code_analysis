get_ipython().magic('matplotlib inline')
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ASF import ASF
from gradutil import *
from pyomo.opt import SolverFactory
seedn = 2

get_ipython().run_cell_magic('time', '', "revenue, carbon, deadwood, ha = init_boreal()\nn_revenue = nan_to_bau(revenue)\nn_carbon= nan_to_bau(carbon)\nn_deadwood = nan_to_bau(deadwood)\nn_ha = nan_to_bau(ha)\nide = ideal(False)\nnad = nadir(False)\nopt = SolverFactory('glpk')")

x = pd.concat((n_revenue, n_carbon, n_deadwood, n_ha), axis=1)
x_stack = np.dstack((n_revenue, n_carbon, n_deadwood, n_ha))

norm_revenue = new_normalize(n_revenue)
norm_carbon = new_normalize(n_carbon)
norm_deadwood = new_normalize(n_deadwood)
norm_ha = new_normalize(n_ha)

x_norm = np.concatenate((norm_revenue, norm_carbon, norm_deadwood, norm_ha), axis=1)
x_norm_stack = np.dstack((norm_revenue, norm_carbon, norm_deadwood, norm_ha))

get_ipython().run_cell_magic('time', '', 'nclust1 = 200\nc, xtoc, dist = cluster(x_norm, nclust1, seedn, verbose=1)\nnvar = len(x_norm)\nw = np.array([sum(xtoc == i) for i in range(nclust1)])/nvar')

c_mean = np.array([x_norm_stack[xtoc == i].mean(axis=0) for i in range(nclust1)])

from scipy.spatial.distance import euclidean
c_close = np.array([x_norm_stack[min(np.array(range(len(xtoc)))[xtoc == i],
                                     key=lambda index: euclidean(x_norm[index],
                                                                 np.mean(x_norm[xtoc == i], axis=0)))]
                    for i in range(nclust1)
                    if sum(xtoc == i) > 0])

ref = np.array((ide[0], 0, 0, 0))
mean_asf = ASF(ide, nad, ref, c_mean, weights=w, nvar=nvar)
res_asf_mean = opt.solve(mean_asf.model)
mean_stom = ASF(ide, nad, ref, c_mean, weights=w, nvar=nvar, scalarization='stom')
res_stom_mean = opt.solve(mean_stom.model)
mean_guess = ASF(ide, nad, ref, c_mean, weights=w, nvar=nvar, scalarization='guess')
res_stom_mean = opt.solve(mean_guess.model)

(model_to_real_values(x_stack, mean_asf.model, xtoc),
model_to_real_values(x_stack, mean_stom.model, xtoc),
model_to_real_values(x_stack, mean_guess.model, xtoc),
ide)

ref = np.array((ide[0], 0, 0, 0))
close_asf = ASF(ide, nad, ref, c_close, weights=w, nvar=nvar)
res_close = opt.solve(close_asf.model)
close_stom = ASF(ide, nad, ref, c_close, weights=w, nvar=nvar, scalarization='stom')
res_stom_close = opt.solve(close_stom.model)
close_guess = ASF(ide, nad, ref, c_close, weights=w, nvar=nvar, scalarization='guess')
res_stom_close = opt.solve(close_guess.model)

(model_to_real_values(x_stack, close_asf.model, xtoc),
model_to_real_values(x_stack, close_stom.model, xtoc),
model_to_real_values(x_stack, close_guess.model, xtoc),
ide)

ref = np.array((0, ide[1], 0, 0))
close_asf = ASF(ide, nad, ref, c_close, weights=w, nvar=nvar)
res_close = opt.solve(close_asf.model)
close_stom = ASF(ide, nad, ref, c_close, weights=w, nvar=nvar, scalarization='stom')
res_stom_close = opt.solve(close_stom.model)
close_guess = ASF(ide, nad, ref, c_close, weights=w, nvar=nvar, scalarization='guess')
res_stom_close = opt.solve(close_guess.model)

(model_to_real_values(x_stack, close_asf.model, xtoc),
model_to_real_values(x_stack, close_stom.model, xtoc),
model_to_real_values(x_stack, close_guess.model, xtoc),
ide)

ref = np.array((0, 0,ide[2], 0))
close_asf = ASF(ide, nad, ref, c_close, weights=w, nvar=nvar)
res_close = opt.solve(close_asf.model)
close_stom = ASF(ide, nad, ref, c_close, weights=w, nvar=nvar, scalarization='stom')
res_stom_close = opt.solve(close_stom.model)
close_guess = ASF(ide, nad, ref, c_close, weights=w, nvar=nvar, scalarization='guess')
res_stom_close = opt.solve(close_guess.model)

(model_to_real_values(x_stack, close_asf.model, xtoc),
model_to_real_values(x_stack, close_stom.model, xtoc),
model_to_real_values(x_stack, close_guess.model, xtoc), 
ide)

ref = np.array((0, 0, 0, ide[3]))
close_asf = ASF(ide, nad, ref, c_close, weights=w, nvar=nvar)
res_close = opt.solve(close_asf.model)
close_stom = ASF(ide, nad, ref, c_close, weights=w, nvar=nvar, scalarization='stom')
res_stom_close = opt.solve(close_stom.model)
close_guess = ASF(ide, nad, ref, c_close, weights=w, nvar=nvar, scalarization='guess')
res_stom_close = opt.solve(close_guess.model)

(model_to_real_values(x_stack, close_asf.model, xtoc),
model_to_real_values(x_stack, close_stom.model, xtoc),
model_to_real_values(x_stack, close_guess.model, xtoc),
ide)

