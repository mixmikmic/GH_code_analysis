from gradutil import *
import pandas as pd
from pyomo.opt import SolverFactory

revenue, carbon, deadwood, ha = init_boreal()
x_revenue = nan_to_bau(revenue)
x_carbon = nan_to_bau(carbon)
x_deadwood = nan_to_bau(deadwood)
x_ha = nan_to_bau(ha)
x = np.concatenate((x_revenue.values, x_carbon.values, x_deadwood.values, x_ha.values), axis=1)

n_revenue = new_normalize(x_revenue.values)
n_carbon = new_normalize(x_carbon.values)
n_deadwood = new_normalize(x_deadwood.values)
n_ha = new_normalize(x_ha.values)
x_norm = np.concatenate((n_revenue, n_carbon, n_deadwood, n_ha), axis=1)

opt = SolverFactory('glpk')
value_revenue, value_carbon, value_deadwood, value_ha = cNopt(x, x_norm, x, opt, nclust=10, seed=2)

value_revenue, value_carbon, value_deadwood, value_ha

tmp_x = np.concatenate((x_ha.values,x_ha.values,x_ha.values,x_ha.values), axis=1)
tmp = cNopt(tmp_x, normalize(tmp_x), tmp_x, opt, nclust=100, seed=4)

centers, xtoc, dist = cluster(x_ha.values, 50, seed=3)

get_ipython().run_cell_magic('time', '', "weights = np.array([sum(xtoc==i) for i in range(len(centers))])\nclustProblemHA = BorealWeightedProblem(centers,weights)\nopt = SolverFactory('glpk')\nresClustHA = opt.solve(clustProblemHA.model, False)")

val_ha = model_to_real_values(x_ha.values, clustProblemHA.model, xtoc)
real_ha = ideal()['ha']
(val_ha-real_ha)/real_ha

