get_ipython().magic('matplotlib inline')
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from gradutil import *
from pyomo.opt import SolverFactory

seedn = 2
opt = SolverFactory('glpk')
solutions = ideal()
revenue, carbon, deadwood, ha = init_boreal()
x = np.concatenate((revenue, carbon, deadwood, ha), axis=1)

norm_data = x.copy()
inds = np.where(np.isnan(norm_data))
norm_data[inds] = np.take(np.nanmin(norm_data, axis=0),inds[1])

min_norm_x = normalize(norm_data)

get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (15,12)

def hist_plot_norm(data, ax, limits):
    ax[0,0].hist(data[:, :7])
    ax[0,0].axis(limits)
    ax[0,0].set_title('Timber Harvest Revenues')

    ax[0,1].hist(data[:, 7:14])
    ax[0,1].axis(limits)
    ax[0,1].set_title('Carbon storage')

    ax[1,0].hist(data[:, 14:21])
    ax[1,0].axis(limits)
    ax[1,0].set_title('Deadwood')

    ax[1,1].hist(data[:, 21:])
    ax[1,1].axis(limits)
    ax[1,1].set_title('Habitat availability')
    return ax

data = min_norm_x
fig, ax = plt.subplots(2,2)
limits = [.0, 1., 0, 30000]
hist_plot_norm(data, ax, limits)
plt.show()

get_ipython().run_cell_magic('time', '', 'nclust = 50\noptim_revenue50, optim_carbon50, optim_deadwood50, optim_ha50 = cNopt(x, min_norm_x, min_norm_x, opt, nclust, seedn)')

print('Relative differences to original values, 50 clusters')
print("(i) Harvest revenues difference {:.3f}".format((optim_revenue50-solutions['revenue'])/solutions['revenue']))
print("(ii) Carbon storage {:.3f}".format((optim_carbon50-solutions['carbon'])/solutions['carbon']))
print("(iii) Deadwood index {:.3f}".format((optim_deadwood50-solutions['deadwood'])/solutions['deadwood']))
print("(iv) Combined Habitat {:.3f}".format((optim_ha50-solutions['ha'])/solutions['ha']))

no_nan_x = x.copy()
inds = np.where(np.isnan(no_nan_x))
no_nan_x[inds] = np.take(np.nanmin(no_nan_x, axis=0) - np.nanmax(no_nan_x, axis=0), inds[1])

get_ipython().run_cell_magic('time', '', 'nclust = 50\npenalty_optim_revenue50, penalty_optim_carbon50, penalty_optim_deadwood50, penalty_optim_ha50 = cNopt(x, min_norm_x, no_nan_x, opt, nclust, seedn)')

print('Relative differences to original values, 50 clusters')
print("(i) Harvest revenues difference {:.3f}".format((penalty_optim_revenue50-solutions['revenue'])/solutions['revenue']))
print("(ii) Carbon storage {:.3f}".format((penalty_optim_carbon50-solutions['carbon'])/solutions['carbon']))
print("(iii) Deadwood index {:.3f}".format((penalty_optim_deadwood50-solutions['deadwood'])/solutions['deadwood']))
print("(iv) Combined Habitat {:.3f}".format((penalty_optim_ha50-solutions['ha'])/solutions['ha']))

norm_data = x.copy()
inds = np.where(np.isnan(norm_data))
norm_data[inds] = np.take((np.nanmin(norm_data, axis=0)-np.nanmax(norm_data, axis=0))/2,inds[1])
penalty_norm_x = normalize(norm_data)

fig, ax = plt.subplots(2,2)
limits = [.0, 1., 0, 30000]
hist_plot_norm(penalty_norm_x, ax, limits)
plt.show()

get_ipython().run_cell_magic('time', '', 'nclust = 50\nhalf_optim_revenue50, half_optim_carbon50, half_optim_deadwood50, half_optim_ha50 = cNopt(x, penalty_norm_x, no_nan_x, opt, nclust, seedn)')

print('Relative differences to original values, 50 clusters')
print("(i) Harvest revenues difference {:.3f}".format((half_optim_revenue50-solutions['revenue'])/solutions['revenue']))
print("(ii) Carbon storage {:.3f}".format((half_optim_carbon50-solutions['carbon'])/solutions['carbon']))
print("(iii) Deadwood index {:.3f}".format((half_optim_deadwood50-solutions['deadwood'])/solutions['deadwood']))
print("(iv) Combined Habitat {:.3f}".format((half_optim_ha50-solutions['ha'])/solutions['ha']))

norm_data = x.copy()
inds = np.where(np.isnan(norm_data))
norm_data[inds] = np.take((np.nanmin(norm_data, axis=0)-np.nanmax(norm_data, axis=0))*2,inds[1])
ridiculous_norm_x = normalize(norm_data)

fig, ax = plt.subplots(2,2)
limits = [.0, 1., 0, 30000]
hist_plot_norm(ridiculous_norm_x, ax, limits)
plt.show()

get_ipython().run_cell_magic('time', '', 'nclust = 50\nridic_optim_revenue50, ridic_optim_carbon50, ridic_optim_deadwood50, ridic_optim_ha50 = cNopt(x, ridiculous_norm_x, no_nan_x, opt, nclust, seedn)')

print('Relative differences to original values, 50 clusters')
print("(i) Harvest revenues difference {:.3f}".format((ridic_optim_revenue50-solutions['revenue'])/solutions['revenue']))
print("(ii) Carbon storage {:.3f}".format((ridic_optim_carbon50-solutions['carbon'])/solutions['carbon']))
print("(iii) Deadwood index {:.3f}".format((ridic_optim_deadwood50-solutions['deadwood'])/solutions['deadwood']))
print("(iv) Combined Habitat {:.3f}".format((ridic_optim_ha50-solutions['ha'])/solutions['ha']))

x_nan = x[any(np.isnan(x), axis=1),:]
np.shape(x_nan)

x_num = x[all(~np.isnan(x), axis=1),:]
np.shape(x_num)

x_nany2y4y6 = x_nan[np.logical_and(np.logical_and(np.isnan(x_nan[:,2]),np.isnan(x_nan[:,4])), np.isnan(x_nan[:,6])),:]
x_nany2y4n6 = x_nan[np.logical_and(np.logical_and(np.isnan(x_nan[:,2]),np.isnan(x_nan[:,4])), ~np.isnan(x_nan[:,6])),:]
x_nany2n4y6 = x_nan[np.logical_and(np.logical_and(np.isnan(x_nan[:,2]),~np.isnan(x_nan[:,4])), np.isnan(x_nan[:,6])),:]
x_nany2n4n6 = x_nan[np.logical_and(np.logical_and(np.isnan(x_nan[:,2]),~np.isnan(x_nan[:,4])), ~np.isnan(x_nan[:,6])),:]
x_nann2y4y6 = x_nan[np.logical_and(np.logical_and(~np.isnan(x_nan[:,2]),np.isnan(x_nan[:,4])), np.isnan(x_nan[:,6])),:]
x_nann2y4n6 = x_nan[np.logical_and(np.logical_and(~np.isnan(x_nan[:,2]),np.isnan(x_nan[:,4])), ~np.isnan(x_nan[:,6])),:]
x_nann2n4y6 = x_nan[np.logical_and(np.logical_and(~np.isnan(x_nan[:,2]),~np.isnan(x_nan[:,4])), np.isnan(x_nan[:,6])),:]
x_nann2n4n6 = x_nan[np.logical_and(np.logical_and(~np.isnan(x_nan[:,2]),~np.isnan(x_nan[:,4])), ~np.isnan(x_nan[:,6])),:]

np.shape(x_nany2y4y6), np.shape(x_nany2y4n6), np.shape(x_nany2n4y6), np.shape(x_nany2n4n6)

np.shape(x_nann2y4y6), np.shape(x_nann2y4n6), np.shape(x_nann2n4y6), np.shape(x_nann2n4n6)

np.shape(x_nany2y4y6)[0]+np.shape(x_nany2y4n6)[0]+np.shape(x_nann2y4n6)[0]+np.shape(x_nann2n4y6)[0]

x_nan1 = x_nany2y4y6
x_nan2 = x_nany2y4n6
x_nan3 = x_nann2y4n6
x_nan4 = x_nann2n4y6

clust_x_nan1 = np.concatenate((x_nan1[:,:6],x_nan1[:,7:13],x_nan1[:,14:20], x_nan1[:,21:27]),axis=1)
norm_clust_nan1 = normalize(clust_x_nan1)

get_ipython().run_cell_magic('time', '', 'nclust = 10\nc, xtoc, dist = cluster(norm_clust_nan1, nclust, seedn, verbose=1)\nweights = np.array([sum(xtoc == i) for i in range(len(c))])\nopt_x = np.array([x_nan1[xtoc == i].mean(axis=0)\n                  for i in range(nclust)])')

c_nan2 = x_nan2.mean(axis=0)
c_nan3 = x_nan3.mean(axis=0)
c_nan4 = x_nan4.mean(axis=0)

w_nan2 = np.shape(x_nan2)[0]
w_nan3 = np.shape(x_nan3)[0]
w_nan4 = np.shape(x_nan4)[0]

combined_data = np.concatenate((opt_x,np.array((c_nan2, c_nan3, c_nan4))), axis=0)
combined_weights = np.concatenate((weights, np.array((w_nan2, w_nan3, w_nan4))), axis=0)

res_x = np.concatenate((x_nan1, x_nan2, x_nan3, x_nan4), axis=0)
res_xtoc = np.concatenate((xtoc, 
                           np.ones(np.shape(x_nan2)[0])*(nclust), 
                           np.ones(np.shape(x_nan3)[0])*(nclust+1), 
                           np.ones(np.shape(x_nan4)[0])*(nclust+2)))

opt = SolverFactory('glpk')

prob1, prob2, prob3, prob4 = optimize_all(normalize(combined_data), combined_weights, opt)

val1 = model_to_real_values(res_x[:, :7], prob1[0].model, res_xtoc)
val2 = model_to_real_values(res_x[:, 7:14], prob2[0].model, res_xtoc)
val3 = model_to_real_values(res_x[:, 14:21], prob3[0].model, res_xtoc)
val4 = model_to_real_values(res_x[:, 21:], prob4[0].model, res_xtoc)

norm_num_x = normalize(x_num)
norm_nan_x = normalize(x_nan)

get_ipython().run_cell_magic('time', '', "opt = SolverFactory('glpk')\nreal_nan_revenue, real_nan_carbon, real_nan_deadwood, real_nan_ha = optimize_all(norm_nan_x, np.ones(len(norm_nan_x)), opt)")

revenue_list = values_to_list(real_nan_revenue[0].model, x_nan[:,:7])
carbon_list = values_to_list(real_nan_carbon[0].model, x_nan[:,7:14])
deadwood_list = values_to_list(real_nan_deadwood[0].model, x_nan[:,14:21])
ha_list = values_to_list(real_nan_ha[0].model, x_nan[:,21:])

get_ipython().run_cell_magic('time', '', 'nclust = 100\nn_nan_opt_revenue, n_nan_opt_carbon, n_nan_opt_deadwood, n_nan_opt_ha = cNopt(x_nan, norm_nan_x, norm_nan_x, opt, nclust, seedn)')

print('Relative differences to original values (calculated with Nan:s), 50 clusters')
print("(i) Harvest revenues difference {:.3f}".format((n_nan_opt_revenue-sum(revenue_list))/sum(revenue_list)))
print("(ii) Carbon storage {:.3f}".format((n_nan_opt_carbon-sum(carbon_list))/sum(carbon_list)))
print("(iii) Deadwood index {:.3f}".format((n_nan_opt_deadwood-sum(deadwood_list))/sum(deadwood_list)))
print("(iv) Combined Habitat {:.3f}".format((n_nan_opt_ha-sum(ha_list))/sum(ha_list)))

get_ipython().run_cell_magic('time', '', 'nclust = 25\nn_num_opt_revenue, n_num_opt_carbon, n_num_opt_deadwood, n_num_opt_ha = cNopt(x_num, norm_num_x, norm_num_x, opt, nclust, seedn)')

print('Relative differences to original values, 50 clusters')
print("(i) Harvest revenues difference {:.3f}".format((n_nan_opt_revenue + n_num_opt_revenue-solutions['revenue'])/solutions['revenue']))
print("(ii) Carbon storage {:.3f}".format((n_nan_opt_carbon + n_num_opt_carbon-solutions['carbon'])/solutions['carbon']))
print("(iii) Deadwood index {:.3f}".format((n_nan_opt_deadwood + n_num_opt_deadwood-solutions['deadwood'])/solutions['deadwood']))
print("(iv) Combined Habitat {:.3f}".format((n_nan_opt_ha + n_num_opt_ha-solutions['ha'])/solutions['ha']))

