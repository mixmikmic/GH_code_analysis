from gradutil import *
from BorealWeights import BorealWeightedProblem
from pyomo.opt import SolverFactory

seedn = 4
opt = SolverFactory('glpk')
solutions = ideal()
revenue, carbon, deadwood, ha = init_boreal()
x = np.concatenate((revenue, carbon, deadwood, ha), axis=1)

norm_revenue = normalize(revenue.values)
norm_carbon = normalize(carbon.values)
norm_deadwood = normalize(deadwood.values)
norm_ha = normalize(ha.values)
cluster_x = np.concatenate((norm_revenue, norm_carbon, norm_deadwood, norm_ha), axis=1)

no_nan_x = x.copy()
inds = np.where(np.isnan(no_nan_x))
no_nan_x[inds] = np.take(np.nanmin(no_nan_x, axis=0) - np.nanmax(no_nan_x, axis=0), inds[1])

get_ipython().run_cell_magic('time', '', 'nclust = 5\noptim_revenue5, optim_carbon5, optim_deadwood5, optim_ha5 = cNopt(x, cluster_x, no_nan_x, opt, nclust, seedn)')

print('Relative differences to original values, 5 clusters')
print("(i) Harvest revenues difference {:.3f}".format((optim_revenue5-solutions['revenue'])/solutions['revenue']))
print("(ii) Carbon storage {:.3f}".format((optim_carbon5-solutions['carbon'])/solutions['carbon']))
print("(iii) Deadwood index {:.3f}".format((optim_deadwood5-solutions['deadwood'])/solutions['deadwood']))
print("(iv) Combined Habitat {:.3f}".format((optim_ha5-solutions['ha'])/solutions['ha']))

get_ipython().run_cell_magic('time', '', 'nclust = 25\noptim_revenue25, optim_carbon25, optim_deadwood25, optim_ha25 = cNopt(x, cluster_x, no_nan_x, opt, nclust, seedn)')

print('Relative differences to original values, 25 clusters')
print("(i) Harvest revenues difference {:.3f}".format((optim_revenue25-solutions['revenue'])/solutions['revenue']))
print("(ii) Carbon storage {:.3f}".format((optim_carbon25-solutions['carbon'])/solutions['carbon']))
print("(iii) Deadwood index {:.3f}".format((optim_deadwood25-solutions['deadwood'])/solutions['deadwood']))
print("(iv) Combined Habitat {:.3f}".format((optim_ha25-solutions['ha'])/solutions['ha']))

get_ipython().run_cell_magic('time', '', 'nclust = 50\noptim_revenue50, optim_carbon50, optim_deadwood50, optim_ha50 = cNopt(x, cluster_x, no_nan_x, opt, nclust, seedn)')

print('Relative differences to original values, 50 clusters')
print("(i) Harvest revenues difference {:.3f}".format((optim_revenue50-solutions['revenue'])/solutions['revenue']))
print("(ii) Carbon storage {:.3f}".format((optim_carbon50-solutions['carbon'])/solutions['carbon']))
print("(iii) Deadwood index {:.3f}".format((optim_deadwood50-solutions['deadwood'])/solutions['deadwood']))
print("(iv) Combined Habitat {:.3f}".format((optim_ha50-solutions['ha'])/solutions['ha']))

get_ipython().run_cell_magic('time', '', "nclust = 100\nopt = SolverFactory('glpk')\noptim_revenue100, optim_carbon100, optim_deadwood100, optim_ha100 = cNopt(x, cluster_x, no_nan_x, opt, nclust, seedn)")

print('Relative differences to original values, 100 clusters')
print("(i) Harvest revenues difference {:.3f}".format((optim_revenue100-solutions['revenue'])/solutions['revenue']))
print("(ii) Carbon storage {:.3f}".format((optim_carbon100-solutions['carbon'])/solutions['carbon']))
print("(iii) Deadwood index {:.3f}".format((optim_deadwood100-solutions['deadwood'])/solutions['deadwood']))
print("(iv) Combined Habitat {:.3f}".format((optim_ha100-solutions['ha'])/solutions['ha']))

get_ipython().run_cell_magic('time', '', "nclust = 500\nopt = SolverFactory('glpk')\noptim_revenue500, optim_carbon500, optim_deadwood500, optim_ha500 = cNopt(x, cluster_x, no_nan_x, opt, nclust, seedn)")

print('Relative differences to original values, 100 clusters')
print("(i) Harvest revenues difference {:.3f}".format((optim_revenue500-solutions['revenue'])/solutions['revenue']))
print("(ii) Carbon storage {:.3f}".format((optim_carbon500-solutions['carbon'])/solutions['carbon']))
print("(iii) Deadwood index {:.3f}".format((optim_deadwood500-solutions['deadwood'])/solutions['deadwood']))
print("(iv) Combined Habitat {:.3f}".format((optim_ha500-solutions['ha'])/solutions['ha']))

z = np.concatenate((revenue.dropna(axis=0, how='any'), 
                    carbon.dropna(axis=0, how='any'), 
                    deadwood.dropna(axis=0, how='any'), 
                    ha.dropna(axis=0, how='any')), axis=1)

cluster_z = normalize(z)

get_ipython().run_cell_magic('time', '', 'nclust = 50\noptim_revenuez5, optim_carbonz5, optim_deadwoodz5, optim_haz5 = cNopt(z, cluster_z, z, opt, nclust, seedn)')

sol_revenue, sol_carbon, sol_deadwood, sol_ha = optimize_all(z, np.ones(len(z)), opt)

revenue_list = values_to_list(sol_revenue[0].model, z[:,:7])
carbon_list = values_to_list(sol_carbon[0].model, z[:,7:14])
deadwood_list = values_to_list(sol_deadwood[0].model, z[:,14:21])
ha_list = values_to_list(sol_ha[0].model, z[:,21:])

print('Relative differences to original values (calculated without Nan:s), 50 clusters')
print("(i) Harvest revenues difference {:.3f}".format((optim_revenuez5-sum(revenue_list))/sum(revenue_list)))
print("(ii) Carbon storage {:.3f}".format((optim_carbonz5-sum(carbon_list))/sum(carbon_list)))
print("(iii) Deadwood index {:.3f}".format((optim_deadwoodz5-sum(deadwood_list))/sum(deadwood_list)))
print("(iv) Combined Habitat {:.3f}".format((optim_haz5-sum(ha_list))/sum(ha_list)))

