import gradutil as gu
import numpy as np
import pandas as pd
import logging
import datetime
import simplejson as json
from time import time
from pyomo.opt import SolverFactory
from scipy.spatial.distance import euclidean
from BorealWeights import BorealWeightedProblem

def clustering(x, nclusts, seeds, logger=None, starttime=None):
    res = dict()
    for nclust in nclusts:
        res_clust = dict()
        for seedn in seeds:
            c, xtoc, dist = gu.cluster(x, nclust, seedn, verbose=0)
            res_clust[seedn] = {'c': c.tolist(),
                                'xtoc': xtoc.tolist(),
                                'dist': dist.tolist()}
            if logger:
                logger.info('Clustered to {} clusters. Seed {}'.format(nclust, seedn))
            if starttime:
                logger.info('Since start {}.'.format(str(datetime.timedelta(seconds=time()-starttime))))
        res[nclust] = res_clust
        if logger:
            logger.info('Clustered to {:2.0f} clusters'.format(nclust))
        if starttime:
            logger.info('Since start {}.'.format(str(datetime.timedelta(seconds=time()-starttime))))
        #with open('clusterings/new_{}.json'.format(nclust), 'w') as file:
         #   json.dump(res_clust, file)
    return res

def clustering_to_dict(readfile):
    with open(readfile, 'r') as rfile:
        clustering = json.loads(rfile.read())

    new_clustering = dict()
    for seedn in clustering.keys():
        new_clustering[eval(seedn)] = dict()
        for key in clustering[seedn].keys():
            new_clustering[eval(seedn)][key] = np.array(clustering[seedn][key])
    return new_clustering

#    def clustering_to_optims(x_orig, x_clust, x_opt, names, nclust, opt, logger=None, starttime=None):
start = time()
logger.info('Started optimizing')
names = ['revenue', 'carbon', 'deadwood', 'ha']
nclusts4 = range(1700, 8501, 200)
x_orig = x_stack
x_clust = x_norm
x_opt = x_norm_stack
starttime = start
for nclust in nclusts4:
        readfile = 'clusterings/new_{}.json'.format(nclust)
        with open(readfile, 'r') as rfile:
            read_clustering = json.loads(rfile.read())

        clustering = dict()
        for seedn in read_clustering.keys():
            clustering[eval(seedn)] = dict()
            for key in read_clustering[seedn].keys():
                clustering[eval(seedn)][key] = np.array(read_clustering[seedn][key])
        
        n_optims = dict()
        for seedn in clustering.keys():
            xtoc = np.array(clustering[seedn]['xtoc'])
            #if logger:
            logger.info('Assigning weights')
            #if starttime:
            logger.info('Since start {}.'.format(str(datetime.timedelta(seconds=int(time()-starttime)))))
            w = np.array([sum(xtoc == i)
                          for i in range(nclust)
                          if sum(xtoc == i) > 0])
            # Calculate the euclidian center of the cluster (mean)
            # and then the point closest to that center according to
            # euclidian distance, and then use the data format meant
            # for optimization
            #if logger:
            logger.info('Assigning centers')
            #if starttime:
            logger.info('Since start {}.'.format(str(datetime.timedelta(seconds=int(time()-starttime)))))
            indices = [min(np.array(range(len(xtoc)))[xtoc == i],
                           key=lambda index: euclidean(x_clust[index],
                                                       np.mean(x_clust[xtoc == i],
                                                               axis=0)))
                       for i in range(nclust) if sum(xtoc == i) > 0]
            c_close = x_opt[indices]
            x_close = x_orig[indices]
            problems = [BorealWeightedProblem(c_close[:, :, i], weights=w)
                        for i in range(np.shape(c_close)[-1])]
            #if logger:
            logger.info('Solving problems')
            #if starttime:
            logger.info('Since start {}.'.format(str(datetime.timedelta(seconds=int(time()-starttime)))))
            for p in problems:
                opt.solve(p.model)
            n_optims[seedn] = dict()
            for ind, name in enumerate(names):
                n_optims[seedn][name] = dict()
                n_optims[seedn][name]['real'] = gu.model_to_real_values(
                    x_orig[:, :, ind],
                    problems[ind].model,
                    xtoc)
                n_optims[seedn][name]['surrogate'] = gu.cluster_to_value(
                    x_close[:, :, ind], gu.res_to_list(problems[ind].model), w)
            #if logger:
            logger.info('Optimized {} clusters with seed {}'.format(nclust, seedn))
            #if starttime:
            logger.info('Since start {}.'.format(str(datetime.timedelta(seconds=int(time()-starttime)))))
        #if logger:
        logger.info('Optimized {} clusters with every seed'.format(nclust))
        #if starttime:
        logger.info('Since start {}.'.format(str(datetime.timedelta(seconds=int(time()-starttime)))))
        with open('optimizations/hope_{}.json'.format(nclust), 'w') as file:
            json.dump(n_optims, file)

revenue, carbon, deadwood, ha = gu.init_boreal()

n_revenue = gu.nan_to_bau(revenue)
n_carbon = gu.nan_to_bau(carbon)
n_deadwood = gu.nan_to_bau(deadwood)
n_ha = gu.nan_to_bau(ha)

revenue_norm = gu.new_normalize(n_revenue.values)
carbon_norm = gu.new_normalize(n_carbon.values)
deadwood_norm = gu.new_normalize(n_deadwood.values)
ha_norm = gu.new_normalize(n_ha.values)

ide = gu.ideal(False)
nad = gu.nadir(False)
opt = SolverFactory('cplex')

x = np.concatenate((n_revenue.values, n_carbon.values, n_deadwood.values, n_ha.values), axis=1)
x_stack = np.dstack((n_revenue, n_carbon, n_deadwood, n_ha))

x_norm = np.concatenate((revenue_norm, carbon_norm, deadwood_norm, ha_norm), axis=1)
x_norm_stack = np.dstack((revenue_norm, carbon_norm, deadwood_norm, ha_norm))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

get_ipython().run_cell_magic('time', '', "start = time()\nlogger.info('Started clustering')\nnclusts3 = range(1600, 1700, 50)\nseeds = range(2, 12)\n\nclustering(x_norm, [600], [2], logger, start)\nlogger.info('All clustered to 50. Time since start {}.'.format(str(datetime.timedelta(seconds=time()-start))))")

start = time()
logger.info('Started optimizing')
names = ['revenue', 'carbon', 'deadwood', 'ha']
nclusts4 = range(1700, 8501, 200)
for nclust in nclusts4:
    clustering_to_optims(x_stack, x_norm, x_norm_stack, names, nclust, opt, logger=logger, starttime=start)
logger.info('All optimized: 1700-8500-200. Since start {}'.format(str(datetime.timedelta(seconds=int(time()-start)))))

