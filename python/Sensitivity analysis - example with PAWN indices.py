import brightway2 as bw

from time import time

bw.projects.set_current("bw2_seminar_2017")

db = bw.Database("ecoinvent 2.2")

a = db.random()
print(a)

for x in a.technosphere():
    print(x['amount'], x['uncertainty type'], x['scale'], bw.mapping[x['input']], x.input)

m = ('CML 2001', 'climate change', 'GWP 500a')

pv = bw.ParameterVectorLCA({a.key: 1}, m)
next(pv)

index = bw.mapping[a]

ra, _, _ = pv.reverse_dict()

indices = []

for i, x in enumerate(pv.tech_params):
    if x['output'] == index:
        if x['output'] != x['input']:
            indices.append(i)
        print(i, bw.get_activity(ra[int(x['row'])]))

from bw2calc import ParameterVectorLCA, ParallelMonteCarlo
from scipy.stats import ks_2samp
from stats_arrays import uncertainty_choices
import multiprocessing
import numpy as np


def pawn_worker(args):
    fu, method, index, unconditional_cdf, n_c, n = args
    lca = ParameterVectorLCA(fu, method)
    lca.load_data()
    next(lca)
    array = lca.params[index:index + 1]
    lhc = (np.random.random(size=n) + np.arange(n)) / n
    n_samples = uncertainty_choices[int(array['uncertainty_type'][0])].ppf(
        array, lhc.reshape((1, -1))).ravel()
    results = []

    for fixed in n_samples:

        scores = []

        for _ in range(n_c):
            sample = lca.rng.next()
            sample[index] = fixed

            lca.rebuild_all(sample)
            lca.lci_calculation()
            lca.lcia_calculation()
            scores.append(lca.score)

        results.append(ks_2samp(unconditional_cdf, np.array(scores))[0])

    return (index, np.median(results))
    

def pawn_sensitivity(fu, method, indices, cpus=None, n_u=1000, n_c=100, n=20):
    unconditional_cdf = ParallelMonteCarlo(fu, method, n_u, cpus=cpus).calculate()

    with multiprocessing.Pool(processes=cpus) as pool:
        results = pool.map(
            pawn_worker,
            [(fu, method, index, unconditional_cdf.copy(), n_c, n) 
             for index in indices]
        )
    return results

def pawn_worker_full(args):
    fu, method, index, unconditional_cdf, n_c, n = args
    lca = ParameterVectorLCA(fu, method)
    lca.load_data()
    next(lca)
    array = lca.params[index:index + 1]
    lhc = (np.random.random(size=n) + np.arange(n)) / n
    n_samples = uncertainty_choices[int(array['uncertainty_type'][0])].ppf(
        array, lhc.reshape((1, -1))).ravel()
    results = []

    for fixed in n_samples:

        scores = []

        for _ in range(n_c):
            sample = lca.rng.next()
            sample[index] = fixed

            lca.rebuild_all(sample)
            lca.lci_calculation()
            lca.lcia_calculation()
            scores.append(lca.score)

        results.append(np.array(scores))

    return (index, results)
    

def pawn_sensitivity_full(fu, method, indices, cpus=None, n_u=1000, n_c=100, n=20):
    unconditional_cdf = ParallelMonteCarlo(fu, method, n_u, cpus=cpus).calculate()

    with multiprocessing.Pool(processes=cpus) as pool:
        results = pool.map(
            pawn_worker_full,
            [(fu, method, index, unconditional_cdf.copy(), n_c, n) 
             for index in indices]
        )
    return unconditional_cdf, results

get_ipython().magic('matplotlib inline')

from matplotlib import pyplot as plt

empirical_cdf = lambda xs: np.linspace(0, 1, int(xs.shape[0]))

start = time()
unconditional_cdf, results = pawn_sensitivity_full({a.key: 1}, m, indices)
(time() - start) / 60

for index, conditional_cdfs in results:

    xs = np.sort(unconditional_cdf)
    plt.plot(xs, empirical_cdf(xs), lw=2, color='black')

    for ds in conditional_cdfs:
        ds = np.sort(ds)
        plt.plot(ds, empirical_cdf(ds), lw=1, ls='-')
    plt.show()

for index, conditional_cdfs in results:
    print(index, np.median([ks_2samp(unconditional_cdf, sample)[0] for sample in conditional_cdfs]))



