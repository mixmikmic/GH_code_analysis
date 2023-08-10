import multiprocessing

from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

num_cores

from math import sqrt

Parallel(n_jobs=num_cores)(delayed(sqrt)(i**2) for i in range(10))

import quantecon as qe
import numpy as np
from scipy.linalg import eigvals

def lucas_tree_spec_rad(beta=0.96,
                        gamma=2.0,
                        sigma=0.1,
                        b=0.0,
                        rho=0.9,
                        n=200):

    mc = qe.tauchen(rho, sigma, n=n)  
    s = mc.state_values + b
    J = mc.P * np.exp((1 - gamma) * s)
    return beta * np.max(np.abs(eigvals(J)))

lucas_tree_spec_rad(n=200)

b_vals = np.linspace(0.0, 0.5, 500)

get_ipython().run_cell_magic('timeit', '', '[lucas_tree_spec_rad(b=b) for b in b_vals]')

get_ipython().run_cell_magic('timeit', '', 'Parallel(n_jobs=4)(delayed(lucas_tree_spec_rad)(b=b) for b in b_vals)')



