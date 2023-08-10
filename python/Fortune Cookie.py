get_ipython().magic('matplotlib inline')
get_ipython().magic('config InlineBackend.figure_format = "retina"')

import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import stats as st

from utils import plt, sns

data = np.array([
    [35, 26, 56, 10, 32, 52],
    [26, 38, 2, 23, 27, 50],
    [11, 34, 24, 49, 2, 45],
    [14, 1, 55, 19, 32, 45],

    # Sampled these from his Google image search
    [37, 53, 1, 17, 32, 42],
    [26, 55, 25, 51, 11, 27],
])

with pm.Model() as model:
    # Prior
    digit = pm.DiscreteUniform("digit", lower=1, upper=1000)
    # Likelihood
    y = pm.DiscreteUniform("y", lower=1, upper=digit, observed=data.ravel())
    # Need more draws with the Metropolis sampler
    trace = pm.sample(draws=20000, njobs=4, chain=4)
    
burn_in = 10000
trace = trace[burn_in:]

print(pm.summary(trace))
pm.traceplot(trace)

pm.plot_posterior(trace, point_estimate="median")

pm.diagnostics.gelman_rubin(trace)

with pm.Model() as model:
    # Prior
    digit = pm.Uniform("digit", lower=max(data.ravel()), upper=1000)
    # Likelihood
    y = pm.Uniform("y", lower=1, upper=digit, observed=data.ravel())
    # Sample
    trace = pm.sample(draws=6000, njobs=4, chain=4)

burn_in = 2000
trace = trace[burn_in:]

print(pm.summary(trace))
pm.traceplot(trace)

pm.plot_posterior(trace, point_estimate="median")

