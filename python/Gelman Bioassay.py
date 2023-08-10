get_ipython().magic('matplotlib inline')
get_ipython().magic('config InlineBackend.figure_format = "retina"')

import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.stats as st

from utils import plt, sns

# Example data
patients = np.array([5, 5, 5, 5])
deaths = np.array([0, 1, 3, 5])
dose = np.array([-0.86, -0.3, -0.05, 0.73])

with pm.Model() as model:
    # Prior
    sigma_a = pm.HalfCauchy("sigma_a", 2.5)
    alpha = pm.Normal("alpha", mu=0, sd=sigma_a)

    sigma_b = pm.HalfCauchy("sigma_b", 2.5)
    beta = pm.Normal("beta", mu=0, sd=sigma_b)

    theta = pm.math.invlogit(alpha + beta * dose)
    probability = pm.Deterministic("probability", theta)

    # Likelihood
    y = pm.Binomial("y", n=patients, p=probability, observed=deaths)

    # Sample
    trace = pm.sample(draws=5000, njobs=4, chain=4)

burn_in = 2000
trace = trace[burn_in:]

print(pm.df_summary(trace))
pm.traceplot(trace)

pm.plot_posterior(trace, point_estimate="median")

pm.diagnostics.gelman_rubin(trace)

