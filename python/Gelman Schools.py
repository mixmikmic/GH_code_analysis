get_ipython().magic('matplotlib inline')
get_ipython().magic('config InlineBackend.figure_format = "retina"')

import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.stats as st

from utils import plt, sns

# Example data
data = np.array([28,  8, -3,  7, -1,  1, 18, 12])
sigma = np.array([15, 10, 16, 11,  9, 11, 10, 18])

with pm.Model() as model:
    # Priors
    alpha_mu = pm.Normal("alpha_mu", 0, 1)
    alpha_sigma = pm.HalfCauchy("alpha_sigma", 5)
    alpha = pm.Normal("alpha", mu=alpha_mu, sd=alpha_sigma)

    beta = pm.Normal("beta", mu=0, sd=10, shape=len(data))
    tau = pm.HalfCauchy("tau", 5)

    theta = alpha + beta * tau

    # Likelihood
    y = pm.Normal("y", mu=theta, sd=sigma, observed=data)

    # Sample
    trace = pm.sample(draws=5000, njobs=4, chain=4)

burn_in = 2000
trace = trace[burn_in:]

print(pm.df_summary(trace))
pm.traceplot(trace)

pm.plot_posterior(trace, point_estimate="median")

pm.diagnostics.gelman_rubin(trace)

