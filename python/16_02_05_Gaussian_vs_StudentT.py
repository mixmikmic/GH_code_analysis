get_ipython().magic('matplotlib inline')
get_ipython().magic('config InlineBackend.figure_format = "retina"')

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from scipy import stats as st

from utils import plt, sns

y = [-2, -1, 0, 1, 2, 15]
print y

with pm.Model() as model:
    # Priors
    sigma_norm = pm.Uniform("sigma_norm", 0, 100)
    mu_norm = pm.Normal("mu_norm", mu=0, tau=1E-8)

    sigma_t = pm.Uniform("sigma_t", 0, 100)
    mu_t = pm.Normal("mu_t", mu=0, tau=1E-8)
    nu = pm.Exponential("nu", 1.0 / len(y))
    # Likelihood
    y_norm = pm.Normal("y_norm", mu=mu_norm, sd=sigma_norm, observed=y)
    y_t = pm.StudentT("y_t", nu=nu, mu=mu_t, sd=sigma_t, observed=y)
    # Sample
    trace = pm.sample(draws=5000, tune=1000, njobs=3)
    
burn_in = 1000

pm.df_summary(trace[burn_in:])

pm.traceplot(trace[burn_in:])

pm.plot_posterior(trace[burn_in:], point_estimate="mode")

