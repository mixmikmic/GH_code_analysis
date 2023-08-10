get_ipython().magic('matplotlib inline')
get_ipython().magic('config InlineBackend.figure_format = "retina"')

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from scipy import stats as st

from utils import plt, sns

df = pd.read_csv("../data/TwoGroupIQ.csv")
# Only work with the "Smart Drug" group
smart = df[df.Group == "Smart Drug"]

print len(smart)
smart.head()

with pm.Model() as model:
    # Priors
    sigma = pm.HalfCauchy("sigma", 25)  # Gelman 2006
    mu = pm.Normal("mu", mu=0, tau=1E-8)
    nu = pm.Exponential("nu", 1.0 / 29)   # Taken from the book
    # Likelihood
    y = pm.StudentT("y", nu=nu, mu=mu, sd=sigma, observed=smart.Score)
    # Sample
    trace = pm.sample(draws=6000, tune=1000, chain=3)
    
burn_in = 2000
trace = trace[burn_in:]

print(pm.df_summary(trace))
pm.traceplot(trace)

pm.plot_posterior(trace, point_estimate="median")

df["label"] = (df.Group == "Smart Drug").astype(int)
df.head()

n_groups = 2
with pm.Model() as model_two:
    # Priors
    sigma = pm.HalfCauchy("sigma", 25, shape=n_groups)
    mu = pm.Normal("mu", mu=0, tau=1E-8, shape=n_groups)
    nu = pm.Gamma("nu", 2, .1)  # Recommendation from Stan docs
    # Likelihood
    y = pm.StudentT("y", nu=nu, mu=mu[df.label], sd=sigma[df.label], observed=df.Score)
    # Sample
    trace = pm.sample(draws=6000, tune=1000, chain=3)
    
burn_in = 2000
trace = trace[burn_in:]

print(pm.df_summary(trace))
pm.traceplot(trace)

varnames = ["mu", "sigma", "nu"]
pm.plot_posterior(trace, varnames=varnames)

