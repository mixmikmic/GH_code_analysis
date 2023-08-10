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
df = df[df.Group == "Smart Drug"]

print len(df)
df.head()

with pm.Model() as model:
    # Priors
    sigma = pm.Uniform("sigma", df.Score.std() / 1000, df.Score.std() * 1000)
    mu = pm.Normal("mu", mu=0, tau=1E-8)
    # Likelihood
    y = pm.Normal("y", mu=mu, sd=sigma, observed=df.Score)
    # Sample
    trace = pm.sample(draws=5000, tune=1000, chain=3)
    
burn_in = 1000
trace = trace[burn_in:]

print pm.df_summary(trace)
pm.traceplot(trace)

pm.plot_posterior(trace, point_estimate="mode")

with pm.Model() as robust_model:
    # Priors
    sigma = pm.HalfCauchy("sigma", 25)
    mu = pm.Normal("mu", mu=0, tau=1E-8)
    nu = pm.Gamma("nu", 2, .1)  # Taken from `Stan` docs
    # Likelihood
    y = pm.StudentT("y", nu=nu, mu=mu, sd=sigma, observed=df.Score)
    effect_size = (mu - 100) / sigma
    normality = tt.log10(nu)
    # Sample
    trace = pm.sample(draws=5000, tune=1000, chain=3)
    
burn_in = 1000
trace = trace[burn_in:]

print pm.df_summary(trace)
pm.traceplot(trace)

pm.plot_posterior(trace)

