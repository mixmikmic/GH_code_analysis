get_ipython().magic('matplotlib inline')
get_ipython().magic('config InlineBackend.figure_format = "retina"')

import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import stats as st

from utils import plt, sns

# Sandwich data is in ounces...
cols = ["female", "deli", "sandwich"]
df = pd.DataFrame([
    # Male sandwiches
    [0, 1, 10.6],
    [0, 2, 22.2],
    [0, 3, 12.0],
    [0, 4, 11.7],
    # Female sandwiches
    [1, 1, 10.9],
    [1, 2, 25.1],
    [1, 3, 10.8],
    [1, 4, 13.7],
], columns=cols)
df.head()

with pm.Model() as model:
    # Custom Distributions
    BoundedNormal = pm.Bound(pm.Normal, lower=0)
    
    # Priors
    intercept = pm.Normal("intercept", 0, 10)

    beta_mu = pm.Normal("beta_mu", 0, 10)
    beta_sigma = pm.HalfCauchy("beta_sigma", 2.5)  # Gelman 2006
    beta_nu = pm.Gamma("beta_nu", 2, 0.1)          # From the `Stan` docs
    beta = pm.StudentT("beta", nu=beta_nu, mu=beta_mu, sd=beta_sigma, shape=2)

    theta = intercept + beta[df.female.values]
    sigma = pm.HalfCauchy("sigma", 2.5)

    # Likelihood
    y = BoundedNormal("y", mu=theta, sd=sigma, observed=df.sandwich.values)

    # Sample
    trace = pm.sample(draws=5000, njobs=4, chain=4)

burn_in = 1000
trace = trace[burn_in:]

print(pm.df_summary(trace))
pm.traceplot(trace)

pm.plot_posterior(trace, point_estimate="median")

pm.diagnostics.gelman_rubin(trace)

ppc = pm.sample_ppc(trace, samples=2000, model=model)

print(len(ppc["y"]))
ppc

male = pd.Series(ppc["y"][:, :4].ravel())
female = pd.Series(ppc["y"][:, 4:].ravel())

print male.head()
print("\n")
print female.head()

print(male.describe())
print(female.describe())

sns.kdeplot(male, shade=True, label="Male")
sns.kdeplot(female, shade=True, label="Female")

