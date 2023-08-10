get_ipython().magic('matplotlib inline')
get_ipython().magic('config InlineBackend.figure_format = "retina"')

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from scipy import stats as st

import matplotlib.pyplot as plt
import seaborn as sns

# Kruschke's light blue color
sns.set_palette(["#87ceeb"])
sns.set_context("talk")

df = pd.read_csv("data/18_03_shrinkage.csv")
print len(df)
df.head()

cols = ["Spend", "PrcntTake", "xRand1", "xRand2", "xRand3", "xRand4", "xRand5",
        "xRand6", "xRand7", "xRand8", "xRand9", "xRand10", "xRand11", "xRand12"]
X = df[cols]
y = df["SATT"]

# A little bit convoluted with the two columns
zx = ((X - X.mean().values) / X.std().values).values
zy = (y - y.mean()) / y.std()

n_cols = len(cols)

with pm.Model() as model:
    # Priors
    beta_0 = pm.Normal("beta_0", mu=0, tau=1E-8)

    nu_b = pm.Gamma("nu_b", 2, .1)          # Stan docs recommendation
    sigma_b = pm.HalfCauchy("sigma_b", 25)  # Gelman 2006
    beta_1 = pm.StudentT("beta_1", nu=nu_b, mu=0, sd=sigma_b, shape=n_cols)

    theta = beta_0 + tt.dot(beta_1, zx.T)
    sigma = pm.HalfCauchy("sigma", 25)
    nu = pm.Gamma("nu", 2, .1)

    # Likelihood
    y = pm.StudentT("y", nu=nu, mu=theta, sd=sigma, observed=zy)
    
    # Sample
    step = pm.Metropolis()
    trace = pm.sample(20000, step)
    
burn_in = 10000
trace = trace[burn_in:]

print(pm.df_summary(trace))
pm.traceplot(trace)

pm.plot_posterior(trace, point_estimate="median")

normality = np.log10(trace["nu"])
ax = pm.plot_posterior(normality, point_estimate="median")
ax.set_title("Normality")
ax.set_xlabel(r"log10($\nu$)")

