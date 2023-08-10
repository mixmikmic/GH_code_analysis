get_ipython().magic('matplotlib inline')
get_ipython().magic('config InlineBackend.figure_format = "retina"')

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from scipy import stats as st

from utils import plt, sns

df = pd.read_csv("../data/Guber1999data.csv")
print len(df)
df.head()

X = df[["Spend", "PrcntTake"]]
X["SpendXPrcnt"] = X.Spend * X.PrcntTake
y = df["SATT"]

# A little bit convoluted with the two columns
zx = ((X - X.mean().values) / X.std().values).values
zy = (y - y.mean()) / y.std()

n_cols = 3
with pm.Model() as model:
    # Priors
    beta_0 = pm.Normal("beta_0", mu=0, tau=1E-8)
    beta_1 = pm.Normal("beta_1", mu=0, tau=1E-8, shape=n_cols)
    theta = beta_0 + tt.dot(beta_1, zx.T)

    sigma = pm.HalfCauchy("sigma", 5)    # Gelman 2006
    nu = pm.Exponential("nu", 1.0 / 29)   # Taken from the book
    # Likelihood
    y = pm.StudentT("y", nu=nu, mu=theta, sd=sigma, observed=zy)
    # Sample
    trace = pm.sample(draws=6000, tune=1000, njobs=3, chain=3)
    
burn_in = 2000
trace = trace[burn_in:]

print(pm.df_summary(trace))
pm.traceplot(trace)

pm.plot_posterior(trace, point_estimate="median")

normality = np.log10(trace["nu"])
ax = pm.plot_posterior(normality, point_estimate="median")
ax.set_title("Normality")
ax.set_xlabel(r"log10($\nu$)")

# PyMC's Rhat
pm.diagnostics.gelman_rubin(trace)

