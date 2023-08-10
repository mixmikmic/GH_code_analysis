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
    beta_0 = pm.Normal("beta_0", mu=0, sd=2)
    beta_j = pm.Normal("beta_j", mu=0, sd=2, shape=n_cols)
    delta_j = pm.Bernoulli("delta_j", 0.5, shape=n_cols)

    theta = beta_0 + tt.dot(delta_j * beta_j, zx.T)

    nu = pm.Gamma("nu", 2, .1)          # Stan docs recommendation
    sigma = pm.HalfCauchy("sigma", 25)  # Gelman 2006

    # Likelihood
    y = pm.StudentT("y", nu=nu, mu=theta, sd=sigma, observed=zy)
    
    # Sample
    step_1 = pm.BinaryGibbsMetropolis([delta_j])
    step_2 = pm.Metropolis([beta_0, beta_j, theta, nu, sigma, y])
    trace = pm.sample(10000, [step_1, step_2])
    
burn_in = 5000
trace = trace[burn_in:]

print(pm.df_summary(trace))
pm.traceplot(trace)

pm.plot_posterior(trace, point_estimate="median")

normality = np.log10(trace["nu"])
ax = pm.plot_posterior(normality, point_estimate="median")
ax.set_title("Normality")
ax.set_xlabel(r"log10($\nu$)")

with pm.Model() as sd1_model:
    # Priors
    beta_0 = pm.Normal("beta_0", mu=0, sd=1)
    beta_j = pm.Normal("beta_j", mu=0, sd=1, shape=n_cols)
    delta_j = pm.Bernoulli("delta_j", 0.5, shape=n_cols)
    
    theta = beta_0 + tt.dot(delta_j * beta_j, zx.T)
    
    nu = pm.Gamma("nu", 2, .01)
    sigma = pm.HalfCauchy("sigma", 25)
    
    # Likelihood
    y = pm.StudentT("y", nu=nu, mu=theta, sd=sigma, observed=zy)
    
    # Sample
    trace = pm.sample(5000)

burn_in = 2000
trace = trace[burn_in:]

pm.traceplot(trace)

pm.plot_posterior(trace, point_estimate="median")

with pm.Model() as sd10_model:
    beta_0 = pm.Normal("beta_0", mu=0, sd=10)
    beta_j = pm.Normal("beta_j", mu=0, sd=10, shape=n_cols)
    delta_j = pm.Bernoulli("delta_j", 0.5, shape=n_cols)
    
    theta = beta_0 + tt.dot(delta_j * beta_j, zx.T)
    
    nu = pm.Gamma("nu", 2, .1)
    sigma = pm.HalfCauchy("sigma", 25)
    
    # Likelihood
    y = pm.StudentT("y", nu=nu, mu=theta, sd=sigma, observed=zy)
    
    # Sample
    trace = pm.sample(5000)

burn_in = 2000
trace = trace[burn_in:]

pm.traceplot(trace)

pm.plot_posterior(trace, point_estimate="median")

X = df[["Spend", "PrcntTake", "StuTeaRat", "Salary"]]
X.corr()

zx_shrink = ((X - X.mean().values) / X.std().values).values
zx_shrink[:5]

n_shrink = 4
with pm.Model() as shrink_model:
    beta_0 = pm.Normal("beta_0", mu=0, sd=2)
    
    sigma_b = pm.HalfCauchy("sigma_b", 25)
    beta_j = pm.StudentT("beta_j", nu=1, mu=0, sd=sigma_b, shape=n_shrink)
    delta_j = pm.Bernoulli("delta_j", .5, shape=n_shrink)
    
    theta = beta_0 + tt.dot(delta_j * beta_j, zx_shrink.T)
    nu = pm.Exponential("nu", 1.0 / 30)  # From the book
    sigma = pm.HalfCauchy("sigma", 25)
    
    # Likelihood
    y = pm.StudentT("y", nu=nu, mu=theta, sd=sigma, observed=zy)
    # Sample
    trace = pm.sample(5000)
    
burn_in = 2000
trace = trace[burn_in:]

pm.traceplot(trace)

pm.plot_posterior(trace)

