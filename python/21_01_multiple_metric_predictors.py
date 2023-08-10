get_ipython().magic('matplotlib inline')
get_ipython().magic('config InlineBackend.figure_format = "retina"')

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from scipy import stats as st

from utils import plt, sns

df = pd.read_csv('data/Salary.csv', usecols=[0,3,5])
df["Org"] = df.Org.astype("category")
df["Pos"] = df.Pos.astype("category")
print df.info()
df.head()

df.groupby("Pos").apply(lambda x: x.head(2))

X1 = df.Pos.cat.codes.values
n_X1 = len(df.Pos.cat.categories)

X2 = df.Org.cat.codes.values
n_X2 = len(df.Org.cat.categories)

y = df.Salary
y_mean = y.mean()
y_std = y.std()
y_shape, y_rate = gamma_shape_rate(y_std/2, 2 * y_std)

n_cols = 2
with pm.Model() as model:
    # Priors
    sigma_0 = pm.HalfCauchy("sigma_0", 25)
    beta_0 = pm.Normal("beta_0", mu=0, sd=sigma_0)
    
    sigma_1 = pm.HalfCauchy("sigma_1", 25)
    beta_1 = pm.Normal("beta_1", mu=0, sd=sigma_0, shape=n_cols)
    
    theta = pm.invlogit(beta_0 + beta_1[data])

    # Likelihood
    y = pm.Bernoulli("y", mu=theta, observed=data)

    # Sample
    trace = pm.sample(10000)
    
burn_in = 5000
trace = trace[burn_in:]

print(pm.df_summary(trace))
pm.traceplot(trace)

pm.forestplot(trace)

