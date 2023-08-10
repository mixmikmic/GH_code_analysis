get_ipython().magic('matplotlib inline')
get_ipython().magic('config InlineBackend.figure_format = "retina"')

import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import stats as st

from utils import plt, sns

df = pd.read_csv("../data/earnings.csv")
df["log_earnings"] = np.log(df.earnings)
df["male"] = 2 - df.sex

df.head()

with pm.Model() as model:
    # Priors
    intercept = pm.Normal("intercept", mu=0, sd=10)
    height = pm.Normal("height", mu=0, sd=10)
    sex = pm.Normal("sex", mu=0, sd=10)

    sigma = pm.HalfCauchy("sigma", 2.5)

    theta = intercept                            + height * df.height.values            + sex * df.sex.values

    # Likelihood
    log_earnings = pm.Normal("log_earnings", mu=theta, sd=sigma,
                             observed=df.log_earnings.values)
    # Sample
    trace = pm.sample(draws=6000, njobs=4, chain=4)

burn_in = 2000
trace = trace[burn_in:]

print(pm.summary(trace))
pm.traceplot(trace)

pm.plot_posterior(trace, point_estimate="median")

pm.diagnostics.gelman_rubin(trace)

