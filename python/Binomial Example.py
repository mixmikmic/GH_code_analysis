get_ipython().magic('matplotlib inline')
get_ipython().magic('config InlineBackend.figure_format = "retina"')

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from scipy import stats as st

from utils import plt, sns

data = st.binom.rvs(n=1000, p=0.37, size=50)
print pd.Series(data).head()

sns.kdeplot(data, shade=True)
plt.title("Binomial Data")
plt.xlabel("Observed")

with pm.Model() as model:
    # Priors
    p = pm.Beta("p", alpha=2, beta=2)

    # Likelihood
    y = pm.Binomial("y", p=p, n=1000, observed=data)

    # Sample
    trace = pm.sample(draws=6000, tune=2000)
    
burn_in = 2000
trace = trace[burn_in:]

print(pm.summary(trace))
pm.traceplot(trace)

pm.plot_posterior(trace, point_estimate="median")

