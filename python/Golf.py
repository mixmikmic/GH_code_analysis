get_ipython().magic('matplotlib inline')
get_ipython().magic('config InlineBackend.figure_format = "retina"')

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from scipy import stats as st

from utils import plt, sns

df = pd.read_csv("data/golf.csv")
df["per"] = (df.success / df.tries).round(3)
print len(df)
df.head(10)

sns.regplot(df.distance, df.success/df.tries, fit_reg=False)
plt.title("Chance of Making Putt")
plt.ylabel("Successful Putt")
plt.xlabel("Distance")
plt.xlim(0, 20)

def probit_phi(x):
    """ Probit transform assuming 0 mean and 1 sd """
    # http://stackoverflow.com/questions/21849494/
    mu, sd = 0, 1
    return 0.5 * (1 + tt.erf((x - mu) / (sd * tt.sqrt(2))))

tin_cup_radius = (4.25 - 1.68) / 2.0
putt_chance = np.sin(tin_cup_radius / df.distance.values)

with pm.Model() as model:
    # Priors
    sigma = pm.HalfCauchy("sigma", 2.5)
    theta = probit_phi(1.0 / sigma * putt_chance)
    probability = 2.0 * theta - 1.0

    # Likelihood
    y = pm.Binomial("y", df.tries.values, probability, shape=19, observed=df.success.values)
    
    # Sample
    trace = pm.sample(draws=10000, tune=5000, njobs=4, chain=4)
    
burn_in = 5000
trace = trace[burn_in:]

print(pm.df_summary(trace))
pm.traceplot(trace)

pm.plot_posterior(trace, point_estimate="median")

pm.diagnostics.gelman_rubin(trace)

ppc = pm.sample_ppc(trace, model=model)

putt = pd.Series(ppc["y"][:, 0])
print putt.describe()
sns.kdeplot(ppc["y"][:, 0], shade=True)
plt.title("1-Foot Putt KDE")

putt = pd.Series(ppc["y"][:, 0])
print putt.describe()
sns.kdeplot(ppc["y"][:, 0], cumulative=True, shade=True)
plt.title("1-Foot Putt CDF")

