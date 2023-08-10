#imports

import quandl
import pandas as pd

import numpy as np

# visualization
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
get_ipython().magic('matplotlib inline')

# data pull from Quandl
reg = quandl.get("CHRIS/CME_RK2")
gas_m1 = quandl.get("CHRIS/CME_SG2")
gas_m2 = quandl.get("CHRIS/CME_SG3")

# pandas data processing
regrade = pd.DataFrame(reg)
regrade = regrade.dropna()
regrade = regrade[['Settle', 'Open Interest']]
regrade.columns = [['price_reg', 'OI_reg']]

gasoil_m1 = pd.DataFrame(gas_m1)
gasoil_m1 = gasoil_m1.dropna()
gasoil_m1 = gasoil_m1[['Settle', 'Open Interest']]
gasoil_m1.columns = [['price_m1', 'OI_m1']]

gasoil_m2 = pd.DataFrame(gas_m2)
gasoil_m2 = gasoil_m2.dropna()
gasoil_m2 = gasoil_m2[['Settle', 'Open Interest']]
gasoil_m2.columns = [['price_m2', 'OI_m2']]

gasoil = pd.concat([gasoil_m1, gasoil_m2], axis = 1)
gasoil['spread_m1m2'] = gasoil.price_m1 - gasoil.price_m2

df = pd.concat([gasoil, regrade], axis = 1)

# keep only business days
ts_days = pd.to_datetime(df.index.date)
bdays = pd.bdate_range(start=df.index[0].date(), end=df.index[-1].date())
df = df[ts_days.isin(bdays)]
df = df.dropna()

# filter daily glitch moves of over 6 standard deviations
stdev_filter = 6
df['spread_return'] = (df.spread_m1m2 - df.spread_m1m2.shift(1).fillna(value=0))
df['spread_return'][0] = 0
df['reg_return'] = (df.price_reg - df.price_reg.shift(1).fillna(value=0))
df['reg_return'][0] = 0
for i in range (0, 2):
    df = df[df.spread_return.diff(1).map(lambda x: not(abs(x)>stdev_filter*np.std(df.spread_return)))]
    
for i in range (0, 2):
    df = df[df.reg_return.diff(1).map(lambda x: not(abs(x)>stdev_filter*np.std(df.reg_return)))]
df = df.dropna()

# visualize to check

plt.plot(df.index, df.price_reg, 'r', df.spread_m1m2, 'g')
plt.show()

# target = 1 if regrade is higher than 1.5 usd per bbl, 0 otherwise
df['target']= 0
df.loc[df.price_reg > 1.5, 'target'] = 1

df.head(5)

import pymc3 as pm

import theano.tensor as tt

plt.scatter(df.spread_m1m2, df.target, s=75, color="k",
            alpha=0.5)
plt.yticks([0, 1])
plt.ylabel("Regrade over 1 usd per bbl?")
plt.xlabel("gasoil in usd per bbl")
plt.title("Jet kero regrade vs gasoil flat price");

# pymc3 model

levels = df.spread_m1m2
D = df.target # defect or not?

#notice the`value` here. We explain why below.
with pm.Model() as model:
    beta = pm.Normal("beta", mu=0, tau=0.001, testval=0)
    alpha = pm.Normal("alpha", mu=0, tau=0.001, testval=0)
    p = pm.Deterministic("p", 1.0/(1. + tt.exp(beta*levels + alpha)))

# connect the probabilities in `p` with our observations through a
# Bernoulli random variable.
with model:
    observed = pm.Bernoulli("bernoulli_obs", p, observed=D)
    
    # Mysterious code to be explained in Chapter 3
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(120000, step=step, start=start)
    burned_trace = trace[100000::2]

alpha_samples = burned_trace["alpha"][:, None]  # best to make them 1d
beta_samples = burned_trace["beta"][:, None]

figsize(12.5, 6)

#histogram of the samples:
plt.subplot(211)
plt.title(r"Posterior distributions of the variables $\alpha, \beta$")
plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\beta$", color="#7A68A6", normed=True)
plt.legend()

plt.subplot(212)
plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\alpha$", color="#A60628", normed=True)
plt.legend();

def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))

t = np.linspace(levels.min() - 10, levels.max()+10, 50)[:, None]
p_t = logistic(t.T, beta_samples, alpha_samples)

mean_prob_t = p_t.mean(axis=0)

figsize(12.5, 8)

plt.plot(t, mean_prob_t, lw=3, label="average posterior \nprobability of defect")
plt.plot(t, p_t[0, :], ls="--", label="realization from posterior")
plt.plot(t, p_t[-2, :], ls="--", label="realization from posterior")
plt.scatter(levels, D, color="k", s=50, alpha=0.5)
plt.title("Posterior expected value of probability of over 1.5 regrade; plus realizations")
plt.legend(loc="lower left")
plt.ylim(-0.1, 1.1)
plt.xlim(t.min(), t.max())
plt.ylabel("probability")
plt.xlabel("gasoil front spread ");

from scipy.stats.mstats import mquantiles

# vectorized bottom and top 2.5% quantiles for "confidence interval"
qs = mquantiles(p_t, [0.025, 0.975], axis=0)
plt.fill_between(t[:, 0], *qs, alpha=0.7,
                 color="#ffcc19")

plt.plot(t[:, 0], qs[0], label="95% CI", color="#7A68A6", alpha=0.7)
plt.plot(t[:, 0], qs[-1], label="95% CI", color="#7A68A6", alpha=0.7)

plt.plot(t, mean_prob_t, lw=1, ls="--", color="k",
         label="average posterior \nprobability of high regrade")

plt.xlim(t.min(), t.max())
plt.ylim(-0.02, 1.02)
plt.legend(loc="lower left")
plt.scatter(levels, D, color="#111199", s=50, alpha=0.5)
plt.xlabel("gasoil spread m1m2, $usd per bbl$")

plt.ylabel("probability estimate")
plt.title("Posterior probability estimates of high regrade given gasoil timespread. $usd per bbl$");



