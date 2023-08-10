get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import theano
import pymc3 as pm
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context('notebook')
np.random.seed(12345)
rc = {'xtick.labelsize': 10, 'ytick.labelsize': 10, 'axes.labelsize': 10, 'font.size': 10, 
      'legend.fontsize': 12.0, 'axes.titlesize': 10, "figure.figsize": [14, 6]}
sns.set(rc = rc)
sns.set_style("whitegrid")

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

with pm.Model() as lm:
    
    # specify the priors
    alpha = pm.Normal("alpha", mu = 0, sd = 1)
    beta = pm.Normal("beta", mu = 0, sd = 1, shape = 2)
    sigma = pm.HalfCauchy("sigma", beta = 1)
    
    # specify the expected value mu
    mean = alpha + beta[0]*X1 + beta[1]*X2
    
    # specifiy the likelhood
    obs = pm.Normal("obs", mu = mean, sd = sigma, observed = Y)

with lm:
    # obtain starting values via MAP
    start = pm.find_MAP()
    
    step = pm.NUTS(target_accept = 0.99)
    posterior = pm.sample(draws = 5000, njobs = 4, tune = 1000, start = start, step = step)

pm.traceplot(posterior)
fig1 = plt.gcf()

pm.gelman_rubin(posterior)

pm.energyplot(posterior)
fig2 = plt.gcf()

pm.forestplot(posterior, varnames = ["beta"])
fig3 = plt.gcf()

pm.forestplot(posterior, varnames = ["alpha", "sigma"])

pm.df_summary(posterior)

help(pd.DataFrame.to_dict)

pm.df_summary(posterior).to_dict("list")

pm.plot_posterior(posterior)
fig4 = plt.gcf()

# save the posterior to the file 'posterior_lm.pkl' for later use
with open('posterior_lm.pkl', 'wb') as f:
    pickle.dump(posterior, f, protocol=pickle.HIGHEST_PROTOCOL)

# load it at some future point
with open('posterior_lm.pkl', 'rb') as f:
    posterior = pickle.load(f)

type(posterior)

posterior.varnames

posterior.get_sampler_stats

posterior.get_values(varname="beta")

import plotly as py
py.offline.init_notebook_mode(connected=True)

pm.traceplot(posterior)
fig1 = plt.gcf()
py.offline.iplot_mpl(fig1)
#py.tools.mpl_to_plotly(fig1)

py.offline.iplot_mpl(fig2)

py.offline.iplot_mpl(fig3)

py.offline.iplot_mpl(fig4)

