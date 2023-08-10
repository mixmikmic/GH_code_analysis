get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import theano
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context('notebook')
np.random.seed(12345)
rc = {'xtick.labelsize': 10, 'ytick.labelsize': 10, 'axes.labelsize': 10, 'font.size': 10, 
      'legend.fontsize': 12.0, 'axes.titlesize': 10, "figure.figsize": [14, 6]}
sns.set(rc = rc)
sns.set_style("whitegrid")

data = pd.read_csv("https://raw.githubusercontent.com/pymc-devs/pymc3/master/pymc3/examples/data/radon.csv")
data.head()

data.shape

data.columns

radon_data = data[["floor", "log_radon", "county"]]
radon_data.head()

data["log_radon"].hist()

X = np.array(data["floor"])
y = np.array(data["log_radon"])

X[:10]

y[:10]

with pm.Model() as complete_pooled_model:
    
    # specify the priors
    slope_group = pm.Normal("slope_group", mu = 0, sd = 5)
    intercept = pm.Normal("intercept", mu = 0, sd = 5)
    error_obs = pm.HalfNormal("error_obs", sd = 5)
    
    # specify the mean function
    mean_group = intercept + slope_group * X
    
    # specify the likelihood of the data
    obs = pm.Normal("obs", mu = mean_group, sd = error_obs, observed = y)

with pm.Model() as hierarchical_model:
    
    # specify priori
    mean_group = pm.Normal("mean_group", mu = 0, sd = 5)
    sd_mean_house = pm.HalfNormal("error_mean_house", sd = 5)
    sd_obs = pm.HalfNormal("sd_obs", sd = 5)
    
    slope_mean_house = pm.Normal("slope_mean_house", mu = 0, sd = 5, shape = len(y))
    intercept_mean_house = pm.Normal("intercept_mean_house", mu = 0, sd = 5, shape = len(y))
    
    # specify the mean functions
    mean_house = intercept_mean_house + slope_mean_house * X
    
    # specify the data likelihood
    obs = pm.Normal("obs", mu = mean_house, sd = sd_obs, observed = y, shape = len(y))    

with complete_pooled_model:
    step = pm.NUTS(target_accept = 0.9)
    posterior_complete_pooled = pm.sample(draws = 500, njobs = 2)

pm.traceplot(posterior_complete_pooled)

pm.gelman_rubin(posterior_complete_pooled)

pm.energyplot(posterior_complete_pooled)

pm.forestplot(posterior_complete_pooled)

pm.summary(posterior_complete_pooled)

pm.plot_posterior(posterior_complete_pooled)



