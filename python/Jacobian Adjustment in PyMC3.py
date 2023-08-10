get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import seaborn as sns

y = np.asarray([0, 1, 0, 1, 1, 0, 0, 1, 0, 0])
N = len(y)
with pm.Model():
    # Set Transform to None so it won't use theta_log_
    theta = pm.Uniform('theta', lower=0, upper=1, transform=None)
    obs = pm.Bernoulli('obs', p=theta, observed=y)
    map_est = pm.find_MAP()
map_est

y = np.asarray([0, 1, 0, 1, 1, 0, 0, 1, 0, 0])
N = len(y)
with pm.Model():
    # Set Transform to None so it won't use theta_log_
    theta = pm.Uniform('theta', lower=0, upper=1)
    obs = pm.Bernoulli('obs', p=theta, observed=y)
    map_est = pm.find_MAP()

from scipy import special
(1 - 0) * special.expit(map_est["theta_interval__"]) + 0

with pm.Model():
    # Set Transform to None so it won't use theta_log_
    theta = pm.Uniform('theta', lower=0, upper=1)
    obs = pm.Bernoulli('obs', p=theta, observed=y)
    trace = pm.sample(3e3, tune=1000, njobs=4)
pm.plot_posterior(trace[1000:]);

with pm.Model():
    # Set Transform to None so it won't use theta_log_
    theta = pm.Uniform('theta', lower=0, upper=1, transform=None)
    obs = pm.Bernoulli('obs', p=theta, observed=y)
    trace = pm.sample(3e3, tune=1000, njobs=4, nuts_kwargs={'target_accept':.99})
pm.plot_posterior(trace[1000:]);

with pm.Model():
    alpha = pm.Flat('alpha')
    theta = pm.Deterministic('theta', pm.math.invlogit(alpha))
    obs = pm.Bernoulli('obs', p=theta, observed=y)
    trace = pm.sample(3e3, tune=1000, njobs=4)
pm.plot_posterior(trace[1000:]);

with pm.Model():
    alpha = pm.Flat('alpha')
    theta = pm.Deterministic('theta', pm.math.invlogit(alpha))
    
    def logp(value):
        return tt.sum(tt.sum(pm.Bernoulli.dist(p=theta).logp(value)) +                       tt.log(theta) + 
                      tt.log(1-theta))
    obs = pm.DensityDist('obs', logp, observed=y)
    
    trace = pm.sample(3e3, tune=1000, njobs=4)
pm.plot_posterior(trace[1000:]);

with pm.Model():
    alpha = pm.Flat('alpha')
    theta = pm.Deterministic('theta', pm.math.invlogit(alpha))
    obs = pm.Bernoulli('obs', p=theta, observed=y)
    constrain = pm.Potential('constrain', tt.log(theta)+tt.log(1-theta))
    trace = pm.sample(3e3, tune=1000, njobs=4)
pm.plot_posterior(trace[1000:]);

