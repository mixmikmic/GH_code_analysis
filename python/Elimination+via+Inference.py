import pymc3 as pm
import numpy as np
import matplotlib.pylab as plt 
import corner
get_ipython().magic('matplotlib inline')

w1, w2, w3, w4 = 2,4,5,6

with pm.Model() as basic_model:
    mu1 = pm.Normal('mu1', mu=5, sd=2)
    mu2 = pm.Normal('mu2', mu=5, sd=2)
    mu3 = pm.Normal('mu3', mu=5, sd=2)
    mu4 = pm.Normal('mu4', mu=5, sd=2)

    # define the relationship between observed data and input
    s1 = pm.Normal('s1', mu=mu1 + mu2, sd=0.01, observed=w1+w2)
    s2 = pm.Normal('s2', mu=mu1 + mu2 + mu3, sd=0.01, observed=w1+w2+w3)
    s3 = pm.Normal('s3', mu=mu3 + mu4, sd=0.01, observed=w3+w4)

with basic_model:
    trace = pm.sample(5000, model=basic_model, step=pm.Metropolis(), njobs = 8)
    _ = pm.traceplot(trace[1000:])

def trace_to_npdata(trace, model, start = 1000):
    var_names = sorted([_ for _ in model.named_vars.keys() if 'mu' in _])
    data = np.array([trace[start:].get_values(_, combine=True) for _ in var_names])
    data = data.T
    return var_names, data

var_names, data = trace_to_npdata(trace, basic_model)

_ = corner.corner(data, 
                  fill_contours=False, 
                  plot_contours = False, 
                  plot_density = False,
                  labels = var_names)

with basic_model:
    v_params = pm.variational.advi(n=50000)
    trace = pm.variational.sample_vp(v_params, draws=50000)

_ = pm.traceplot(trace)

from functools import reduce
from operator import add
import random

random.seed(42)

num_variables = 10 
num_sums = 6
num_var_in_sums = 6

with pm.Model() as set_model:
    variables = []
    # declaring unknowns: note that I'm now modelling the priors as uniforms
    for i in range(num_variables):
        rand_val = random.randint(1, 15)
        var_name = 'mu_{}'.format(i)
        variables.append([pm.Uniform(var_name, lower=1, upper=15), rand_val])
        print("{}={}".format(variables[i][0], variables[i][1]))

    sums = [] 
    # declarig sums 
    for j in range(num_sums):
        random.shuffle(variables)
        s = sum([_[1] for _ in variables[:num_var_in_sums]])
        rv = pm.Normal(name = 's{}'.format(j), 
                       mu = reduce(add, [_[0] for _ in variables[:num_var_in_sums]]), 
                       sd = 0.2, observed = s)
        print("+".join([_[0].name for _ in variables[:num_var_in_sums]]) + "=" + str(s))
        sums.append(rv)
        
    for i, _ in enumerate(variables):
        variables[i][0].name = variables[i][0].name + '=' + str(variables[i][1])

with set_model:
    trace = pm.sample(10000, init=None, step=pm.Metropolis(), njobs = 8)
    _ = pm.traceplot(trace[1000:])

var_names, data = trace_to_npdata(trace, set_model)

_ = corner.corner(data, 
                  fill_contours=False, 
                  plot_contours = False, 
                  plot_density = False,
                  labels = var_names)

with set_model:
    v_params = pm.variational.advi(n=50000)
    trace = pm.variational.sample_vp(v_params, draws=50000)
    _ = pm.traceplot(trace)



