get_ipython().magic('load ../data/melanoma_data.py')

get_ipython().magic('matplotlib inline')
import seaborn as sns; sns.set_context('notebook')
from pymc3 import Normal, Model, DensityDist, sample
from pymc3.math import log, exp

with Model() as melanoma_survival:

    # Convert censoring indicators to indicators for failure event
    failure = (censored==0).astype(int)

    # Parameters (intercept and treatment effect) for survival rate
    β = Normal('β', mu=0.0, sd=1e5, shape=2)

    # Survival rates, as a function of treatment
    λ = exp(β[0] + β[1]*treat)
    
    # Survival likelihood, accounting for censoring
    def logp(failure, value):
        return (failure * log(λ) - λ * value).sum()

    x = DensityDist('x', logp, observed={'failure':failure, 'value':t})

with melanoma_survival:
    trace = sample(1000, init=None)

from pymc3 import traceplot

traceplot(trace[500:], varnames=['β']);

import numpy as np
import matplotlib.pyplot as plt

disasters_data = np.array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                         3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                         2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                         1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                         0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                         3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                         0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

n_years = len(disasters_data)

plt.figure(figsize=(12.5, 3.5))
plt.bar(np.arange(1851, 1962), disasters_data, color="#348ABD")
plt.xlabel("Year")
plt.ylabel("Disasters")
plt.title("UK coal mining disasters, 1851-1962")
plt.xlim(1851, 1962);

from pymc3 import DiscreteUniform

with Model() as disaster_model:

    switchpoint = DiscreteUniform('switchpoint', lower=0, upper=n_years)

oops = DiscreteUniform('oops', lower=0, upper=10)

x = DiscreteUniform.dist(lower=0, upper=100)
x

from pymc3 import distributions
distributions.__all__

from pymc3 import Exponential

with disaster_model:
    
    early_mean = Exponential('early_mean', 1)
    late_mean = Exponential('late_mean', 1)

from pymc3.math import switch

with disaster_model:
    
    rate = switch(switchpoint >= np.arange(n_years), early_mean, late_mean)

from pymc3 import Poisson

with disaster_model:
    
    disasters = Poisson('disasters', mu=rate, observed=disasters_data)

disaster_model.vars

disaster_model.deterministics

from pymc3 import Deterministic

with disaster_model:
    
    rate = Deterministic('rate', switch(switchpoint >= np.arange(n_years), early_mean, late_mean))

disaster_model.deterministics

disasters.dtype

early_mean.init_value

plt.hist(switchpoint.random(size=1000));

early_mean.transformed

switchpoint.distribution

type(switchpoint)

type(disasters)

switchpoint.logp({'switchpoint':55, 'early_mean_log_':1, 'late_mean_log_':1})

disasters.logp({'switchpoint':55, 'early_mean_log_':1, 'late_mean_log_':1})

with disaster_model:
    trace = sample(2000, init=None)

trace

help(sample)

from pymc3 import Slice

with disaster_model:
    step_trace = sample(1000, step=Slice(vars=[early_mean, late_mean]))

trace['late_mean']

trace['late_mean', -5:]

trace['late_mean', ::10]

plt.hist(trace['late_mean']);

from pymc3 import traceplot

traceplot(trace[500:], varnames=['early_mean', 'late_mean', 'switchpoint']);

from pymc3 import summary

summary(trace[500:], varnames=['early_mean', 'late_mean'])

