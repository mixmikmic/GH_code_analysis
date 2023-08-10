import numpy as np
import pandas as pd

import glam

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

np.random.seed(1)

m1 = glam.GLAM()

groups = ['high', 'low']

N = dict(high=5, low=3)

n_trials = 50
n_items = 5

gamma = dict(high=dict(mu=-0.3, sd=0.1),
             low=dict(mu=0.5, sd=0.1))

for group in groups:
    
    parameters = dict(v=np.clip(np.random.normal(loc=0.00007, scale=0.00001, size=N[group]), 0.00003, 0.00015),
                      s=np.clip(np.random.normal(loc=0.008, scale=0.001, size=N[group]), 0.005, 0.011),
                      gamma=np.clip(np.random.normal(loc=gamma[group]['mu'], scale=gamma[group]['sd'], size=N[group]), -1, 1),
                      tau=np.clip(np.random.normal(loc=0.9, scale=0.1, size=N[group]), 0.2, 2),
                      t0=np.zeros(N[group]))
    
    m1.simulate_group(kind='individual',
                      parameters=parameters,
                      n_individuals=N[group],
                      n_trials=n_trials,
                      n_items=n_items,
                      label=group)
m1.data.rename(columns=dict(condition='group'), inplace=True)

m1.make_model(kind='hierarchical', 
              depends_on=dict(gamma='group'),
              t0_val=0)
m1.fit(method='ADVI', n_advi=10000)

estimates = glam.utils.get_estimates(m1)
estimates

np.random.seed(1)

m2 = glam.GLAM()

groups = ['high', 'low']

N = 4

n_trials = 50
n_items = 5

gamma = dict(high=dict(mu=-0.3, sd=0.1),
             low=dict(mu=0.5, sd=0.1))

for group in groups:
    
    parameters = dict(v=np.clip(np.random.normal(loc=0.00007, scale=0.00001, size=N), 0.00003, 0.00015),
                      s=np.clip(np.random.normal(loc=0.008, scale=0.001, size=N), 0.005, 0.011),
                      gamma=np.clip(np.random.normal(loc=gamma[group]['mu'], scale=gamma[group]['sd'], size=N), -1, 1),
                      tau=np.clip(np.random.normal(loc=0.9, scale=0.1, size=N), 0.2, 2),
                      t0=np.zeros(N))
    
    m2.simulate_group(kind='individual',
                      parameters=parameters,
                      n_individuals=N,
                      individual_idx=np.arange(N),
                      n_trials=n_trials,
                      n_items=n_items,
                      label=group)
m2.data.rename(columns=dict(condition='group'), inplace=True)

m2.make_model(kind='hierarchical', 
              depends_on=dict(gamma='group'),
              t0_val=0)
m2.fit(method='ADVI', n_advi=10000)

estimates = glam.utils.get_estimates(m2)
estimates

np.random.seed(1)

m3 = glam.GLAM()

N = 5

n_trials = 50
n_items = 5

gamma = dict(high=dict(mu=-0.3, sd=0.1),
             low=dict(mu=0.5, sd=0.1))
v = dict(slow=dict(mu=0.00005, sd=0.00001),
         fast=dict(mu=0.00010, sd=0.00001))

for bias in ['high', 'low']:
    for speed in ['slow', 'fast']:
    
        parameters = dict(v=np.clip(np.random.normal(loc=v[speed]['mu'], scale=v[speed]['sd'], size=N), 0.00003, 0.00015),
                          s=np.clip(np.random.normal(loc=0.008, scale=0.001, size=N), 0.005, 0.011),
                          gamma=np.clip(np.random.normal(loc=gamma[bias]['mu'], scale=gamma[bias]['sd'], size=N), -1, 1),
                          tau=np.clip(np.random.normal(loc=0.9, scale=0.1, size=N), 0.2, 2),
                          t0=np.zeros(N))

        m3.simulate_group(kind='individual',
                          parameters=parameters,
                          n_individuals=N,
                          n_trials=n_trials,
                          n_items=n_items,
                          label=bias + '_' + speed)

m3.data['bias'], m3.data['speed'] = m3.data['condition'].str.split('_', 1).str

m3.make_model(kind='hierarchical', 
              depends_on=dict(gamma='bias',
                              v='speed'),
              t0_val=0)
m3.fit(method='ADVI', n_advi=10000)

estimates = glam.utils.get_estimates(m3)
estimates

np.random.seed(1)

m4 = glam.GLAM()

N = 5

n_trials = 50
n_items = 5

gamma = dict(high=dict(mu=-0.3, sd=0.1),
             low=dict(mu=0.5, sd=0.1))
v = dict(slow=dict(mu=0.00005, sd=0.00001),
         fast=dict(mu=0.00010, sd=0.00001))

for bias in ['high', 'low']:
    for speed in ['slow', 'fast']:
    
        parameters = dict(v=np.clip(np.random.normal(loc=v[speed]['mu'], scale=v[speed]['sd'], size=N), 0.00003, 0.00015),
                          s=np.clip(np.random.normal(loc=0.008, scale=0.001, size=N), 0.005, 0.011),
                          gamma=np.clip(np.random.normal(loc=gamma[bias]['mu'], scale=gamma[bias]['sd'], size=N), -1, 1),
                          tau=np.clip(np.random.normal(loc=0.9, scale=0.1, size=N), 0.2, 2),
                          t0=np.zeros(N))

        m4.simulate_group(kind='individual',
                          parameters=parameters,
                          individual_idx=np.arange(N),
                          n_individuals=N,
                          n_trials=n_trials,
                          n_items=n_items,
                          label=bias + '_' + speed)

m4.data['bias'], m4.data['speed'] = m4.data['condition'].str.split('_', 1).str

m4.make_model(kind='hierarchical', 
              depends_on=dict(gamma='bias',
                              v='speed'),
              t0_val=0)
m4.fit(method='ADVI', n_advi=10000)

estimates = glam.utils.get_estimates(m4)
estimates

np.random.seed(110101)

m5 = glam.GLAM()

groups = ['high', 'low']

N = dict(high=2, low=2)

n_trials = 50
n_items = 4

gamma = dict(high=dict(mu=-0.3, sd=0.1),
             low=dict(mu=0.5, sd=0.1))

for group in groups:

    parameters = dict(v=np.clip(np.random.normal(loc=0.00012, scale=0.00001, size=N[group]), 0.00003, 0.00015),
                      s=np.clip(np.random.normal(loc=0.01, scale=0.001, size=N[group]), 0.005, 0.011),
                      gamma=np.clip(np.random.normal(loc=gamma[group]['mu'], scale=gamma[group]['sd'], size=N[group]), -1, 1),
                      tau=np.clip(np.random.normal(loc=0.9, scale=0.1, size=N[group]), 0.2, 2),
                      t0=np.zeros(N[group]))
    
    m5.simulate_group(kind='individual',
                      parameters=parameters,
                      n_individuals=N[group],
                      n_trials=n_trials,
                      n_items=n_items,
                      label=group)
m5.data.rename(columns=dict(condition='group'), inplace=True)

m5.make_model(kind='individual', 
              depends_on=dict(gamma='group'),
              t0_val=0)
m5.fit(method='NUTS', n_samples=1000, chains=2, njobs=1)

estimates = glam.utils.get_estimates(m5)
estimates

np.random.seed(10101)

m6 = glam.GLAM()

groups = ['high', 'low']

N = 2

n_trials = 50
n_items = 5

gamma = dict(high=dict(mu=-0.3, sd=0.1),
             low=dict(mu=0.5, sd=0.1))

for group in groups:
    
    parameters = dict(v=np.clip(np.random.normal(loc=0.00007, scale=0.00001, size=N), 0.00003, 0.00015),
                      s=np.clip(np.random.normal(loc=0.008, scale=0.001, size=N), 0.005, 0.011),
                      gamma=np.clip(np.random.normal(loc=gamma[group]['mu'], scale=gamma[group]['sd'], size=N), -1, 1),
                      tau=np.clip(np.random.normal(loc=0.9, scale=0.1, size=N), 0.2, 2),
                      t0=np.zeros(N))
    
    m6.simulate_group(kind='individual',
                      parameters=parameters,
                      n_individuals=N,
                      individual_idx=np.arange(N),
                      n_trials=n_trials,
                      n_items=n_items,
                      label=group)
m6.data.rename(columns=dict(condition='group'), inplace=True)

m6.make_model(kind='individual', 
              depends_on=dict(gamma='group'),
              t0_val=0)
m6.fit(method='NUTS', n_samples=1000, chains=2, njobs=1)

estimates = glam.utils.get_estimates(m6)
estimates

np.random.seed(101110)

m7 = glam.GLAM()

N = 1

n_trials = 50
n_items = 5

gamma = dict(high=dict(mu=-0.3, sd=0.1),
             low=dict(mu=0.5, sd=0.1))
v = dict(slow=dict(mu=0.00005, sd=0.00001),
         fast=dict(mu=0.00010, sd=0.00001))

for bias in ['high', 'low']:
    for speed in ['slow', 'fast']:
    
        parameters = dict(v=np.clip(np.random.normal(loc=v[speed]['mu'], scale=v[speed]['sd'], size=N), 0.00003, 0.00015),
                          s=np.clip(np.random.normal(loc=0.008, scale=0.001, size=N), 0.005, 0.011),
                          gamma=np.clip(np.random.normal(loc=gamma[bias]['mu'], scale=gamma[bias]['sd'], size=N), -1, 1),
                          tau=np.clip(np.random.normal(loc=0.9, scale=0.1, size=N), 0.2, 2),
                          t0=np.zeros(N))

        m7.simulate_group(kind='individual',
                          parameters=parameters,
                          n_individuals=N,
                          n_trials=n_trials,
                          n_items=n_items,
                          label=bias + '_' + speed)

m7.data['bias'], m7.data['speed'] = m7.data['condition'].str.split('_', 1).str

m7.make_model(kind='individual', 
              depends_on=dict(gamma='bias',
                              v='speed'),
              t0_val=0)
m7.fit(method='NUTS', n_samples=1000, chains=2, njobs=1)

estimates = glam.utils.get_estimates(m7)
estimates

np.random.seed(1011)

m8 = glam.GLAM()

N = 4

n_trials = 50
n_items = 5

gamma = dict(high=dict(mu=-0.3, sd=0.1),
             low=dict(mu=0.5, sd=0.1))
v = dict(slow=dict(mu=0.00005, sd=0.00001),
         fast=dict(mu=0.00010, sd=0.00001))

for bias in ['high', 'low']:
    for speed in ['slow', 'fast']:
    
        parameters = dict(v=np.clip(np.random.normal(loc=v[speed]['mu'], scale=v[speed]['sd'], size=N), 0.00003, 0.00015),
                          s=np.clip(np.random.normal(loc=0.008, scale=0.001, size=N), 0.005, 0.011),
                          gamma=np.clip(np.random.normal(loc=gamma[bias]['mu'], scale=gamma[bias]['sd'], size=N), -1, 1),
                          tau=np.clip(np.random.normal(loc=0.9, scale=0.1, size=N), 0.2, 2),
                          t0=np.zeros(N))

        m8.simulate_group(kind='individual',
                          parameters=parameters,
                          individual_idx=np.arange(N),
                          n_individuals=N,
                          n_trials=n_trials,
                          n_items=n_items,
                          label=bias + '_' + speed)

m8.data['bias'], m8.data['speed'] = m8.data['condition'].str.split('_', 1).str

m8.make_model(kind='individual', 
              depends_on=dict(gamma='bias',
                              v='speed'),
              t0_val=0)
m8.fit(method='NUTS', n_samples=1000, chains=2, njobs=1)

estimates = glam.utils.get_estimates(m8)
estimates

