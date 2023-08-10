get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import seaborn as sns
import theano.tensor as tt

import os
fatality_data = 'fatality_data.csv'

if not os.path.exists(fatality_data):
    df = pd.read_html('https://www.fhwa.dot.gov/ohim/onh00/onh2p11.htm')[0]
    df.columns = df.iloc[0]
    df = df.reindex(df.index.drop([0, 52]))  # also drop the aggregate row
    df.to_csv(fatality_data, index=False)
df = pd.read_csv(fatality_data)

# Pandas did not detect most columns as numeric, so we can cast them all here
df[df.columns[1:]] = df[df.columns[1:]].apply(pd.to_numeric)

# Extract the data we need as numpy arrays
miles_e6 = (df['Annual VMT (Millions)']).as_matrix()
fatalities = df['Total Highway Fatalities'].as_matrix()

def car_model(miles, fatalities, google_miles=1.5, google_fatalities=0):
    with pm.Model() as model:
        pooled_rate = pm.Uniform('pooled_rate', lower=0.0, upper=1.0)
        κ_log = pm.Exponential('κ_log', lam=1.5)
        κ = pm.Deterministic('κ', tt.exp(κ_log))

        state_rate = pm.Beta('state_rate', 
                             alpha=pooled_rate*κ, 
                             beta=(1.0-pooled_rate)*κ, 
                             shape=len(fatalities))
        observed_fatalities = pm.Poisson('y', mu=state_rate*miles, observed=fatalities)

        google_rate = pm.Beta('google_rate', 
                              alpha=pooled_rate*κ, 
                              beta=(1.0-pooled_rate)*κ)
        observed_google_fatalities = pm.Poisson('y_new', 
                                                mu=google_miles*google_rate, 
                                                observed=google_fatalities)
    return model

with car_model(miles_e6, fatalities):
    trace = pm.sample(10000, njobs=4)

pm.traceplot(trace);

plt.style.use('seaborn-talk')

fig, ax = plt.subplots(1, 1)
shared_kwargs = {'hist': False, 'ax': ax, 'kde_kws': {'linewidth': 5}}
for state in ('MA', 'CA', 'TX', 'AK'):
    sns.distplot(trace['state_rate'][:, df.State == state], label='{} fatality rate'.format(state), **shared_kwargs)
sns.distplot(trace['google_rate'], label='Google fatality rate', **shared_kwargs)
fig.set_size_inches(18.5, 10.5)
ax.set_xlim(xmin=0)
ax.set_ylim(ymin=-10)
ax.set_title('Estimated fatality rate of self driving cars compared to a few states')
ax.set_xlabel('Fatalities per 1 million miles')

ax.legend();

with car_model(miles_e6, fatalities, google_miles=4.):
    updated_trace = pm.sample(10000, njobs=4)

plt.style.use('seaborn-talk')
fig, ax = plt.subplots(1, 1)
shared_kwargs = {'hist': False, 'ax': ax, 'kde_kws': {'linewidth': 5}}
sns.distplot(trace['google_rate'], label='1.5 million miles', **shared_kwargs)
sns.distplot(updated_trace['google_rate'], label='4 million miles', **shared_kwargs)
fig.set_size_inches(18.5, 10.5)
ax.set_xlim(xmin=0)
ax.set_title('Estimated fatality rate with\n1.5 million miles vs 4 million miles')
ax.set_xlabel('Fatalities per 1 million miles')
ax.legend();

import matplotlib
matplotlib.rcParams['figure.figsize'] = (18, 18)
plt.style.use('fivethirtyeight')
pm.forestplot(trace, ('state_rate',), 
              chain_spacing=0, 
              ylabels=df.State, 
              rhat=False, 
              xtitle="Fatalities per 1 million miles",
             plot_kwargs={"linewidth": 10, 'color': 'green'});



