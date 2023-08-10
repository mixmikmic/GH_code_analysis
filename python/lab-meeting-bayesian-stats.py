import pandas as pd
get_ipython().magic('matplotlib inline')
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_context('poster')
sns.set_style('white')

baseline_name = 'Water'

data = pd.read_csv('bandoro-microbiome.csv')

# Encode the "Sample" column with indices.
sample_names = dict()
for i, name in enumerate(sorted(np.unique(data['Sample'].values))):
    sample_names[name] = i

for name, i in sample_names.items():
    data['Indices'] = data['Sample'].apply(lambda x: sample_names[x])
    
data = data.sort_values(by='Indices')
data = data.reset_index(drop=True)
    
baseline_idx = sample_names[baseline_name]
indices = list(set(data['Indices']))
indices.remove(baseline_idx)
print(indices)

data.tail(20)

summary_stats = pd.DataFrame()
summary_stats['mean'] = data.groupby('Sample').mean()['Normalized % GFP']
# We want the total width of the uncertainty measure, so we multiply the stdev by 2.
summary_stats['std'] = data.groupby('Sample').std()['Normalized % GFP'] * 2
summary_stats['3sd'] = summary_stats['std'] * 3
summary_stats['sem'] = summary_stats['std'] / np.sqrt(4)
summary_stats['95ci'] = 1.96 * summary_stats['sem']
summary_stats['range'] = data.groupby('Sample').max()['Normalized % GFP'] - data.groupby('Sample').min()['Normalized % GFP']
summary_stats

with pm.Model() as model:
    # Hyperpriors
    upper = pm.Exponential('upper', lam=0.05)
    # nu_inv = pm.Uniform('nu', lower=1E-10, upper=0.5, shape=len(sample_names))
    nu = pm.Exponential('nu_minus_one', 1/29.) + 1

    # "fold", which is the estimated fold change.
    fold = pm.Uniform('fold', lower=1E-10, upper=upper, shape=len(sample_names))

    # Assume that data have heteroskedastic (i.e. variable) error but are drawn from the same distribution
    # sigma = pm.Gamma('sigma', alpha=1, beta=1, shape=n_genotypes+2)
    sigma = pm.HalfCauchy('sigma', beta=1, shape=len(sample_names))

    # Model prediction
    mu = fold[data['Indices']]
    sig = sigma[data['Indices']]
    # nu_model = nu[data['Indices']]
    
    # Data likelihood
    like = pm.StudentT('like', nu=nu, mu=mu, sd=sig**-2, observed=data['Normalized % GFP'])    
    # like = pm.Normal('like', mu=mu, sd=sig, observed=data['Normalized % GFP'])
    
    diffs = pm.Deterministic('diffs', fold[indices] - fold[baseline_idx])
    s_pooled = pm.Deterministic('s_pooled', np.sqrt((sigma[indices] ** 2 + sigma[baseline_idx] ** 2) / 2))
    effect_size = pm.Deterministic('effect_size', diffs / s_pooled)
    
    # z_factor = pm.Deterministic('z_factor', 1 - (3 * sigma[indices, :] + 3 * sigma[baseline_idx]) / np.abs(fold[indices, :] - fold[baseline_idx]))

with model:
    n_steps = 200000
    params = pm.variational.advi(n=n_steps)
    trace = pm.variational.sample_vp(params, draws=2000)

lower, upper = np.percentile(trace['fold'], [2.5, 97.5], axis=0)
width = upper - lower

summary_stats['hpd'] = width
summary_stats['mean_bayes'] = trace['fold'].mean(axis=0)

summary_stats

pm.traceplot(trace, varnames=['nu_minus_one'])

trace['nu_minus_one'].mean(axis=0)

fig = plt.figure()
ax = fig.add_subplot(111)

# summary_stats[['std', '3sd', 'sem', '95ci', 'range', 'hpd']].plot(ax=ax)

lower, lower_q, upper_q, upper = np.percentile(trace['fold'], [2.5, 25, 75, 97.5], axis=0)
err_low = trace['fold'].mean(axis=0) - lower
err_high = upper - trace['fold'].mean(axis=0)
iqr_low = trace['fold'].mean(axis=0) - lower_q
iqr_high = upper_q - trace['fold'].mean(axis=0)

summary_stats['mean_bayes'].plot(rot=90, ls='', ax=ax, yerr=[err_low, err_high])
summary_stats['mean_bayes'].plot(rot=90, ls='', ax=ax, yerr=[iqr_low, iqr_high], elinewidth=4, color='red')
sns.swarmplot(data=data, x='Sample', y='Normalized % GFP', orient='v', ax=ax)
# sns.violinplot(data=trace_df, ax=ax)
plt.xticks(rotation='vertical')
plt.ylabel('mean activity')

pm.df_summary(trace)

pm.forestplot(trace, varnames=['fold'], ylabels=sorted(sample_names.keys()))

ax = summary_stats.plot(kind='scatter', x='range', y='sem', color='green', label='SEM')
summary_stats.plot(kind='scatter', x='range', y='95ci', ax=ax, color='red', label='frequentist CI', legend=True)
summary_stats.plot(kind='scatter', x='range', y='hpd', ax=ax, color='blue', label='bayesian CI', legend=True)
ax.legend(loc='upper left')
plt.ylabel('uncertainty')

data[data['Sample'] == 'Escherichia coli K12']

outlier_test_data = data[data['Sample'] == 'Escherichia coli K12']['Normalized % GFP'].values
outlier_test_data

largest = sorted(outlier_test_data)[-1]
second_largest = sorted(outlier_test_data)[-2]
smallest = sorted(outlier_test_data)[0]

gap = largest - second_largest
data_range = largest - smallest

gap / data_range

sample_names_without_water = sorted(sample_names.keys())
sample_names_without_water.remove('Water')
pm.forestplot(trace, varnames=['effect_size'], ylabels=sample_names_without_water)



