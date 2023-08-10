get_ipython().run_line_magic('matplotlib', 'inline')
import pymc3 as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('colorblind')
sns.set_color_codes()

# load data
d = pd.read_csv('wafflehouse_divorce.csv')
# standardize predictor
d['median_age_marriage_s'] = (d.median_marriage_age - 
                              d.median_marriage_age.mean()) / d.median_marriage_age.std()
d['marriage_rate_s'] = (d.marriage_rate - d.marriage_rate.mean())/ d.marriage_rate.std()

d.head().T

with pm.Model() as multi_linear:
    alpha = pm.Normal('alpha', mu=10, sd=10)
    beta_a = pm.Normal('beta_a', mu=0, sd=1)
    beta_r = pm.Normal('beta_r', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = alpha + beta_a * d.median_age_marriage_s + beta_r * d.marriage_rate_s
    divorce_hat = pm.Normal('divorce_hat', mu, sigma, observed=d.divorce_rate)
    divorce_trace = pm.sample(1000)

pm.traceplot(divorce_trace);

divorce_trace_df = pm.trace_to_dataframe(divorce_trace)

sns.pairplot(divorce_trace_df);

divorce_trace_df.corr().round(2)

pm.hpd(divorce_trace_df.as_matrix())

pm.forestplot(divorce_trace, ['alpha', 'beta_a', 'beta_r'], alpha=0.11);

grid = np.linspace(-2, 2, 100)

def get_marginal_mu_and_prediction_hpds(trace_df, grid, weights):
    mu_means = np.zeros(len(grid))
    mu_hpds = np.zeros((len(grid), 2))
    divorce_hpds = np.zeros((len(grid), 2))

    for i, v in enumerate(grid):
        mus = (
            trace_df['alpha'] 
            + trace_df['beta_a'] * weights['beta_a'] * grid[i]
            + trace_df['beta_r'] * weights['beta_r'] * grid[i]
        )
        divorce_predictions = stats.norm.rvs(loc=mus, scale=trace_df['sigma'])

        mu_means[i] =  mus.mean()
        mu_hpds[i, :] = pm.hpd(mus, 0.11)
        divorce_hpds[i, :] = pm.hpd(divorce_predictions, 0.11)
    return mu_means, mu_hpds, divorce_hpds

mu_means_a, mu_hpds_a, divorce_hpds_a = get_marginal_mu_and_prediction_hpds(
    divorce_trace_df, grid, {'beta_a': 1, 'beta_r': 0})

mu_means_r, mu_hpds_r, divorce_hpds_r = get_marginal_mu_and_prediction_hpds(
    divorce_trace_df, grid, {'beta_a': 0, 'beta_r': 1})

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4));
ax0.plot(grid, mu_means_a);
ax0.fill_between(grid, mu_hpds_a[:, 0], mu_hpds_a[:, 1], alpha=0.2);
ax0.fill_between(grid, divorce_hpds_a[:, 0], divorce_hpds_a[:, 1], alpha=0.2);

ax1.plot(grid, mu_means_r);
ax1.fill_between(grid, mu_hpds_r[:, 0], mu_hpds_r[:, 1], alpha=0.2);
ax1.fill_between(grid, divorce_hpds_r[:, 0], divorce_hpds_r[:, 1], alpha=0.2);
#FIXME: share y-axis

divorce_pred = pm.sample_ppc(divorce_trace, samples=1000, model=multi_linear)['divorce_hat']

plt.scatter(d.divorce_rate, divorce_pred.mean(0))
plt.plot(d.divorce_rate, d.divorce_rate,'r-');
plt.gca().set_aspect('equal', adjustable='box');

d2 = d.loc[:, ('divorce_rate', 'loc')].copy()

d2['divorce_hat'] = divorce_pred.mean(0)

d2['divorce_hpd_min'] = pm.hpd(divorce_pred)[:, 0]
d2['divorce_hpd_max'] = pm.hpd(divorce_pred)[:, 1]

d2.sort_values('divorce_rate').head()

plt.figure(figsize=(4, 12));
plt.scatter(d2['divorce_hat'], d2['loc']);
plt.hlines(d2['loc'], d2['divorce_hpd_min'], d2['divorce_hpd_max'], color='steelblue', alpha=0.5)

