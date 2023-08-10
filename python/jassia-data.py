import pandas as pd
import pymc3 as pm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context('poster')
sns.set_style('white')

get_ipython().magic('matplotlib inline')

df = pd.read_csv('jass-data.csv', index_col=0)
df.head()
df['weight_delta_0_7'] = (df['Wt(g) Day7'] - df['Wt(g) Day0']) / df['Wt(g) Day0']

df.columns

coinfected = df[df['Group'] == 'Hp+Flu+'].dropna(subset=['weight_delta_0_7'])
coinfected.shape

flu_only = df[df['Group'] == 'Hp-Flu+'].dropna(subset=['weight_delta_0_7'])
flu_only.shape

hp_only = df[df['Group'] == 'Hp+Flu-'].dropna(subset=['weight_delta_0_7'])
hp_only.shape

control = df[df['Group'] == 'Hp-Flu-'].dropna(subset=['weight_delta_0_7'])
control.shape

# Model specification.
with pm.Model() as model:
    nu_coinf = pm.Exponential('nu_coinf', 1/(len(coinfected) - 1)) + 1    
    mu_coinf = pm.Flat('wt_change_coinf')
    sd_coinf = pm.Exponential('sd_coinf', lam=1)
    #coinf_wt = pm.Normal('coinfected weight change', mu=mu_coinf, sd=sd_coinf, 
    #                      observed=coinfected['weight_delta_0_7'].values)
    coinf_wt = pm.StudentT('coinfected weight change',
                           mu=mu_coinf,
                           lam=sd_coinf**-2,
                           nu=nu_coinf,
                           observed=coinfected['weight_delta_0_7'].values)

    nu_flu = pm.Exponential('nu_flu', 1/(len(flu_only) - 1)) + 1    
    mu_flu = pm.Flat('wt_change_flu')
    sd_flu = pm.Exponential('sd_flu', lam=1)
    # flu_wt = pm.Normal('flu-only weight change', mu=mu_flu, sd=sd_flu,
    #                    observed=flu_only['weight_delta_0_7'].values)
    flu_wt = pm.StudentT('flu-only weight change',
                         mu=mu_flu,
                         lam=sd_flu**-2,
                         nu=nu_flu,
                         observed=flu_only['weight_delta_0_7'].values)

    nu_hp = pm.Exponential('nu_hp', 1/(len(hp_only) - 1)) + 1        
    mu_hp = pm.Flat('wt_change_hp')
    sd_hp = pm.Exponential('sd_hp', lam=1)
    # hp_wt = pm.Normal('hp-only weight change', mu=mu_hp, sd=sd_hp,
    #                   observed=hp_only['weight_delta_0_7'].values)
    hp_wt = pm.StudentT('hp-only weight change',
                        mu=mu_hp,
                        lam=sd_hp**-2,
                        nu=nu_hp,
                        observed=hp_only['weight_delta_0_7'].values)

    nu_ctrl = pm.Exponential('nu_ctrl', 1/(len(control) - 1)) + 1        
    mu_ctrl = pm.Flat('wt_change_ctrl')
    sd_ctrl = pm.Exponential('sd_ctrl', lam=1)
    # ctrl_wt = pm.Normal('control weight change', mu=mu_ctrl, sd=sd_ctrl,
    #                     observed=control['weight_delta_0_7'].values)
    ctrl_wt = pm.StudentT('control weight change',
                          mu=mu_ctrl,
                          lam=sd_ctrl**-2,
                          nu=nu_ctrl,
                          observed=control['weight_delta_0_7'].values)


    # ratio_cf = pm.Deterministic('ratio_cf', mu_coinf / mu_flu)  # ratio between coinfected and flu-only
    diff_cf = pm.Deterministic('diff_cf', mu_coinf - mu_flu)
    effect_cf = pm.Deterministic('effect_cf', abs(diff_cf) / pm.sqrt((sd_flu**2 + sd_coinf**2) / 2))
    
    # ratio_ch = pm.Deterministic('ratio_ch', mu_coinf / mu_hp)  # ratio between coinfected and hp-only
    diff_ch = pm.Deterministic('diff_ch', mu_coinf - mu_hp)
    effect_ch = pm.Deterministic('effect_ch', abs(diff_ch) / pm.sqrt((sd_hp**2 + sd_coinf**2) / 2))

get_ipython().run_cell_magic('time', '', 'with model:\n    params = pm.variational.advi(n=50000)\n    trace = pm.variational.sample_vp(params, draws=5000)')

pm.traceplot(trace)
plt.show()

pm.plot_posterior(trace[2000:],
                  varnames=['wt_change_ctrl', 'wt_change_flu', 'wt_change_hp', 'wt_change_coinf'], 
                  color='#87ceeb')
plt.show()

pm.plot_posterior(trace[2000:], varnames=['diff_cf', 'effect_cf'], color='#87ceeb')
plt.show()

pm.plot_posterior(trace[2000:], varnames=['diff_ch', 'effect_ch'], color='#87ceeb')
plt.show()

from scipy.stats import ttest_ind

ttest_ind(a=coinfected['weight_delta_0_7'], b=flu_only['weight_delta_0_7'])

ttest_ind(a=coinfected['weight_delta_0_7'], b=hp_only['weight_delta_0_7'])



