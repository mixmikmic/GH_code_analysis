from crash_modeling_tools import *
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

crash_data = pd.read_csv('../data/crash_modeling_tools_demo_data/crash_data_final_90.csv')
crash_data = crash_data.dropna()
crash_data.head()

show_summary_stats(crash_data,[0,9])

show_summary_stats(crash_data)

offset_term = np.log(crash_data['seg_lng'] * 3)

# need to cast log_avg_aadt to float since statsmodels thinks (incorrectly) that it is categorical
crash_data['log_aadt'] = crash_data.log_avg_aadt.astype(np.float)
mod_nb = smf.glm('tot_acc_ct~log_aadt+lanewid+avg_grad+C(curve)+C(surf_typ)',data=crash_data,offset=offset_term,
                 family=sm.families.NegativeBinomial()).fit()
mod_nb.summary()

data_eb = pd.read_csv('../data/crash_modeling_tools_demo_data/crash_data_eb.csv')
data_eb = data_eb.dropna()
data_eb['log_aadt'] = data_eb.log_avg_aadt.astype(np.float)
compute_spf(mod_nb,data_eb)

eb_safety_estimates = estimate_empirical_bayes(mod_nb,data_eb,data_eb['seg_lng'],data_eb['tot_acc_ct'])
eb_safety_estimates.head()

arp = calc_accid_reduc_potential(mod_nb,data_eb,data_eb['seg_lng'],data_eb['tot_acc_ct'])
arp.head()

data_design = pd.read_csv('../data/crash_modeling_tools_demo_data/data_design.csv')
var_eta_hat = calc_var_eta_hat(mod_nb,data_design)

mu_hat = calc_mu_hat_nb(mod_nb,data_design)
mu_hat

ci_mu_nb = calc_ci_mu_nb(mu_hat,var_eta_hat)
ci_mu_nb.head()

pi_m_nb = calc_pi_m_nb(mod_nb,mu_hat,var_eta_hat)
pi_m_nb.head() 

pi_y_nb = calc_pi_y_nb(mod_nb,mu_hat,var_eta_hat)
pi_y_nb.head()

get_ipython().magic('matplotlib inline')
aadt_range = np.arange(9700,148400,100)

plot_and_save_nb_cis_and_pis(data_design,mod_nb,mu_hat,var_eta_hat,aadt_range,'AADT')



