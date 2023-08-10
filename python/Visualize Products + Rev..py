import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pprint
import matplotlib.ticker as mtick
import scipy.stats as stats
import pandas as pd
get_ipython().magic('pylab inline')
from collections import defaultdict

clients_over_time_per_week_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_all_clients")
cumulative_clients_over_time_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/times_series_cumulative_clients")
#cash management
cumulative_cm_customers_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_cash_management")
rev_customer_cm_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_cash_management_rev_per_customer")
total_weekly_rev_cm_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_cash_management_total_weekly_rev")
# checking
cumulative_checking_customers_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_checking")
rev_customer_checking_no_evidence  = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_checking_rev_per_customer")
total_weekly_rev_checking_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_checking_total_weekly_rev")
#CMMA
cumulative_cmma_customers_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_collateral_mma ")
rev_customer_cmma_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_collateral_mma_rev_per_customer")
total_weekly_rev_cmma_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_collateral_mma_total_weekly_rev")
# Enterprise Sweep
cumulative_es_customers_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_enterprise_sweep")
rev_customer_es_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_enterprise_sweep_rev_per_customer")
total_weekly_rev_es_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_enterprise_sweep_total_weekly_rev")
# FX
cumulative_fx_customers_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_fx")
rev_customer_fx_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_fx_rev_per_customer")
total_weekly_rev_fx_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_fx_total_weekly_rev")
# letters of credit
cumulative_loc_customers_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_letters_of_credit ")
rev_customer_loc_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_letters_of_credit_rev_per_customer")
total_weekly_rev_loc_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_letters_of_credit_total_weekly_rev")
#Money Market Bonus
cumulative_mmb_customers_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_money_market_bonus")
rev_customer_mmb_no_evidence = pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_money_market_bonus_rev_per_customer")
total_weekly_rev_mmb_no_evidence =  pd.read_pickle("../data-104-weeks-no-evidence/time_series_esp_money_market_bonus_total_weekly_rev")




cumulative_clients_week_0_no_evidence = defaultdict(list)
cumulative_clients_week_1_no_evidence = defaultdict(list)
cumulative_clients_week_2_no_evidence = defaultdict(list)
cumulative_clients_week_0_final = []
cumulative_clients_week_1_final = []
cumulative_clients_week_2_final = []
[cumulative_clients_week_0_no_evidence[i[1]].append(i[2]) for i in cumulative_clients_over_time_no_evidence[0]]
[cumulative_clients_week_1_no_evidence[i[1]].append(i[2]) for i in cumulative_clients_over_time_no_evidence[1]]
[cumulative_clients_week_2_no_evidence[i[1]].append(i[2]) for i in cumulative_clients_over_time_no_evidence[2]]
for k,v in cumulative_clients_week_0_no_evidence.items():
    cumulative_clients_week_0_final.append(max(v))
for k,v in cumulative_clients_week_1_no_evidence.items():
    cumulative_clients_week_1_final.append(max(v))  
for k,v in cumulative_clients_week_2_no_evidence.items():
    cumulative_clients_week_2_final.append(max(v)) 
final_cumulative_clients_no_evidence = [cumulative_clients_week_0_final
                                        ,cumulative_clients_week_1_final,cumulative_clients_week_2_final]



final_cumulative_clients_no_evidence = [cumulative_clients_week_0_final
                                        ,cumulative_clients_week_1_final,cumulative_clients_week_2_final]

final_cumulative_clients_no_evidence[2][-1]

sns.tsplot(final_cumulative_clients_no_evidence)
plt.title('Total clients over time no evidence')
plt.xlabel('week number')



# Get the percent across the three simulations to create a 85% confidence interval
cumulative_cm_customers_percent_no_evidence_final = []
cumulative_checking_customers_percent_no_evidence_final = []
cumulative_cmma_customers_percent_no_evidence_final = []
cumulative_es_customers_percent_no_evidence_final = []
cumulative_fx_customers_percent_no_evidence_final = []
cumulative_loc_customers_percent_no_evidence_final = []
cumulative_mmb_customers_percent_no_evidence_final = []

for simulation_cm,simulation_total in zip(cumulative_cm_customers_no_evidence,final_cumulative_clients_no_evidence) :
    cumulative_cm_customers_percent_no_evidence_final.append([i[2]/z for i,z in zip(simulation_cm,simulation_total)])

for simulation_checking,simulation_total in zip(cumulative_checking_customers_no_evidence,
                                                final_cumulative_clients_no_evidence) :
    cumulative_checking_customers_percent_no_evidence_final.append(
        [i[2]/z for i,z in zip(simulation_checking,simulation_total)])

for simulation_cmma,simulation_total in zip(cumulative_cmma_customers_no_evidence,
                                                final_cumulative_clients_no_evidence) :
    cumulative_cmma_customers_percent_no_evidence_final.append(
        [i[2]/z for i,z in zip(simulation_cmma,simulation_total)])

for simulation_es,simulation_total in zip(cumulative_es_customers_no_evidence,
                                                final_cumulative_clients_no_evidence) :
    cumulative_es_customers_percent_no_evidence_final.append(
        [i[2]/z for i,z in zip(simulation_es,simulation_total)])

for simulation_fx,simulation_total in zip(cumulative_fx_customers_no_evidence,
                                                final_cumulative_clients_no_evidence) :
    cumulative_fx_customers_percent_no_evidence_final.append(
        [i[2]/z for i,z in zip(simulation_fx,simulation_total)])

for simulation_loc,simulation_total in zip(cumulative_loc_customers_no_evidence,
                                                final_cumulative_clients_no_evidence) :
    cumulative_loc_customers_percent_no_evidence_final.append(
        [i[2]/z for i,z in zip(simulation_loc,simulation_total)])

for simulation_mmb,simulation_total in zip(cumulative_mmb_customers_no_evidence,
                                                final_cumulative_clients_no_evidence) :
    cumulative_mmb_customers_percent_no_evidence_final.append(
        [i[2]/z for i,z in zip(simulation_mmb,simulation_total)])

sns.set(style="darkgrid")
plt.figure(figsize=(12,8))
#cm
sns.tsplot(data = cumulative_cm_customers_percent_no_evidence_final,value = 'cm',color='green', ci=95)
#checking
sns.tsplot(data =cumulative_checking_customers_percent_no_evidence_final, value = 'checking',color='black',ci=95)
#cmma
sns.tsplot(data =cumulative_cmma_customers_percent_no_evidence_final,value='cmma',color='orange',ci=95)
#es
sns.tsplot(data =cumulative_es_customers_percent_no_evidence_final,value='es',color='purple',ci=95)
#fx
sns.tsplot(data =cumulative_fx_customers_percent_no_evidence_final,value='fx',ci=95)
#loc
sns.tsplot(data =cumulative_loc_customers_percent_no_evidence_final,value='loc',ci=95)
#mmb
sns.tsplot(data =cumulative_mmb_customers_percent_no_evidence_final,value='mmb',ci=95, color ='red')


plt.legend(['cm','checking','cmma','es','fx','loc','mmb'],fontsize = 'large')
plt.ylabel('Percent')
plt.xlabel('week number')
plt.title('95% CI percent of customer with each product -  no starting evidence')

sns.set(style="darkgrid")
plt.figure(figsize=(12,8))
#cm
sns.tsplot(data = cumulative_cm_customers_percent_no_evidence_final,value = 'product 1',color='green', ci=95)
#checking
sns.tsplot(data =cumulative_checking_customers_percent_no_evidence_final, value = 'product 2',color='black',ci=95)
#cmma
sns.tsplot(data =cumulative_cmma_customers_percent_no_evidence_final,value='product 3',color='orange',ci=95)
#es
sns.tsplot(data =cumulative_es_customers_percent_no_evidence_final,value='product 4',color='purple',ci=95)
#fx
sns.tsplot(data =cumulative_fx_customers_percent_no_evidence_final,value='product 5',ci=95)
#loc
sns.tsplot(data =cumulative_loc_customers_percent_no_evidence_final,value='product 6',ci=95)
#mmb
sns.tsplot(data =cumulative_mmb_customers_percent_no_evidence_final,value='product 7',ci=95, color ='red')

#checking.legend(['checking'])
plt.legend(['product 1','product 2','product 3','product 4','product 5','product 6','product 7'])
plt.ylabel('Percent')
plt.xlabel('week number')
plt.title('95% CI percent of customer with product -  no starting evidence')



cumulative_cm_rev_per_customer_no_evidence_final = []
cumulative_checking_rev_per_customer_no_evidence_final = []
cumulative_cmma_rev_per_customer_no_evidence_final = []
cumulative_es_rev_per_customer_no_evidence_final = []
cumulative_fx_rev_per_customer_no_evidence_final = []
cumulative_loc_rev_per_customer_no_evidence_final = []
cumulative_mmb_rev_per_customer_no_evidence_final = []

for simulation in rev_customer_cm_no_evidence:
    cumulative_cm_rev_per_customer_no_evidence_final.append([i[2] for i in simulation])

for simulation in rev_customer_checking_no_evidence:
    cumulative_checking_rev_per_customer_no_evidence_final.append([i[2] for i in simulation])

for simulation in rev_customer_cmma_no_evidence:
    cumulative_cmma_rev_per_customer_no_evidence_final.append([i[2] for i in simulation])

for simulation in rev_customer_es_no_evidence:
    cumulative_es_rev_per_customer_no_evidence_final.append([i[2] for i in simulation])

for simulation in rev_customer_fx_no_evidence:
    cumulative_fx_rev_per_customer_no_evidence_final.append([i[2] for i in simulation])

for simulation in rev_customer_loc_no_evidence:
    cumulative_loc_rev_per_customer_no_evidence_final.append([i[2] for i in simulation])

for simulation in rev_customer_mmb_no_evidence:
    cumulative_mmb_rev_per_customer_no_evidence_final.append([i[2] for i in simulation])





sns.set(style="darkgrid")

plt.figure(figsize=(15,12))
#cm
sns.tsplot(data = cumulative_cm_rev_per_customer_no_evidence_final,value = 'cm',color='green', ci=95)
#checking
sns.tsplot(data =cumulative_checking_rev_per_customer_no_evidence_final, value = 'checking',color='black',ci=95)
#cmma
sns.tsplot(data =cumulative_cmma_rev_per_customer_no_evidence_final,value='cmma',color='orange',ci=95)
#es
sns.tsplot(data =cumulative_es_rev_per_customer_no_evidence_final,value='es',color='maroon',ci=95)
#fx
sns.tsplot(data =cumulative_fx_rev_per_customer_no_evidence_final,value='fx',ci=95)
#loc
sns.tsplot(data =cumulative_loc_rev_per_customer_no_evidence_final,value='loc',ci=95,color='pink')
#mmb
sns.tsplot(data =cumulative_mmb_rev_per_customer_no_evidence_final,value='mmb',ci=95, color ='red')


plt.legend(['cm','checking','cmma','es','fx','mmb','loc'], fontsize = 'large')
plt.ylabel('GP',fontsize = 'large')
plt.xlabel('week number')
plt.title('95% CI of expected weekly GP per product with no starting evidence')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
#cm
sns.tsplot(data = cumulative_cm_rev_per_customer_no_evidence_final,value = 'product 7',color='green', ci=95)
#checking
sns.tsplot(data =cumulative_checking_rev_per_customer_no_evidence_final, value = 'product 6',color='black',ci=95)
#cmma
sns.tsplot(data =cumulative_cmma_rev_per_customer_no_evidence_final,value='product 5',color='orange',ci=95)
#es
sns.tsplot(data =cumulative_es_rev_per_customer_no_evidence_final,value='product 4',color='pink',ci=95)
#fx
sns.tsplot(data =cumulative_fx_rev_per_customer_no_evidence_final,value='product 3',ci=95)
#loc
#sns.tsplot(data =cumulative_loc_rev_per_customer_no_evidence_final,value='loc',ci=95)
#mmb
sns.tsplot(data =cumulative_mmb_rev_per_customer_no_evidence_final,value='product 2',ci=95, color ='red')


plt.legend(['product 2','product 3','product 4','product 5','product 6','product 7'])
plt.ylabel('GP')
plt.xlabel('week number')
plt.title('95% CI of time-adjusted revenue per product with no starting evidence')

print('Mean CM GP for last period = {}'.format(mean([cumulative_cm_rev_per_customer_no_evidence_final[0][-1],
        cumulative_cm_rev_per_customer_no_evidence_final[1][-1],
        cumulative_cm_rev_per_customer_no_evidence_final[2][-1]])))

print('Mean Checking GP for last period = {}'.format(mean([cumulative_checking_rev_per_customer_no_evidence_final[0][-1],
        cumulative_checking_rev_per_customer_no_evidence_final[1][-1],
        cumulative_checking_rev_per_customer_no_evidence_final[2][-1]])))

print('Mean cmma GP for last period = {}'.format(mean([cumulative_cmma_rev_per_customer_no_evidence_final[0][-1],
        cumulative_cmma_rev_per_customer_no_evidence_final[1][-1],
        cumulative_cmma_rev_per_customer_no_evidence_final[2][-1]])))

print('Mean es GP for last period = {}'.format(mean([cumulative_es_rev_per_customer_no_evidence_final[0][-1],
        cumulative_es_rev_per_customer_no_evidence_final[1][-1],
        cumulative_es_rev_per_customer_no_evidence_final[2][-1]])))

print('Mean fx GP for last period = {}'.format(mean([cumulative_fx_rev_per_customer_no_evidence_final[0][-1],
        cumulative_fx_rev_per_customer_no_evidence_final[1][-1],
        cumulative_fx_rev_per_customer_no_evidence_final[2][-1]])))

print('Mean loc GP for last period = {}'.format(mean([cumulative_loc_rev_per_customer_no_evidence_final[0][-1],
        cumulative_loc_rev_per_customer_no_evidence_final[1][-1],
        cumulative_loc_rev_per_customer_no_evidence_final[2][-1]])))





plt.figure(figsize=(12,8))
sns.tsplot(data =cumulative_loc_rev_per_customer_no_evidence_final,value='loc',ci=95,color='maroon')
plt.legend(['loc'])
plt.ylabel('GP')
plt.xlabel('week number')
plt.title('95% CI of GP per product with no starting evidence')

clients_over_time_per_week_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_all_clients")
cumulative_clients_over_time_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/times_series_cumulative_clients")
#cash management
cumulative_cm_customers_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_cash_management")
rev_customer_cm_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_cash_management_rev_per_customer")
total_weekly_rev_cm_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_cash_management_total_weekly_rev")
# checking
cumulative_checking_customers_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_checking")
rev_customer_checking_evidence_checking_cm  = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_checking_rev_per_customer")
total_weekly_rev_checking_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_checking_total_weekly_rev")
#CMMA
cumulative_cmma_customers_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_collateral_mma ")
rev_customer_cmma_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_collateral_mma_rev_per_customer")
total_weekly_rev_cmma_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_collateral_mma_total_weekly_rev")
# Enterprise Sweep
cumulative_es_customers_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_enterprise_sweep")
rev_customer_es_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_enterprise_sweep_rev_per_customer")
total_weekly_rev_es_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_enterprise_sweep_total_weekly_rev")
# FX
cumulative_fx_customers_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_fx")
rev_customer_fx_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_fx_rev_per_customer")
total_weekly_rev_fx_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_fx_total_weekly_rev")
# letters of credit
cumulative_loc_customers_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_letters_of_credit ")
rev_customer_loc_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_letters_of_credit_rev_per_customer")
total_weekly_rev_loc_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_letters_of_credit_total_weekly_rev")
#Money Market Bonus
cumulative_mmb_customers_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_money_market_bonus")
rev_customer_mmb_evidence_checking_cm = pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_money_market_bonus_rev_per_customer")
total_weekly_rev_mmb_evidence_checking_cm =  pd.read_pickle("../data-104-weeks-checking-cm/time_series_esp_money_market_bonus_total_weekly_rev")


# get cumulative customers
cumulative_clients_week_0_evidence_checking_cm = defaultdict(list)
cumulative_clients_week_1_evidence_checking_cm = defaultdict(list)
cumulative_clients_week_2_evidence_checking_cm = defaultdict(list)
cumulative_clients_week_0_final_evid_checking_cm = []
cumulative_clients_week_1_final_evid_checking_cm = []
cumulative_clients_week_2_final_evid_checking_cm = []
[cumulative_clients_week_0_evidence_checking_cm[i[1]].append(i[2]) for i in cumulative_clients_over_time_evidence_checking_cm[0]]
[cumulative_clients_week_1_evidence_checking_cm[i[1]].append(i[2]) for i in cumulative_clients_over_time_evidence_checking_cm[1]]
[cumulative_clients_week_2_evidence_checking_cm[i[1]].append(i[2]) for i in cumulative_clients_over_time_evidence_checking_cm[2]]
for k,v in cumulative_clients_week_0_evidence_checking_cm.items():
   cumulative_clients_week_0_final_evid_checking_cm.append(max(v))
for k,v in cumulative_clients_week_1_evidence_checking_cm.items():
   cumulative_clients_week_1_final_evid_checking_cm.append(max(v))  
for k,v in cumulative_clients_week_2_evidence_checking_cm.items():
   cumulative_clients_week_2_final_evid_checking_cm.append(max(v)) 
final_cumulative_clients_evidence_checking_cm = [cumulative_clients_week_0_final_evid_checking_cm
                                       ,cumulative_clients_week_1_final_evid_checking_cm,
                                       cumulative_clients_week_2_final_evid_checking_cm]

final_cumulative_clients_evidence_checking_cm[2][-1]

plt.figure(figsize=(10,8))
sns.tsplot(final_cumulative_clients_evidence_checking_cm, value = 'checking & cm evidence', color = 'green')


sns.tsplot(final_cumulative_clients_no_evidence, value = 'no evidence')
plt.legend(['checking & cm evidence' , 'no evidence'])
plt.xlabel('week number')

# Get the percent across the three simulations to create a 85% confidence interval
cumulative_cm_customers_percent_evidence_checking_cm_final = []
cumulative_checking_customers_percent_evidence_checking_cm_final = []
cumulative_cmma_customers_percent_evidence_checking_cm_final = []
cumulative_es_customers_percent_evidence_checking_cm_final = []
cumulative_fx_customers_percent_evidence_checking_cm_final = []
cumulative_loc_customers_percent_evidence_checking_cm_final = []
cumulative_mmb_customers_percent_evidence_checking_cm_final = []

for simulation_fx,simulation_total in zip(cumulative_fx_customers_evidence_checking_cm,
                                                final_cumulative_clients_evidence_checking_cm) :
    cumulative_fx_customers_percent_evidence_checking_cm_final.append(
        [i[2]/z for i,z in zip(simulation_fx,simulation_total)])

for simulation_cm,simulation_total in zip(cumulative_cm_customers_evidence_checking_cm,
                                                final_cumulative_clients_evidence_checking_cm) :
    cumulative_cm_customers_percent_evidence_checking_cm_final.append(
        [i[2]/z for i,z in zip(simulation_cm,simulation_total)])

for simulation_loc,simulation_total in zip(cumulative_loc_customers_evidence_checking_cm,
                                                final_cumulative_clients_evidence_checking_cm) :
    cumulative_loc_customers_percent_evidence_checking_cm_final.append(
        [i[2]/z for i,z in zip(simulation_loc,simulation_total)])

for simulation_checking,simulation_total in zip(cumulative_checking_customers_evidence_checking_cm,
                                                final_cumulative_clients_evidence_checking_cm) :
    cumulative_checking_customers_percent_evidence_checking_cm_final.append(
        [i[2]/z for i,z in zip(simulation_checking,simulation_total)])

for simulation_cmma,simulation_total in zip(cumulative_cmma_customers_evidence_checking_cm,
                                                final_cumulative_clients_evidence_checking_cm) :
    cumulative_cmma_customers_percent_evidence_checking_cm_final.append(
        [i[2]/z for i,z in zip(simulation_cmma,simulation_total)])

for simulation_es,simulation_total in zip(cumulative_es_customers_evidence_checking_cm,
                                                final_cumulative_clients_evidence_checking_cm) :
    cumulative_es_customers_percent_evidence_checking_cm_final.append(
        [i[2]/z for i,z in zip(simulation_es,simulation_total)])

for simulation_mmb,simulation_total in zip(cumulative_mmb_customers_evidence_checking_cm,
                                                final_cumulative_clients_evidence_checking_cm) :
    cumulative_mmb_customers_percent_evidence_checking_cm_final.append(
        [i[2]/z for i,z in zip(simulation_mmb,simulation_total)])

sns.set(style="darkgrid")
plt.figure(figsize=(12,8))
#cm
sns.tsplot(data = cumulative_cm_customers_percent_evidence_checking_cm_final,value = 'cm',color='green', ci=95)
#checking
sns.tsplot(data =cumulative_checking_customers_percent_evidence_checking_cm_final, value = 'checking',color='black',ci=95)
#cmma
sns.tsplot(data =cumulative_cmma_customers_percent_evidence_checking_cm_final,value='cmma',color='orange',ci=95)
#es
sns.tsplot(data =cumulative_es_customers_percent_evidence_checking_cm_final,value='es',color='purple',ci=95)
#fx
sns.tsplot(data =cumulative_fx_customers_percent_evidence_checking_cm_final,value='fx',ci=95)
#loc
sns.tsplot(data =cumulative_loc_customers_percent_evidence_checking_cm_final,value='loc',ci=95)
#mmb
sns.tsplot(data =cumulative_mmb_customers_percent_evidence_checking_cm_final,value='mmb',ci=95, color ='red')

plt.legend(['cm','checking','cmma','es','fx','loc','mmb'],fontsize='large')
plt.ylabel('Percent')
plt.xlabel('week number')
plt.title('95% CI percent of customer with product -  starting evidence (Checking & cm)')

cumulative_cm_rev_per_customer_evidence_checking_cm_final = []
cumulative_checking_rev_per_customer_evidence_checking_cm_final = []
cumulative_cmma_rev_per_customer_evidence_checking_cm_final = []
cumulative_es_rev_per_customer_evidence_checking_cm_final = []
cumulative_fx_rev_per_customer_evidence_checking_cm_final = []
cumulative_loc_rev_per_customer_evidence_checking_cm_final = []
cumulative_mmb_rev_per_customer_evidence_checking_cm_final = []

for simulation in rev_customer_cm_evidence_checking_cm:
    cumulative_cm_rev_per_customer_evidence_checking_cm_final.append([i[2] for i in simulation])
for simulation in rev_customer_checking_evidence_checking_cm:
    cumulative_checking_rev_per_customer_evidence_checking_cm_final.append([i[2] for i in simulation])
for simulation in rev_customer_cmma_evidence_checking_cm:
    cumulative_cmma_rev_per_customer_evidence_checking_cm_final.append([i[2] for i in simulation])
for simulation in rev_customer_es_evidence_checking_cm:
    cumulative_es_rev_per_customer_evidence_checking_cm_final.append([i[2] for i in simulation])
for simulation in rev_customer_fx_evidence_checking_cm:
    cumulative_fx_rev_per_customer_evidence_checking_cm_final.append([i[2] for i in simulation])
for simulation in rev_customer_loc_evidence_checking_cm:
    cumulative_loc_rev_per_customer_evidence_checking_cm_final.append([i[2] for i in simulation])
for simulation in rev_customer_mmb_evidence_checking_cm:
    cumulative_mmb_rev_per_customer_evidence_checking_cm_final.append([i[2] for i in simulation])

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
#cm
sns.tsplot(data = cumulative_cm_rev_per_customer_evidence_checking_cm_final,value = 'cm',color='green', ci=95)
#checking
sns.tsplot(data =cumulative_checking_rev_per_customer_evidence_checking_cm_final, value = 'checking',color='black',ci=95)
#cmma
sns.tsplot(data =cumulative_cmma_rev_per_customer_evidence_checking_cm_final,value='cmma',color='orange',ci=95)
#es
sns.tsplot(data =cumulative_es_rev_per_customer_evidence_checking_cm_final,value='es',color='pink',ci=95)
#fx
sns.tsplot(data =cumulative_fx_rev_per_customer_evidence_checking_cm_final,value='fx',ci=95,color='maroon')
#loc
sns.tsplot(data =cumulative_loc_rev_per_customer_evidence_checking_cm_final,value='loc',ci=95)
#mmb
sns.tsplot(data =cumulative_mmb_rev_per_customer_evidence_checking_cm_final,value='mmb',ci=95, color ='red')


plt.legend(['cm','checking','cmma','es','fx','mmb','loc'],fontsize='large')
plt.ylabel('GP')
plt.xlabel('week number')
plt.title('95% CI of GP per product evidece = (Checking & Cm)')

plt.figure(figsize=(12,8))
sns.tsplot(data =cumulative_loc_rev_per_customer_evidence_checking_cm_final,value='loc',ci=95)


plt.legend(['loc'])
plt.ylabel('GP')
plt.xlabel('week number')
plt.title('95% CI of GP per product evidence = (Checking & CM)')

print('Mean CM GP for last period (evid = checking , cm) = {}'.format(mean([cumulative_cm_rev_per_customer_evidence_checking_cm_final[0][-1],
        cumulative_cm_rev_per_customer_evidence_checking_cm_final[1][-1],
        cumulative_cm_rev_per_customer_evidence_checking_cm_final[2][-1]])))

print('Mean Checking GP for last period (evid = checking , cm) = {}'.format(mean([cumulative_checking_rev_per_customer_evidence_checking_cm_final[0][-1],
        cumulative_checking_rev_per_customer_evidence_checking_cm_final[1][-1],
        cumulative_checking_rev_per_customer_evidence_checking_cm_final[2][-1]])))

print('Mean cmma GP for last period (evid = checking , cm) = {}'.format(mean([cumulative_cmma_rev_per_customer_evidence_checking_cm_final[0][-1],
        cumulative_cmma_rev_per_customer_evidence_checking_cm_final[1][-1],
        cumulative_cmma_rev_per_customer_evidence_checking_cm_final[2][-1]])))

print('Mean es GP for last period (evid = checking , cm) = {}'.format(mean([cumulative_es_rev_per_customer_evidence_checking_cm_final[0][-1],
        cumulative_es_rev_per_customer_evidence_checking_cm_final[1][-1],
        cumulative_es_rev_per_customer_evidence_checking_cm_final[2][-1]])))

print('Mean fx GP for last period (evid = checking , cm) = {}'.format(mean([cumulative_fx_rev_per_customer_evidence_checking_cm_final[0][-1],
        cumulative_fx_rev_per_customer_evidence_checking_cm_final[1][-1],
        cumulative_fx_rev_per_customer_evidence_checking_cm_final[2][-1]])))

print('Mean loc GP for last period (evid = checking , cm) = {}'.format(mean([cumulative_loc_rev_per_customer_evidence_checking_cm_final[0][-1],
        cumulative_loc_rev_per_customer_evidence_checking_cm_final[1][-1],
        cumulative_loc_rev_per_customer_evidence_checking_cm_final[2][-1]])))



clients_over_time_per_week_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_all_clients")
cumulative_clients_over_time_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/times_series_cumulative_clients")
#cash management
cumulative_cm_customers_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_cash_management")
rev_customer_cm_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_cash_management_rev_per_customer")
total_weekly_rev_cm_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_cash_management_total_weekly_rev")
# checking
cumulative_checking_customers_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_checking")
rev_customer_checking_evidence_checking_mmb  = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_checking_rev_per_customer")
total_weekly_rev_checking_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_checking_total_weekly_rev")
#CMMA
cumulative_cmma_customers_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_collateral_mma ")
rev_customer_cmma_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_collateral_mma_rev_per_customer")
total_weekly_rev_cmma_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_collateral_mma_total_weekly_rev")
# Enterprise Sweep
cumulative_es_customers_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_enterprise_sweep")
rev_customer_es_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_enterprise_sweep_rev_per_customer")
total_weekly_rev_es_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_enterprise_sweep_total_weekly_rev")
# FX
cumulative_fx_customers_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_fx")
rev_customer_fx_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_fx_rev_per_customer")
total_weekly_rev_fx_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_fx_total_weekly_rev")
# letters of credit
cumulative_loc_customers_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_letters_of_credit ")
rev_customer_loc_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_letters_of_credit_rev_per_customer")
total_weekly_rev_loc_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_letters_of_credit_total_weekly_rev")
#Money Market Bonus
cumulative_mmb_customers_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_money_market_bonus")
rev_customer_mmb_evidence_checking_mmb = pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_money_market_bonus_rev_per_customer")
total_weekly_rev_mmb_evidence_checking_mmb =  pd.read_pickle("../data-104-weeks-checking-mmb/time_series_esp_money_market_bonus_total_weekly_rev")


# get cumulative customers
cumulative_clients_week_0_evidence_checking_mmb = defaultdict(list)
cumulative_clients_week_1_evidence_checking_mmb = defaultdict(list)
cumulative_clients_week_2_evidence_checking_mmb = defaultdict(list)
cumulative_clients_week_0_final_evid_checking_mmb = []
cumulative_clients_week_1_final_evid_checking_mmb = []
cumulative_clients_week_2_final_evid_checking_mmb = []
[cumulative_clients_week_0_evidence_checking_mmb[i[1]].append(i[2]) for i in cumulative_clients_over_time_evidence_checking_mmb[0]]
[cumulative_clients_week_1_evidence_checking_mmb[i[1]].append(i[2]) for i in cumulative_clients_over_time_evidence_checking_mmb[1]]
[cumulative_clients_week_2_evidence_checking_mmb[i[1]].append(i[2]) for i in cumulative_clients_over_time_evidence_checking_mmb[2]]
for k,v in cumulative_clients_week_0_evidence_checking_mmb.items():
   cumulative_clients_week_0_final_evid_checking_mmb.append(max(v))
for k,v in cumulative_clients_week_1_evidence_checking_mmb.items():
   cumulative_clients_week_1_final_evid_checking_mmb.append(max(v))  
for k,v in cumulative_clients_week_2_evidence_checking_mmb.items():
   cumulative_clients_week_2_final_evid_checking_mmb.append(max(v)) 
final_cumulative_clients_evidence_checking_mmb = [cumulative_clients_week_0_final_evid_checking_mmb
                                       ,cumulative_clients_week_1_final_evid_checking_mmb,
                                       cumulative_clients_week_2_final_evid_checking_mmb]

sns.tsplot(final_cumulative_clients_evidence_checking_mmb)
plt.title('Total clients over time evidence = checking & mmb')
plt.xlabel('week number')

# Get the percent across the three simulations to create a 85% confidence interval
cumulative_cm_customers_percent_evidence_checking_mmb_final = []
cumulative_checking_customers_percent_evidence_checking_mmb_final = []
cumulative_cmma_customers_percent_evidence_checking_mmb_final = []
cumulative_es_customers_percent_evidence_checking_mmb_final = []
cumulative_fx_customers_percent_evidence_checking_mmb_final = []
cumulative_loc_customers_percent_evidence_checking_mmb_final = []
cumulative_mmb_customers_percent_evidence_checking_mmb_final = []


for simulation_fx,simulation_total in zip(cumulative_fx_customers_evidence_checking_mmb,
                                                final_cumulative_clients_evidence_checking_mmb) :
    cumulative_fx_customers_percent_evidence_checking_mmb_final.append(
        [i[2]/z for i,z in zip(simulation_fx,simulation_total)])
for simulation_cm,simulation_total in zip(cumulative_cm_customers_evidence_checking_mmb,
                                                final_cumulative_clients_evidence_checking_mmb) :
    cumulative_cm_customers_percent_evidence_checking_mmb_final.append(
        [i[2]/z for i,z in zip(simulation_cm,simulation_total)])
for simulation_loc,simulation_total in zip(cumulative_loc_customers_evidence_checking_mmb,
                                                final_cumulative_clients_evidence_checking_mmb) :
    cumulative_loc_customers_percent_evidence_checking_mmb_final.append(
        [i[2]/z for i,z in zip(simulation_loc,simulation_total)])
for simulation_checking,simulation_total in zip(cumulative_checking_customers_evidence_checking_mmb,
                                                final_cumulative_clients_evidence_checking_mmb) :
    cumulative_checking_customers_percent_evidence_checking_mmb_final.append(
        [i[2]/z for i,z in zip(simulation_checking,simulation_total)])
for simulation_cmma,simulation_total in zip(cumulative_cmma_customers_evidence_checking_mmb,
                                                final_cumulative_clients_evidence_checking_mmb) :
    cumulative_cmma_customers_percent_evidence_checking_mmb_final.append(
        [i[2]/z for i,z in zip(simulation_cmma,simulation_total)])
for simulation_es,simulation_total in zip(cumulative_es_customers_evidence_checking_mmb,
                                                final_cumulative_clients_evidence_checking_mmb) :
    cumulative_es_customers_percent_evidence_checking_mmb_final.append(
        [i[2]/z for i,z in zip(simulation_es,simulation_total)])
for simulation_mmb,simulation_total in zip(cumulative_mmb_customers_evidence_checking_mmb,
                                                final_cumulative_clients_evidence_checking_mmb) :
    cumulative_mmb_customers_percent_evidence_checking_mmb_final.append(
        [i[2]/z for i,z in zip(simulation_mmb,simulation_total)])
    
#plot
sns.set(style="darkgrid")
plt.figure(figsize=(12,8))
#cm
sns.tsplot(data = cumulative_cm_customers_percent_evidence_checking_mmb_final,value = 'cm',color='green', ci=95)
#checking
sns.tsplot(data =cumulative_checking_customers_percent_evidence_checking_mmb_final, value = 'checking',color='black',ci=95)
#cmma
sns.tsplot(data =cumulative_cmma_customers_percent_evidence_checking_mmb_final,value='cmma',color='orange',ci=95)
#es
sns.tsplot(data =cumulative_es_customers_percent_evidence_checking_mmb_final,value='es',color='purple',ci=95)
#fx
sns.tsplot(data =cumulative_fx_customers_percent_evidence_checking_mmb_final,value='fx',ci=95)
#loc
sns.tsplot(data =cumulative_loc_customers_percent_evidence_checking_mmb_final,value='loc',ci=95)
#mmb
sns.tsplot(data =cumulative_mmb_customers_percent_evidence_checking_mmb_final,value='mmb',ci=95, color ='red')

plt.legend(['cm','checking','cmma','es','fx','loc','mmb'],fontsize='large')
plt.ylabel('Percent')
plt.xlabel('week number')
plt.title('95% CI percent of customer with product -  starting evidence (Checking & mmb)')

cumulative_cm_rev_per_customer_evidence_checking_mmb_final = []
cumulative_checking_rev_per_customer_evidence_checking_mmb_final = []
cumulative_cmma_rev_per_customer_evidence_checking_mmb_final = []
cumulative_es_rev_per_customer_evidence_checking_mmb_final = []
cumulative_fx_rev_per_customer_evidence_checking_mmb_final = []
cumulative_loc_rev_per_customer_evidence_checking_mmb_final = []
cumulative_mmb_rev_per_customer_evidence_checking_mmb_final = []



for simulation in rev_customer_cm_evidence_checking_mmb:
    cumulative_cm_rev_per_customer_evidence_checking_mmb_final.append([i[2] for i in simulation])
for simulation in rev_customer_checking_evidence_checking_mmb:
    cumulative_checking_rev_per_customer_evidence_checking_mmb_final.append([i[2] for i in simulation])
for simulation in rev_customer_cmma_evidence_checking_mmb:
    cumulative_cmma_rev_per_customer_evidence_checking_mmb_final.append([i[2] for i in simulation])
for simulation in rev_customer_es_evidence_checking_mmb:
    cumulative_es_rev_per_customer_evidence_checking_mmb_final.append([i[2] for i in simulation])
for simulation in rev_customer_fx_evidence_checking_mmb:
    cumulative_fx_rev_per_customer_evidence_checking_mmb_final.append([i[2] for i in simulation])
for simulation in rev_customer_loc_evidence_checking_mmb:
    cumulative_loc_rev_per_customer_evidence_checking_mmb_final.append([i[2] for i in simulation])
for simulation in rev_customer_mmb_evidence_checking_mmb:
    cumulative_mmb_rev_per_customer_evidence_checking_mmb_final.append([i[2] for i in simulation])
    
    
#plot
plt.figure(figsize=(15,12))
#cm
sns.tsplot(data = cumulative_cm_rev_per_customer_evidence_checking_mmb_final,value = 'cm',color='green', ci=95)
#checking
sns.tsplot(data =cumulative_checking_rev_per_customer_evidence_checking_mmb_final, value = 'checking',color='black',ci=95)
#cmma
sns.tsplot(data =cumulative_cmma_rev_per_customer_evidence_checking_mmb_final,value='cmma',color='orange',ci=95)
#es
sns.tsplot(data =cumulative_es_rev_per_customer_evidence_checking_mmb_final,value='es',color='pink',ci=95)
#fx
sns.tsplot(data =cumulative_fx_rev_per_customer_evidence_checking_mmb_final,value='fx',ci=95,color='maroon')
#loc
sns.tsplot(data =cumulative_loc_rev_per_customer_evidence_checking_mmb_final,value='loc',ci=95)
#mmb
sns.tsplot(data =cumulative_mmb_rev_per_customer_evidence_checking_mmb_final,value='mmb',ci=95, color ='red')


plt.legend(['cm','checking','cmma','es','fx','mmb'],fontsize='large')
plt.ylabel('GP')
plt.xlabel('week number')
plt.title('95% CI of GP per product evidece = (Checking & mmb)')

plt.figure(figsize=(12,8))
sns.tsplot(data =cumulative_loc_rev_per_customer_evidence_checking_mmb_final,value='loc',ci=95)


plt.legend(['loc'])
plt.ylabel('GP')
plt.xlabel('week number')
plt.title('95% CI of GP per product evidence = (Checking & mmb)')



clients_over_time_per_week_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_all_clients")
cumulative_clients_over_time_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-mmb/times_series_cumulative_clients")
#cash management
cumulative_cm_customers_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_cash_management")
rev_customer_cm_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_cash_management_rev_per_customer")
total_weekly_rev_cm_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_cash_management_total_weekly_rev")
# checking
cumulative_checking_customers_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_checking")
rev_customer_checking_evidence_checking_loc  = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_checking_rev_per_customer")
total_weekly_rev_checking_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_checking_total_weekly_rev")
#CMMA
cumulative_cmma_customers_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_collateral_mma ")
rev_customer_cmma_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_collateral_mma_rev_per_customer")
total_weekly_rev_cmma_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_collateral_mma_total_weekly_rev")
# Enterprise Sweep
cumulative_es_customers_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_enterprise_sweep")
rev_customer_es_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_enterprise_sweep_rev_per_customer")
total_weekly_rev_es_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_enterprise_sweep_total_weekly_rev")
# FX
cumulative_fx_customers_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_fx")
rev_customer_fx_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_fx_rev_per_customer")
total_weekly_rev_fx_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_fx_total_weekly_rev")
# letters of credit
cumulative_loc_customers_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_letters_of_credit ")
rev_customer_loc_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_letters_of_credit_rev_per_customer")
total_weekly_rev_loc_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_letters_of_credit_total_weekly_rev")
#Money Market Bonus
cumulative_mmb_customers_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_money_market_bonus")
rev_customer_mmb_evidence_checking_loc = pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_money_market_bonus_rev_per_customer")
total_weekly_rev_mmb_evidence_checking_loc =  pd.read_pickle("../data-104-weeks-checking-loc/time_series_esp_money_market_bonus_total_weekly_rev")


# get cumulative customers
cumulative_clients_week_0_evidence_checking_loc = defaultdict(list)
cumulative_clients_week_1_evidence_checking_loc = defaultdict(list)
cumulative_clients_week_2_evidence_checking_loc = defaultdict(list)
cumulative_clients_week_0_final_evid_checking_loc = []
cumulative_clients_week_1_final_evid_checking_loc = []
cumulative_clients_week_2_final_evid_checking_loc = []
[cumulative_clients_week_0_evidence_checking_loc[i[1]].append(i[2]) for i in cumulative_clients_over_time_evidence_checking_loc[0]]
[cumulative_clients_week_1_evidence_checking_loc[i[1]].append(i[2]) for i in cumulative_clients_over_time_evidence_checking_loc[1]]
[cumulative_clients_week_2_evidence_checking_loc[i[1]].append(i[2]) for i in cumulative_clients_over_time_evidence_checking_loc[2]]
for k,v in cumulative_clients_week_0_evidence_checking_loc.items():
   cumulative_clients_week_0_final_evid_checking_loc.append(max(v))
for k,v in cumulative_clients_week_1_evidence_checking_loc.items():
   cumulative_clients_week_1_final_evid_checking_loc.append(max(v))  
for k,v in cumulative_clients_week_2_evidence_checking_loc.items():
   cumulative_clients_week_2_final_evid_checking_loc.append(max(v)) 
final_cumulative_clients_evidence_checking_loc = [cumulative_clients_week_0_final_evid_checking_loc
                                       ,cumulative_clients_week_1_final_evid_checking_loc,
                                       cumulative_clients_week_2_final_evid_checking_loc]

#plot it
sns.tsplot(final_cumulative_clients_evidence_checking_loc)
plt.title('Total clients over time evidence = checking & loc')
plt.xlabel('week number')

# Get the percent across the three simulations to create a 85% confidence interval
cumulative_cm_customers_percent_evidence_checking_loc_final = []
cumulative_checking_customers_percent_evidence_checking_loc_final = []
cumulative_cmma_customers_percent_evidence_checking_loc_final = []
cumulative_es_customers_percent_evidence_checking_loc_final = []
cumulative_fx_customers_percent_evidence_checking_loc_final = []
cumulative_loc_customers_percent_evidence_checking_loc_final = []
cumulative_mmb_customers_percent_evidence_checking_loc_final = []


for simulation_fx,simulation_total in zip(cumulative_fx_customers_evidence_checking_loc,
                                                final_cumulative_clients_evidence_checking_loc) :
    cumulative_fx_customers_percent_evidence_checking_loc_final.append(
        [i[2]/z for i,z in zip(simulation_fx,simulation_total)])
    
for simulation_cm,simulation_total in zip(cumulative_cm_customers_evidence_checking_loc,
                                                final_cumulative_clients_evidence_checking_loc) :
    cumulative_cm_customers_percent_evidence_checking_loc_final.append(
        [i[2]/z for i,z in zip(simulation_cm,simulation_total)])
    
for simulation_loc,simulation_total in zip(cumulative_loc_customers_evidence_checking_loc,
                                                final_cumulative_clients_evidence_checking_loc) :
    cumulative_loc_customers_percent_evidence_checking_loc_final.append(
        [i[2]/z for i,z in zip(simulation_loc,simulation_total)])
    
for simulation_checking,simulation_total in zip(cumulative_checking_customers_evidence_checking_loc,
                                                final_cumulative_clients_evidence_checking_loc) :
    cumulative_checking_customers_percent_evidence_checking_loc_final.append(
        [i[2]/z for i,z in zip(simulation_checking,simulation_total)])
    
for simulation_cmma,simulation_total in zip(cumulative_cmma_customers_evidence_checking_loc,
                                                final_cumulative_clients_evidence_checking_loc) :
    cumulative_cmma_customers_percent_evidence_checking_loc_final.append(
        [i[2]/z for i,z in zip(simulation_cmma,simulation_total)])
    
for simulation_es,simulation_total in zip(cumulative_es_customers_evidence_checking_loc,
                                                final_cumulative_clients_evidence_checking_loc) :
    cumulative_es_customers_percent_evidence_checking_loc_final.append(
        [i[2]/z for i,z in zip(simulation_es,simulation_total)])
    
for simulation_mmb,simulation_total in zip(cumulative_mmb_customers_evidence_checking_loc,
                                                final_cumulative_clients_evidence_checking_loc) :
    cumulative_mmb_customers_percent_evidence_checking_loc_final.append(
        [i[2]/z for i,z in zip(simulation_mmb,simulation_total)])
    
#plot
sns.set(style="darkgrid")
plt.figure(figsize=(12,8))
#cm
sns.tsplot(data = cumulative_cm_customers_percent_evidence_checking_loc_final,value = 'cm',color='green', ci=95)
#checking
sns.tsplot(data =cumulative_checking_customers_percent_evidence_checking_loc_final, value = 'checking',color='black',ci=95)
#cmma
sns.tsplot(data =cumulative_cmma_customers_percent_evidence_checking_loc_final,value='cmma',color='orange',ci=95)
#es
sns.tsplot(data =cumulative_es_customers_percent_evidence_checking_loc_final,value='es',color='purple',ci=95)
#fx
sns.tsplot(data =cumulative_fx_customers_percent_evidence_checking_loc_final,value='fx',ci=95)
#loc
sns.tsplot(data =cumulative_loc_customers_percent_evidence_checking_loc_final,value='loc',ci=95)
#mmb
sns.tsplot(data =cumulative_mmb_customers_percent_evidence_checking_loc_final,value='mmb',ci=95, color ='red')

#checking.legend(['checking'])
plt.legend(['cm','checking','cmma','es','fx','loc','mmb'])
plt.ylabel('Percent')
plt.xlabel('week number')
plt.title('95% CI percent of customer with product -  starting evidence (Checking & loc)')

cumulative_cm_rev_per_customer_evidence_checking_loc_final = []
cumulative_checking_rev_per_customer_evidence_checking_loc_final = []
cumulative_cmma_rev_per_customer_evidence_checking_loc_final = []
cumulative_es_rev_per_customer_evidence_checking_loc_final = []
cumulative_fx_rev_per_customer_evidence_checking_loc_final = []
cumulative_loc_rev_per_customer_evidence_checking_loc_final = []
cumulative_mmb_rev_per_customer_evidence_checking_loc_final = []



for simulation in rev_customer_cm_evidence_checking_loc:
    cumulative_cm_rev_per_customer_evidence_checking_loc_final.append([i[2] for i in simulation])
for simulation in rev_customer_checking_evidence_checking_loc:
    cumulative_checking_rev_per_customer_evidence_checking_loc_final.append([i[2] for i in simulation])
for simulation in rev_customer_cmma_evidence_checking_loc:
    cumulative_cmma_rev_per_customer_evidence_checking_loc_final.append([i[2] for i in simulation])
for simulation in rev_customer_es_evidence_checking_loc:
    cumulative_es_rev_per_customer_evidence_checking_loc_final.append([i[2] for i in simulation])
for simulation in rev_customer_fx_evidence_checking_loc:
    cumulative_fx_rev_per_customer_evidence_checking_loc_final.append([i[2] for i in simulation])
for simulation in rev_customer_loc_evidence_checking_loc:
    cumulative_loc_rev_per_customer_evidence_checking_loc_final.append([i[2] for i in simulation])
for simulation in rev_customer_mmb_evidence_checking_loc:
    cumulative_mmb_rev_per_customer_evidence_checking_loc_final.append([i[2] for i in simulation])
    
    
#plot
plt.figure(figsize=(15,12))
#cm
sns.tsplot(data = cumulative_cm_rev_per_customer_evidence_checking_loc_final,value = 'cm',color='green', ci=95)
#checking
sns.tsplot(data =cumulative_checking_rev_per_customer_evidence_checking_loc_final, value = 'checking',color='black',ci=95)
#cmma
sns.tsplot(data =cumulative_cmma_rev_per_customer_evidence_checking_loc_final,value='cmma',color='orange',ci=95)
#es
sns.tsplot(data =cumulative_es_rev_per_customer_evidence_checking_loc_final,value='es',color='pink',ci=95)
#fx
sns.tsplot(data =cumulative_fx_rev_per_customer_evidence_checking_loc_final,value='fx',ci=95,color='maroon')
#loc
sns.tsplot(data =cumulative_loc_rev_per_customer_evidence_checking_loc_final,value='loc',ci=95)
#mmb
sns.tsplot(data =cumulative_mmb_rev_per_customer_evidence_checking_loc_final,value='mmb',ci=95, color ='red')


plt.legend(['cm','checking','cmma','es','fx','mmb'],fontsize='large')
plt.ylabel('GP')
plt.xlabel('week number')
plt.title('95% CI of GP per product evidece = (Checking & loc)')

plt.figure(figsize=(12,8))
sns.tsplot(data =cumulative_loc_rev_per_customer_evidence_checking_loc_final,value='loc',ci=95)


plt.legend(['loc'])
plt.ylabel('GP')
plt.xlabel('week number')
plt.title('95% CI of GP per product evidence = (Checking & loc)')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
#cm
sns.tsplot(data = cumulative_cm_rev_per_customer_no_evidence_final,value = 'cm-no evidence',color='red', ci=95)
#cm
sns.tsplot(data = cumulative_cm_rev_per_customer_evidence_checking_cm_final,
           value = 'cm-evidence:checking & cm',color='green', ci=95)
#cm
sns.tsplot(data = cumulative_cm_rev_per_customer_evidence_checking_mmb_final,
           value = 'cm-evidence:checking & mmb',color='blue', ci=95)
#cm
sns.tsplot(data = cumulative_cm_rev_per_customer_evidence_checking_loc_final,
           value = 'cm-evidence:checking&loc',color='orange', ci=95)
plt.legend(['cm-no evidence','cm-evidence:checking & cm','cm-evidence:checking & mmb','cm-evidence:checking&loc'])
plt.title('GP for CM with different starting evidence')
plt.ylabel('GP')
plt.xlabel('Simulated week no.')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
#cm
sns.tsplot(data = cumulative_checking_rev_per_customer_no_evidence_final,value = 'checking-no evidence',color='red', ci=95)
#cm
sns.tsplot(data = cumulative_checking_rev_per_customer_evidence_checking_cm_final,
           value = 'checking-evidence:checking & cm',color='green', ci=95)
#cm
sns.tsplot(data = cumulative_checking_rev_per_customer_evidence_checking_mmb_final,
           value = 'checking-evidence:checking & mmb',color='blue', ci=95)
#cm
sns.tsplot(data = cumulative_checking_rev_per_customer_evidence_checking_loc_final,
           value = 'checking-evidence:checking&loc',color='orange', ci=95)
plt.legend(['checking-no evidence','checking-evidence:checking & cm','checking-evidence:checking & mmb','checking-evidence:checking&loc'])
plt.title('GP for Checking with different starting evidence')
plt.ylabel('GP')
plt.xlabel('Simulated week no.')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
sns.tsplot(data = cumulative_mmb_rev_per_customer_no_evidence_final,value = 'mmb-no evidence',color='red', ci=95)
sns.tsplot(data = cumulative_mmb_rev_per_customer_evidence_checking_cm_final,
           value = 'mmb-evidence:checking & cm',color='green', ci=95)
sns.tsplot(data = cumulative_mmb_rev_per_customer_evidence_checking_mmb_final,
           value = 'mmb-evidence:checking & mmb',color='blue', ci=95)

sns.tsplot(data = cumulative_mmb_rev_per_customer_evidence_checking_loc_final,
           value = 'mmb-evidence:checking&loc',color='orange', ci=95)
plt.legend(['mmb-no evidence','mmb-evidence:checking & cm','mmb-evidence:checking & mmb','mmb-evidence:checking&loc'],
          fontsize='large')
plt.title('GP for mmb with different starting evidence')
plt.ylabel('GP')
plt.xlabel('Simulated week no.')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
sns.tsplot(data = cumulative_loc_rev_per_customer_no_evidence_final,value = 'loc-no evidence',color='red', ci=95)
sns.tsplot(data = cumulative_loc_rev_per_customer_evidence_checking_cm_final,
           value = 'loc-evidence:checking & cm',color='green', ci=95)
#
sns.tsplot(data = cumulative_loc_rev_per_customer_evidence_checking_mmb_final,
           value = 'loc-evidence:checking & mmb',color='blue', ci=95)
sns.tsplot(data = cumulative_loc_rev_per_customer_evidence_checking_loc_final,
           value = 'loc-evidence:checking&loc',color='orange', ci=95)
plt.legend(['loc-no evidence','loc-evidence:checking & cm','loc-evidence:checking & mmb','loc-evidence:checking&loc'])
plt.title('GP for loc with different starting evidence')
plt.ylabel('GP')
plt.xlabel('Simulated week no.')

np.median(cumulative_loc_rev_per_customer_no_evidence_final[0])

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
sns.tsplot(data = cumulative_fx_rev_per_customer_no_evidence_final,value = 'fx-no evidence',color='red', ci=95)
sns.tsplot(data = cumulative_fx_rev_per_customer_evidence_checking_cm_final,
           value = 'fx-evidence:checking & cm',color='green', ci=95)
#
sns.tsplot(data = cumulative_fx_rev_per_customer_evidence_checking_mmb_final,
           value = 'fx-evidence:checking & mmb',color='blue', ci=95)
sns.tsplot(data = cumulative_fx_rev_per_customer_evidence_checking_loc_final,
           value = 'fx-evidence:checking&loc',color='orange', ci=95)
plt.legend(['fx-no evidence','fx-evidence:checking & cm','fx-evidence:checking & mmb','fx-evidence:checking&loc'],
          fontsize='large')
plt.title('GP for fx with different starting evidence')
plt.ylabel('GP')
plt.xlabel('Simulated week no.')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
sns.tsplot(data = cumulative_fx_rev_per_customer_no_evidence_final,value = 'product 5-no evidence',color='red', ci=95)
sns.tsplot(data = cumulative_fx_rev_per_customer_evidence_checking_cm_final,
           value = 'product 5-evidenceproduct 7 & product 2',color='green', ci=95)
#
sns.tsplot(data = cumulative_fx_rev_per_customer_evidence_checking_mmb_final,
           value = 'product 5-evidenceproduct 7 & product 3',color='blue', ci=95)
sns.tsplot(data = cumulative_fx_rev_per_customer_evidence_checking_loc_final,
           value = 'product 5-evidence:product 7 &product 4',color='orange', ci=95)
plt.legend(['product 5-no evidence','product 5-evidenceproduct 7 & product 2','product 5-evidence:product 7 & product 3',
            'product 5-evidence:product 7&product 4'])
plt.title('Expected revenue for product 5 with different starting evidence')
plt.ylabel('GP')
plt.xlabel('Simulated week no.')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
sns.tsplot(data = cumulative_cmma_rev_per_customer_no_evidence_final,value = 'cmma-no evidence',color='red', ci=95)
sns.tsplot(data = cumulative_cmma_rev_per_customer_evidence_checking_cm_final,
           value = 'cmma-evidence:checking & cm',color='green', ci=95)
#
sns.tsplot(data = cumulative_cmma_rev_per_customer_evidence_checking_mmb_final,
           value = 'cmma-evidence:checking & mmb',color='blue', ci=95)
sns.tsplot(data = cumulative_cmma_rev_per_customer_evidence_checking_loc_final,
           value = 'cmma-evidence:checking&loc',color='orange', ci=95)
plt.legend(['cmma-no evidence','cmma-evidence:checking & cm','cmma-evidence:checking & mmb','cmma-evidence:checking&loc'])
plt.title('GP for cmma with different starting evidence')
plt.ylabel('GP')
plt.xlabel('Simulated week no.')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
sns.tsplot(data = cumulative_es_rev_per_customer_no_evidence_final,value = 'es-no evidence',color='red', ci=95)
sns.tsplot(data = cumulative_es_rev_per_customer_evidence_checking_cm_final,
           value = 'es-evidence:checking & cm',color='green', ci=95)
#
sns.tsplot(data = cumulative_es_rev_per_customer_evidence_checking_mmb_final,
           value = 'es-evidence:checking & mmb',color='blue', ci=95)
sns.tsplot(data = cumulative_es_rev_per_customer_evidence_checking_loc_final,
           value = 'es-evidence:checking&loc',color='orange', ci=95)
plt.legend(['es-no evidence','es-evidence:checking & cm','es-evidence:checking & mmb','es-evidence:checking&loc'],
          fontsize='large')
plt.title('GP for es with different starting evidence')
plt.ylabel('GP')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
sns.tsplot(data = cumulative_checking_customers_percent_no_evidence_final,value = 'checking-no evidence',color='red', ci=95)
sns.tsplot(data = cumulative_checking_customers_percent_evidence_checking_cm_final,
           value = 'cm-evidence:checking & cm',color='green', ci=95)
#
sns.tsplot(data = cumulative_checking_customers_percent_evidence_checking_mmb_final,
           value = 'cm-evidence:checking & mmb',color='blue', ci=95)
sns.tsplot(data = cumulative_checking_customers_percent_evidence_checking_loc_final,
           value = 'checking-evidence:checking&loc',color='orange', ci=95)
plt.legend(['checking-no evidence','checking-evidence:checking & cm','checking-evidence:checking & mmb','checking-evidence:checking&loc'])
plt.title('Percent of customers with checking with different starting evidence')
plt.ylabel('Percent')
plt.xlabel('Simulated week no.')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
sns.tsplot(data = cumulative_cm_customers_percent_no_evidence_final,value = 'cm-no evidence',color='red', ci=95)
sns.tsplot(data = cumulative_cm_customers_percent_evidence_checking_cm_final,
           value = 'cm-evidence:checking & cm',color='green', ci=95)
#
sns.tsplot(data = cumulative_cm_customers_percent_evidence_checking_mmb_final,
           value = 'cm-evidence:checking & mmb',color='blue', ci=95)
sns.tsplot(data = cumulative_cm_customers_percent_evidence_checking_loc_final,
           value = 'cm-evidence:checking&loc',color='orange', ci=95)
plt.legend(['cm-no evidence','cm-evidence:checking & cm','cm-evidence:checking & mmb','cm-evidence:checking&loc'],
          fontsize='large')
plt.title('Percent of customers with cm with different starting evidence')
plt.ylabel('Percent')
plt.xlabel('Simulated week no.')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
sns.tsplot(data = cumulative_cmma_customers_percent_no_evidence_final,value = 'cmma-no evidence',color='red', ci=95)
sns.tsplot(data = cumulative_cmma_customers_percent_evidence_checking_cm_final,
           value = 'cmma-evidence:checking & cm',color='green', ci=95)
#
sns.tsplot(data = cumulative_cmma_customers_percent_evidence_checking_mmb_final,
           value = 'cmma-evidence:checking & mmb',color='blue', ci=95)
sns.tsplot(data = cumulative_cmma_customers_percent_evidence_checking_loc_final,
           value = 'cmma-evidence:checking&loc',color='orange', ci=95)
plt.legend(['cmma-no evidence','cmma-evidence:checking & cm','cmma-evidence:checking & mmb',
            'cmma-evidence:checking&loc'],fontsize='large')
plt.title('Percent of customers with cmma with different starting evidence')
plt.ylabel('Percent')
plt.xlabel('Simulated week no.')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
sns.tsplot(data = cumulative_loc_customers_percent_no_evidence_final,value = 'loc-no evidence',color='red', ci=95)
sns.tsplot(data = cumulative_loc_customers_percent_evidence_checking_cm_final,
           value = 'loc-evidence:checking & cm',color='green', ci=95)
#
sns.tsplot(data = cumulative_loc_customers_percent_evidence_checking_mmb_final,
           value = 'loc-evidence:checking & mmb',color='blue', ci=95)
sns.tsplot(data = cumulative_loc_customers_percent_evidence_checking_loc_final,
           value = 'loc-evidence:checking&loc',color='orange', ci=95)
plt.legend(['loc-no evidence','loc-evidence:checking & cm','loc-evidence:checking & mmb','loc-evidence:checking&loc'])
plt.title('Percent of customers with loc with different starting evidence')
plt.ylabel('Percent')
plt.xlabel('Simulated week no.')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
sns.tsplot(data = cumulative_fx_customers_percent_no_evidence_final,value = 'fx-no evidence',color='red', ci=95)
sns.tsplot(data = cumulative_fx_customers_percent_evidence_checking_cm_final,
           value = 'fx-evidence:checking & cm',color='green', ci=95)
#
sns.tsplot(data = cumulative_fx_customers_percent_evidence_checking_mmb_final,
           value = 'fx-evidence:checking & mmb',color='blue', ci=95)
sns.tsplot(data = cumulative_fx_customers_percent_evidence_checking_loc_final,
           value = 'fx-evidence:checking&loc',color='orange', ci=95)
plt.legend(['fx-no evidence','fx-evidence:checking & cm','fx-evidence:checking & mmb','fx-evidence:checking&loc'],
          fontsize='large')
plt.title('Percent of customers with fx with different starting evidence')
plt.ylabel('Percent')
plt.xlabel('Simulated week no.')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
sns.tsplot(data = cumulative_es_customers_percent_no_evidence_final,value = 'es-no evidence',color='red', ci=95)
sns.tsplot(data = cumulative_es_customers_percent_evidence_checking_cm_final,
           value = 'es-evidence:checking & cm',color='green', ci=95)
#
sns.tsplot(data = cumulative_es_customers_percent_evidence_checking_mmb_final,
           value = 'es-evidence:checking & mmb',color='blue', ci=95)
sns.tsplot(data = cumulative_es_customers_percent_evidence_checking_loc_final,
           value = 'es-evidence:checking&loc',color='orange', ci=95)
plt.legend(['es-no evidence','es-evidence:checking & cm','es-evidence:checking & mmb','es-evidence:checking&loc'])
plt.title('Percent of customers with es with different starting evidence')
plt.ylabel('Percent')
plt.xlabel('Simulated week no.')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
sns.tsplot(data = cumulative_mmb_customers_percent_no_evidence_final,value = 'mmb-no evidence',color='red', ci=95)
sns.tsplot(data = cumulative_mmb_customers_percent_evidence_checking_cm_final,
           value = 'mmb-evidence:checking & cm',color='green', ci=95)
#
sns.tsplot(data = cumulative_mmb_customers_percent_evidence_checking_mmb_final,
           value = 'mmb-evidence:checking & mmb',color='blue', ci=95)
sns.tsplot(data = cumulative_mmb_customers_percent_evidence_checking_loc_final,
           value = 'mmb-evidence:checking&loc',color='orange', ci=95)
plt.legend(['mmb-no evidence','mmb-evidence:checking & cm','mmb-evidence:checking & mmb','mmb-evidence:checking&loc'])
plt.title('Percent of customers with mmb with different starting evidence')
plt.ylabel('Percent')
plt.xlabel('Simulated week no.')

sns.set(style="darkgrid")
plt.figure(figsize=(15,12))
sns.tsplot(data = cumulative_mmb_customers_percent_no_evidence_final,value = 'product 2-no evidence',color='red', ci=95)
sns.tsplot(data = cumulative_mmb_customers_percent_evidence_checking_cm_final,
           value = 'product 2-evidence:product-3 & product-4',color='green', ci=95)
#
sns.tsplot(data = cumulative_mmb_customers_percent_evidence_checking_mmb_final,
           value = 'product 2-evidence:product-3 & product-7',color='blue', ci=95)
sns.tsplot(data = cumulative_mmb_customers_percent_evidence_checking_loc_final,
           value = 'product 2-evidence:product-3&product 1',color='orange', ci=95)
plt.legend(['product 2-no evidence','product 2-evidence:product-3 & product-7','product 2-evidence:product-3 & product-4',
            'product 2-evidence:product-3&product-1'])
plt.title('Percent of customers with product 2 with different starting evidence')
plt.ylabel('Percent')
plt.xlabel('Simulated week no.')

cm_customers_percent_no_evid = pd.DataFrame(cumulative_cm_customers_percent_no_evidence_final,index=['iteration_1','iteration_2','iteration_3']).apply(
lambda x: np.mean(x),axis=0)

cmma_customers_percent_no_evid= pd.DataFrame(cumulative_cmma_customers_percent_no_evidence_final,index=['iteration_1','iteration_2','iteration_3']).apply(
lambda x: np.mean(x),axis=0)

loc_customer_percent_no_evid = pd.DataFrame(cumulative_loc_customers_percent_no_evidence_final,index=['iteration_1','iteration_2','iteration_3']).apply(
lambda x: np.mean(x),axis=0)

fx_customer_percent_no_evid = pd.DataFrame(cumulative_fx_customers_percent_no_evidence_final,index=['iteration_1','iteration_2','iteration_3']).apply(
lambda x: np.mean(x),axis=0)

es_customer_percent_no_evid = pd.DataFrame(cumulative_es_customers_percent_no_evidence_final,index=['iteration_1','iteration_2','iteration_3']).apply(
lambda x: np.mean(x),axis=0)

mmb_customer_percent_no_evid = pd.DataFrame(cumulative_mmb_customers_percent_no_evidence_final,index=['iteration_1','iteration_2','iteration_3']).apply(
lambda x: np.mean(x),axis=0)

checking_customer_percent_no_evid = pd.DataFrame(cumulative_checking_customers_percent_no_evidence_final,index=['iteration_1','iteration_2','iteration_3']).apply(
lambda x: np.mean(x),axis=0)

product_percent_no_evid_df = pd.DataFrame(data=np.array([cm_customers_percent_no_evid,cmma_customers_percent_no_evid, loc_customer_percent_no_evid,
                  fx_customer_percent_no_evid,es_customer_percent_no_evid,mmb_customer_percent_no_evid ,
                  checking_customer_percent_no_evid]).T,
             columns=['cm_prob','cmma_prob','loc_prob','fx_prob',"es_prob","mmb_prob","checking_prob"])

product_percent_no_evid_df.index.names=['week_n']

product_percent_no_evid_df.reset_index(inplace=True)

product_percent_no_evid_df.to_csv("product_probability_weekly_noEvidence.csv")

def esp_client_lifetime():
        """Draws from a distribution of client lifetimes (in months) from 2013-2016.
        Return the number of weeks that a client will be alive.

        A client needs to be generating revenue for at least three months, and not
        have generated revenue for three months to be considred a
        'client lifetime'. It is possible for a single client to have Multiple
        'client lifetimes' that feed into the parameters for the Exponential
        distribution.

        Multiply the result by 4 to turn months into weeks"""
        exponential_lifetime_parameters = (2.9999999999982676, 11.500665661185888)
        return round(stats.expon(*exponential_lifetime_parameters ).rvs())*4

def get_average_per_time_period(sim1,sim2,sim3,sim4):
    """This is used to calculated the average across the three simulations. 
    Useful for calculting LTV without having confidence interals"""
    avg_sim1 = []
    avg_sim2 = []
    avg_sim3 = []
    avg_sim4 = []
    for week_n in range(len(sim1[0])):
        avg_sim1.append(np.mean([sim1[0][week_n], sim1[1][week_n], sim1[2][week_n]]))
    for week_n in range(len(sim2[0])):
        avg_sim2.append(np.mean([sim2[0][week_n], sim2[1][week_n], sim2[2][week_n]]))
    for week_n in range(len(sim3[0])):
        avg_sim3.append(np.mean([sim3[0][week_n], sim3[1][week_n], sim3[2][week_n]]))
    for week_n in range(len(sim4[0])):
        avg_sim4.append(np.mean([sim4[0][week_n], sim4[1][week_n], sim4[2][week_n]]))   
    return avg_sim1,avg_sim2,avg_sim3, avg_sim4

cumulative_mmb_customers_percent_evidence_checking_loc_final[0][1]

# Get average GP and percent of customer across simulation for MMB
avg_cust_percent_mmb_no_evidence, avg_cust_percent_mmb_evidence_checking_cm,avg_cust_percent_mmb_evidence_checking_mmb, avg_cust_percent_mmb_evidence_checking_loc = get_average_per_time_period(cumulative_mmb_customers_percent_no_evidence_final,
                            cumulative_mmb_customers_percent_evidence_checking_cm_final,
                           cumulative_mmb_customers_percent_evidence_checking_mmb_final,
                           cumulative_mmb_customers_percent_evidence_checking_loc_final)

avg_weekly_rev_mmb_no_evidence, avg_weekly_rev_mmb_evidence_checking_cm,avg_weekly_rev_mmb_evidence_checking_mmb, avg_weekly_rev_mmb_evidence_checking_loc = get_average_per_time_period(cumulative_mmb_rev_per_customer_no_evidence_final,
                            cumulative_mmb_rev_per_customer_evidence_checking_cm_final,
                           cumulative_mmb_rev_per_customer_evidence_checking_mmb_final,
                           cumulative_mmb_rev_per_customer_evidence_checking_loc_final)





#go throughsimulated time for mmb customers

def ltv(avg_percent1,avg_rev1,avg_percent2,avg_rev2,avg_percent3,avg_rev3,avg_percent4,avg_rev4):
    """Generate the ltv based off the the simulation data for weekly expected GP and perent of customer with product.
    Uses the expected percent of customers to have a product at a time step multipled by the expected revenue at
    that time step"""
    
    client_lifetimes = []
    # generate 5000 customers
    for _ in range(5000):
        lifetime = esp_client_lifetime()
        if lifetime <104: # this is how long we ran our simulation for
            client_lifetimes.append(lifetime)

    total_live_clients = 0
    sim1_rev_weekly = []
    sim2_rev_weekly = []
    sim3_rev_weekly = []
    sim4_rev_weekly = []
    
    for week_n in range(int(max(client_lifetimes)+2)):
        remove_lifetimes = []
        for client_lifetime in client_lifetimes:
            if week_n>client_lifetime:
                remove_lifetimes.append(client_lifetime)

        #  customer churned
        [client_lifetimes.remove(i) for i in remove_lifetimes]
        # live clients
        total_live_clients = len(client_lifetimes)
        if round(total_live_clients * avg_percent1[week_n]) == 0:
            pass
        else:
#             clients1 = round(total_live_clients * avg_percent1[week_n])
#             print(clients1,'cleints 1')
#             print(avg_rev1[week_n],'rev week n')
            sim1_rev_weekly.append(
             avg_rev1[week_n])#/clients1)
        if round(total_live_clients * avg_percent2[week_n]) == 0:
            pass
        else:
            #clients2 = round(total_live_clients * avg_percent2[week_n])

            sim2_rev_weekly.append(
            
                 avg_rev2[week_n])#/clients2 )
        if round(total_live_clients * avg_percent3[week_n]) == 0:
            pass
        else:
            #clients3 = round(total_live_clients * avg_percent3[week_n])

            sim3_rev_weekly.append(
           
                 avg_rev3[week_n])#/clients3 )
        if round(total_live_clients * avg_percent4[week_n]) == 0:
            pass
        else:
            #clients4 = round(total_live_clients * avg_percent4[week_n])

            sim4_rev_weekly.append(
 
                 avg_rev4[week_n])#/#clients4 )
    return sum(sim1_rev_weekly),sum(sim2_rev_weekly),sum(sim3_rev_weekly),sum(sim4_rev_weekly)
        

#go throughsimulated time for mmb customers

def expected_ltv(avg_percent1,avg_rev1,avg_percent2,avg_rev2,avg_percent3,avg_rev3,avg_percent4,avg_rev4):
    """Generate the ltv based off the the simulation data for weekly expected GP and perent of customer with product.
    Uses the expected percent of customers to have a product at a time step multipled by the expected revenue at
    that time step.
    Using expectation of the number of clients who have this product"""
    
    client_lifetimes = []
    # generate 5000 customers
    for _ in range(5000):
        lifetime = esp_client_lifetime()
        if lifetime <104: # this is how long we ran our simulation for
            client_lifetimes.append(lifetime)

    total_live_clients = 0
    sim1_rev_weekly = []
    sim2_rev_weekly = []
    sim3_rev_weekly = []
    sim4_rev_weekly = []
    
    for week_n in range(int(max(client_lifetimes)+2)):
        remove_lifetimes = []
        for client_lifetime in client_lifetimes:
            if week_n>client_lifetime:
                remove_lifetimes.append(client_lifetime)

        #  customer churned
        [client_lifetimes.remove(i) for i in remove_lifetimes]
        # live clients
        total_live_clients = len(client_lifetimes)
        if round(total_live_clients * avg_percent1[week_n]) == 0:
            pass
        else:
#             clients1 = round(total_live_clients * avg_percent1[week_n])
#             print(clients1,'cleints 1')
#             print(avg_rev1[week_n],'rev week n')
            sim1_rev_weekly.append(
             avg_rev1[week_n]* avg_percent1[week_n])#/clients1)
        if round(total_live_clients * avg_percent2[week_n]) == 0:
            pass
        else:
            #clients2 = round(total_live_clients * avg_percent2[week_n])

            sim2_rev_weekly.append(
            
                 avg_rev2[week_n]* avg_percent2[week_n])#/clients2 )
        if round(total_live_clients * avg_percent3[week_n]) == 0:
            pass
        else:
            #clients3 = round(total_live_clients * avg_percent3[week_n])

            sim3_rev_weekly.append(
           
                 avg_rev3[week_n]* avg_percent3[week_n])#/clients3 )
        if round(total_live_clients * avg_percent4[week_n]) == 0:
            pass
        else:
            #clients4 = round(total_live_clients * avg_percent4[week_n])

            sim4_rev_weekly.append(
 
                 avg_rev4[week_n]* avg_percent4[week_n])#/#clients4 )
    return sum(sim1_rev_weekly),sum(sim2_rev_weekly),sum(sim3_rev_weekly),sum(sim4_rev_weekly)
        

#go throughsimulated time for mmb customers

def expected_ltv_loc(avg_percent1,avg_rev1,avg_percent2,avg_rev2,avg_percent3,avg_rev3,avg_percent4,avg_rev4):
    """Generate the ltv based off the the simulation data for weekly expected GP and perent of customer with product.
    Use a exponential distribution, not enoug hdata for expon weibull.
    Also, compute NPV of revenue.
    Uses expectation of number of clients that have this product
    """
    

    federal_funds_rate = .0075 # May 11, 2017
    inflation_rate = .025 # March 2017
    def yearly_to_weekly_interest_rate_conversion(yearly_interest_rate=federal_funds_rate+inflation_rate):
        """Convert from a yearly rate to a weekly rate using the following
        equation.
        Effective rate for period = (1 + annual rate)**(1 / # of periods) – 1
        """
        weekly_interest_rate  = ((1 + yearly_interest_rate)**(1/52))-1
        return weekly_interest_rate
    weekly_interest = yearly_to_weekly_interest_rate_conversion()
    
    def letters_of_credit_weekly_rev( week_n,one = -663.9800000000298, two = 869.07882491183921,
                                     interest =weekly_interest ):
        """ This gives a predicted weekly GP from and Exponential  distribution.
        The default parameters here are from 2016.
        Also, performs the NPV calculation"""
        loc_mweekly = stats.expon.rvs(one,two)/4
        total_wekly_rev = loc_mweekly / (1+interest)**week_n
        return total_wekly_rev
        
    client_lifetimes = []
    # generate 5000 customers
    for _ in range(5000):
        lifetime = esp_client_lifetime()
        if lifetime <104: # this is how long we ran our simulation for
            client_lifetimes.append(lifetime)

    total_live_clients = 0
    sim1_rev_weekly = []
    sim2_rev_weekly = []
    sim3_rev_weekly = []
    sim4_rev_weekly = []
    
    for week_n in range(int(max(client_lifetimes)+2)):
        remove_lifetimes = []
        for client_lifetime in client_lifetimes:
            if week_n>client_lifetime:
                remove_lifetimes.append(client_lifetime)


        [client_lifetimes.remove(i) for i in remove_lifetimes]
        total_live_clients = len(client_lifetimes)
        if round(total_live_clients * avg_percent1[week_n]) == 0:
            pass
        else:
            #clients1 = round(total_live_clients * avg_percent1[week_n])

            sim1_rev_weekly.append(
              letters_of_credit_weekly_rev(week_n)* avg_percent1[week_n])#/clients1)
        if round(total_live_clients * avg_percent2[week_n]) == 0:
            pass
        else:
            #clients2 = round(total_live_clients * avg_percent2[week_n])

            sim2_rev_weekly.append(
             
                 letters_of_credit_weekly_rev(week_n)* avg_percent1[week_n])#/clients2 )
        if round(total_live_clients * avg_percent3[week_n]) == 0:
            pass
        else:
            #clients3 = round(total_live_clients * avg_percent3[week_n])

            sim3_rev_weekly.append(
       
                 letters_of_credit_weekly_rev(week_n)* avg_percent1[week_n])#/clients3 )
        if round(total_live_clients * avg_percent4[week_n]) == 0:
            pass
        else:
            #clients4 = round(total_live_clients * avg_percent4[week_n])

            sim4_rev_weekly.append(
        
                 letters_of_credit_weekly_rev(week_n)* avg_percent1[week_n])#/clients4 )
    return sum(sim1_rev_weekly),sum(sim2_rev_weekly),sum(sim3_rev_weekly),sum(sim4_rev_weekly)
        

#go throughsimulated time for mmb customers

def ltv_loc(avg_percent1,avg_rev1,avg_percent2,avg_rev2,avg_percent3,avg_rev3,avg_percent4,avg_rev4):
    """Generate the ltv based off the the simulation data for weekly expected GP and perent of customer with product.
    Use a exponential distribution, not enoug hdata for expon weibull.
    Also, compute NPV of revenue.
    """
    

    federal_funds_rate = .0075 # May 11, 2017
    inflation_rate = .025 # March 2017
    def yearly_to_weekly_interest_rate_conversion(yearly_interest_rate=federal_funds_rate+inflation_rate):
        """Convert from a yearly rate to a weekly rate using the following
        equation.
        Effective rate for period = (1 + annual rate)**(1 / # of periods) – 1
        """
        weekly_interest_rate  = ((1 + yearly_interest_rate)**(1/52))-1
        return weekly_interest_rate
    weekly_interest = yearly_to_weekly_interest_rate_conversion()
    
    def letters_of_credit_weekly_rev( week_n,one = -663.9800000000298, two = 869.07882491183921,
                                     interest =weekly_interest ):
        """ This gives a predicted weekly GP from and Exponential  distribution.
        The default parameters here are from 2016.
        Also, performs the NPV calculation"""
        loc_mweekly = stats.expon.rvs(one,two)/4
        total_wekly_rev = loc_mweekly / (1+interest)**week_n
        return total_wekly_rev
        
    client_lifetimes = []
    # generate 5000 customers
    for _ in range(5000):
        lifetime = esp_client_lifetime()
        if lifetime <104: # this is how long we ran our simulation for
            client_lifetimes.append(lifetime)

    total_live_clients = 0
    sim1_rev_weekly = []
    sim2_rev_weekly = []
    sim3_rev_weekly = []
    sim4_rev_weekly = []
    
    for week_n in range(int(max(client_lifetimes)+2)):
        remove_lifetimes = []
        for client_lifetime in client_lifetimes:
            if week_n>client_lifetime:
                remove_lifetimes.append(client_lifetime)


        [client_lifetimes.remove(i) for i in remove_lifetimes]
        total_live_clients = len(client_lifetimes)
        if round(total_live_clients * avg_percent1[week_n]) == 0:
            pass
        else:
            #clients1 = round(total_live_clients * avg_percent1[week_n])

            sim1_rev_weekly.append(
              letters_of_credit_weekly_rev(week_n))#/clients1)
        if round(total_live_clients * avg_percent2[week_n]) == 0:
            pass
        else:
            #clients2 = round(total_live_clients * avg_percent2[week_n])

            sim2_rev_weekly.append(
             
                 letters_of_credit_weekly_rev(week_n))#/clients2 )
        if round(total_live_clients * avg_percent3[week_n]) == 0:
            pass
        else:
            #clients3 = round(total_live_clients * avg_percent3[week_n])

            sim3_rev_weekly.append(
       
                 letters_of_credit_weekly_rev(week_n))#/clients3 )
        if round(total_live_clients * avg_percent4[week_n]) == 0:
            pass
        else:
            #clients4 = round(total_live_clients * avg_percent4[week_n])

            sim4_rev_weekly.append(
        
                 letters_of_credit_weekly_rev(week_n))#/clients4 )
    return sum(sim1_rev_weekly),sum(sim2_rev_weekly),sum(sim3_rev_weekly),sum(sim4_rev_weekly)
        

ltv_mmb_no_evidence, ltv_mmb_evidence_checking_cm,ltv_mmb_evidence_checking_mmb, ltv_mmb_evidence_checking_loc  = ltv(avg_cust_percent_mmb_no_evidence,avg_weekly_rev_mmb_no_evidence,
   avg_cust_percent_mmb_evidence_checking_cm, avg_weekly_rev_mmb_evidence_checking_cm,
   avg_cust_percent_mmb_evidence_checking_mmb, avg_weekly_rev_mmb_evidence_checking_mmb,
   avg_cust_percent_mmb_evidence_checking_loc, avg_weekly_rev_mmb_evidence_checking_loc)

plt.figure(figsize=(12,8))
plt.title('Expected LTV MMB ')
fig = sns.barplot(x=['mmb-no-evidence','mmb-evidence-checking-cm','mmb-evidence-checking-mmb','mmb-evidence-checking-loc'],
           y=[ltv_mmb_no_evidence,ltv_mmb_evidence_checking_cm, ltv_mmb_evidence_checking_loc,
              ltv_mmb_evidence_checking_mmb])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick) 

ltv_mmb_no_evidence_expect, ltv_mmb_evidence_checking_cm_expect,ltv_mmb_evidence_checking_mmb_expect, ltv_mmb_evidence_checking_loc_expect  = expected_ltv(avg_cust_percent_mmb_no_evidence,avg_weekly_rev_mmb_no_evidence,
   avg_cust_percent_mmb_evidence_checking_cm, avg_weekly_rev_mmb_evidence_checking_cm,
   avg_cust_percent_mmb_evidence_checking_mmb, avg_weekly_rev_mmb_evidence_checking_mmb,
   avg_cust_percent_mmb_evidence_checking_loc, avg_weekly_rev_mmb_evidence_checking_loc)

plt.figure(figsize=(12,8))
plt.title('Expected LTV MMB - weighted by  product probabilities ')
fig = sns.barplot(x=['mmb-no-evidence','mmb-evidence-checking-cm','mmb-evidence-checking-mmb','mmb-evidence-checking-loc'],
           y=[ltv_mmb_no_evidence_expect,ltv_mmb_evidence_checking_cm_expect, ltv_mmb_evidence_checking_loc_expect,
              ltv_mmb_evidence_checking_mmb_expect])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick) 

print('mmb no evidence',ltv_mmb_no_evidence,'mmb evidece = checking cm',ltv_mmb_evidence_checking_cm, 
      'ltv mmb evidence checking loc',ltv_mmb_evidence_checking_loc,
           'ltv mmb evidence checking mmb',   ltv_mmb_evidence_checking_mmb)

# Get average GP and percent of customer across simulation for checking
avg_cust_percent_es_no_evidence, avg_cust_percent_es_evidence_checking_cm,avg_cust_percent_es_evidence_checking_mmb, avg_cust_percent_es_evidence_checking_loc = get_average_per_time_period(cumulative_es_customers_percent_no_evidence_final,
                            cumulative_es_customers_percent_evidence_checking_cm_final,
                           cumulative_es_customers_percent_evidence_checking_mmb_final,
                           cumulative_es_customers_percent_evidence_checking_loc_final)

avg_weekly_rev_es_no_evidence, avg_weekly_rev_es_evidence_checking_cm,avg_weekly_rev_es_evidence_checking_mmb, avg_weekly_rev_es_evidence_checking_loc = get_average_per_time_period(cumulative_es_rev_per_customer_no_evidence_final,
                            cumulative_es_rev_per_customer_evidence_checking_cm_final,
                           cumulative_es_rev_per_customer_evidence_checking_mmb_final,
                           cumulative_es_rev_per_customer_evidence_checking_loc_final)





ltv_es_no_evidence, ltv_es_evidence_checking_cm,ltv_es_evidence_checking_mmb, ltv_es_evidence_checking_loc  = ltv(avg_cust_percent_es_no_evidence,avg_weekly_rev_es_no_evidence,
   avg_cust_percent_es_evidence_checking_cm, avg_weekly_rev_es_evidence_checking_cm,
   avg_cust_percent_es_evidence_checking_mmb, avg_weekly_rev_es_evidence_checking_mmb,
   avg_cust_percent_es_evidence_checking_loc, avg_weekly_rev_es_evidence_checking_loc)

plt.figure(figsize=(12,8))
plt.title('LTV ES')
fig = sns.barplot(x=[ 'ES-no-evidence', 'ES-evidence-checking-cm', 'ES-evidence-checking-loc', 'ES-evidence-checking-mmb'],
           y=[ltv_es_no_evidence,ltv_es_evidence_checking_cm, ltv_es_evidence_checking_loc,
              ltv_es_evidence_checking_mmb])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick) 

ltv_es_no_evidence_expect, ltv_es_evidence_checking_cm_expect,ltv_es_evidence_checking_mmb_expect, ltv_es_evidence_checking_loc_expect  = expected_ltv(avg_cust_percent_es_no_evidence,avg_weekly_rev_es_no_evidence,
   avg_cust_percent_es_evidence_checking_cm, avg_weekly_rev_es_evidence_checking_cm,
   avg_cust_percent_es_evidence_checking_mmb, avg_weekly_rev_es_evidence_checking_mmb,
   avg_cust_percent_es_evidence_checking_loc, avg_weekly_rev_es_evidence_checking_loc)

plt.figure(figsize=(12,8))
plt.title('Expected LTV ES - weighted by product probability')
fig = sns.barplot(x=[ 'ES-no-evidence', 'ES-evidence-checking-cm', 'ES-evidence-checking-loc', 'ES-evidence-checking-mmb'],
           y=[ltv_es_no_evidence_expect,ltv_es_evidence_checking_cm_expect, ltv_es_evidence_checking_loc_expect,
              ltv_es_evidence_checking_mmb_expect])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick) 

print(ltv_checking_no_evidence,ltv_checking_evidence_checking_cm, ltv_checking_evidence_checking_loc,
              ltv_checking_evidence_checking_mmb)

# Get average GP and percent of customer across simulation for checking
avg_cust_percent_cm_no_evidence, avg_cust_percent_cm_evidence_checking_cm,avg_cust_percent_cm_evidence_checking_mmb, avg_cust_percent_cm_evidence_checking_loc = get_average_per_time_period(cumulative_cm_customers_percent_no_evidence_final,
                            cumulative_cm_customers_percent_evidence_checking_cm_final,
                           cumulative_cm_customers_percent_evidence_checking_mmb_final,
                           cumulative_cm_customers_percent_evidence_checking_loc_final)

avg_weekly_rev_cm_no_evidence, avg_weekly_rev_cm_evidence_checking_cm,avg_weekly_rev_cm_evidence_checking_mmb, avg_weekly_rev_cm_evidence_checking_loc = get_average_per_time_period(cumulative_cm_rev_per_customer_no_evidence_final,
                            cumulative_cm_rev_per_customer_evidence_checking_cm_final,
                           cumulative_cm_rev_per_customer_evidence_checking_mmb_final,
                           cumulative_cm_rev_per_customer_evidence_checking_loc_final)




ltv_cm_no_evidence, ltv_cm_evidence_checking_cm,ltv_cm_evidence_checking_mmb, ltv_cm_evidence_checking_loc  = ltv(avg_cust_percent_cm_no_evidence, avg_weekly_rev_cm_no_evidence,
   avg_cust_percent_cm_evidence_checking_cm, avg_weekly_rev_cm_evidence_checking_cm,
   avg_cust_percent_cm_evidence_checking_mmb, avg_weekly_rev_cm_evidence_checking_mmb,
   avg_cust_percent_cm_evidence_checking_loc, avg_weekly_rev_cm_evidence_checking_loc)


plt.figure(figsize=(12,8))
plt.title('LTV cm')
fig = sns.barplot(x=['cm-no-evidence','cm-evidence-checking-cm','cm-evidence-checking-loc','cm-evidence-checking-mmb'],
           y=[ltv_cm_no_evidence,ltv_cm_evidence_checking_cm, ltv_cm_evidence_checking_loc,
              ltv_cm_evidence_checking_mmb])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick) 

ltv_cm_no_evidence_expect, ltv_cm_evidence_checking_cm_expect,ltv_cm_evidence_checking_mmb_expect, ltv_cm_evidence_checking_loc_expect  = expected_ltv(avg_cust_percent_cm_no_evidence, avg_weekly_rev_cm_no_evidence,
   avg_cust_percent_cm_evidence_checking_cm, avg_weekly_rev_cm_evidence_checking_cm,
   avg_cust_percent_cm_evidence_checking_mmb, avg_weekly_rev_cm_evidence_checking_mmb,
   avg_cust_percent_cm_evidence_checking_loc, avg_weekly_rev_cm_evidence_checking_loc)


plt.figure(figsize=(12,8))
plt.title('LTV cm- weighted by product probability')
fig = sns.barplot(x=['cm-no-evidence','cm-evidence-checking-cm','cm-evidence-checking-loc','cm-evidence-checking-mmb'],
           y=[ltv_cm_no_evidence_expect,ltv_cm_evidence_checking_cm_expect, ltv_cm_evidence_checking_loc_expect,
              ltv_cm_evidence_checking_mmb_expect])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick) 

print('ltv cm no evidence',ltv_cm_no_evidence,'cm evidece = checking cm',ltv_cm_evidence_checking_cm, 
      'ltv cm evidence=checking loc',ltv_cm_evidence_checking_loc,
           'ltv cm evidence =checking mmb',   ltv_cm_evidence_checking_mmb)

# Get average GP and percent of customer across simulation for checking
avg_cust_percent_fx_no_evidence, avg_cust_percent_fx_evidence_checking_cm,avg_cust_percent_fx_evidence_checking_mmb, avg_cust_percent_fx_evidence_checking_loc = get_average_per_time_period(cumulative_fx_customers_percent_no_evidence_final,
                            cumulative_fx_customers_percent_evidence_checking_cm_final,
                           cumulative_fx_customers_percent_evidence_checking_mmb_final,
                           cumulative_fx_customers_percent_evidence_checking_loc_final)

avg_weekly_rev_fx_no_evidence, avg_weekly_rev_fx_evidence_checking_cm,avg_weekly_rev_fx_evidence_checking_mmb, avg_weekly_rev_fx_evidence_checking_loc = get_average_per_time_period(cumulative_fx_rev_per_customer_no_evidence_final,
                            cumulative_fx_rev_per_customer_evidence_checking_cm_final,
                           cumulative_fx_rev_per_customer_evidence_checking_mmb_final,
                           cumulative_fx_rev_per_customer_evidence_checking_loc_final)




ltv_fx_no_evidence, ltv_fx_evidence_checking_cm,ltv_fx_evidence_checking_mmb, ltv_fx_evidence_checking_loc  = ltv(avg_cust_percent_fx_no_evidence,avg_weekly_rev_fx_no_evidence,
   avg_cust_percent_fx_evidence_checking_cm, avg_weekly_rev_fx_evidence_checking_cm,
   avg_cust_percent_fx_evidence_checking_mmb, avg_weekly_rev_fx_evidence_checking_mmb,
   avg_cust_percent_fx_evidence_checking_loc, avg_weekly_rev_fx_evidence_checking_loc)


plt.figure(figsize=(12,8))
plt.title('LTV fx - average revenue per time period')
fig = sns.barplot(x=['fx-no-evidence','fx-evidence-checking-cm','fx-evidence-checking-loc','fx-evidence-checking-mmb'],
           y=[ltv_fx_no_evidence,ltv_fx_evidence_checking_cm, ltv_fx_evidence_checking_loc,
              ltv_fx_evidence_checking_mmb])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick) 

ltv_fx_no_evidence_expect, ltv_fx_evidence_checking_cm_expect,ltv_fx_evidence_checking_mmb_expect,ltv_fx_evidence_checking_loc_expect  = expected_ltv(avg_cust_percent_fx_no_evidence,avg_weekly_rev_fx_no_evidence,
   avg_cust_percent_fx_evidence_checking_cm, avg_weekly_rev_fx_evidence_checking_cm,
   avg_cust_percent_fx_evidence_checking_mmb, avg_weekly_rev_fx_evidence_checking_mmb,
   avg_cust_percent_fx_evidence_checking_loc, avg_weekly_rev_fx_evidence_checking_loc)


plt.figure(figsize=(12,8))
plt.title('LTV fx - weighted by product probabilities')
fig = sns.barplot(x=['fx-no-evidence','fx-evidence-checking-cm','fx-evidence-checking-loc','fx-evidence-checking-mmb'],
           y=[ltv_fx_no_evidence_expect,ltv_fx_evidence_checking_cm_expect, ltv_fx_evidence_checking_loc_expect,
              ltv_fx_evidence_checking_mmb_expect])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick) 

plt.figure(figsize=(15,8))
plt.title('LTV product 4')
fig = sns.barplot(x=['product 4-no-evidence','product 4-evidence-product 2-product 6',
                     'product 4-evidence-product 2-product 3','product 4-evidence-product 2-product 7'],
           y=[ltv_fx_no_evidence,ltv_fx_evidence_checking_cm, ltv_fx_evidence_checking_loc,
              ltv_fx_evidence_checking_mmb])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick) 

ltv_fx_no_evidence_expect, ltv_fx_evidence_checking_cm_expect,ltv_fx_evidence_checking_mmb_expect,ltv_fx_evidence_checking_loc_expect  = expected_ltv(avg_cust_percent_fx_no_evidence,avg_weekly_rev_fx_no_evidence,
   avg_cust_percent_fx_evidence_checking_cm, avg_weekly_rev_fx_evidence_checking_cm,
   avg_cust_percent_fx_evidence_checking_mmb, avg_weekly_rev_fx_evidence_checking_mmb,
   avg_cust_percent_fx_evidence_checking_loc, avg_weekly_rev_fx_evidence_checking_loc)


plt.figure(figsize=(15,8))
plt.title('Expected LTV product 4')
fig = sns.barplot(x=['product 4-no-evidence','product 4-evidence-product 2-product 6',
                     'product 4-evidence-product 2-product 3','product 4-evidence-product 2-product 7'],
           y=[ltv_fx_no_evidence_expect,ltv_fx_evidence_checking_cm_expect, ltv_fx_evidence_checking_loc_expect,
              ltv_fx_evidence_checking_mmb_expect])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick) 

print('ltv fx no evidence',ltv_fx_no_evidence,'fx evidece = checking cm',ltv_fx_evidence_checking_cm, 
      'ltv fx evidence=checking loc',ltv_fx_evidence_checking_loc,
           'ltv fx evidence =checking mmb',   ltv_fx_evidence_checking_mmb)

# Get average GP and percent of customer across simulation for checking
avg_cust_percent_checking_no_evidence, avg_cust_percent_checking_evidence_checking_cm,avg_cust_percent_checking_evidence_checking_mmb, avg_cust_percent_checking_evidence_checking_loc = get_average_per_time_period(cumulative_checking_customers_percent_no_evidence_final,
                            cumulative_checking_customers_percent_evidence_checking_cm_final,
                           cumulative_checking_customers_percent_evidence_checking_mmb_final,
                           cumulative_checking_customers_percent_evidence_checking_loc_final)

avg_weekly_rev_checking_no_evidence, avg_weekly_rev_checking_evidence_checking_cm,avg_weekly_rev_checking_evidence_checking_mmb, avg_weekly_rev_checking_evidence_checking_loc = get_average_per_time_period(cumulative_checking_rev_per_customer_no_evidence_final,
                            cumulative_checking_rev_per_customer_evidence_checking_cm_final,
                           cumulative_checking_rev_per_customer_evidence_checking_mmb_final,
                           cumulative_checking_rev_per_customer_evidence_checking_loc_final)




ltv_checking_no_evidence, ltv_checking_evidence_checking_cm,ltv_checking_evidence_checking_mmb, ltv_checking_evidence_checking_loc  = ltv(avg_cust_percent_checking_no_evidence,avg_weekly_rev_checking_no_evidence,
   avg_cust_percent_checking_evidence_checking_cm, avg_weekly_rev_checking_evidence_checking_cm,
   avg_cust_percent_checking_evidence_checking_mmb, avg_weekly_rev_checking_evidence_checking_mmb,
   avg_cust_percent_checking_evidence_checking_loc, avg_weekly_rev_checking_evidence_checking_loc)


plt.figure(figsize=(12,8))
plt.title('LTV checking - average revenues')
fig = sns.barplot(x=['checking-no-evidence','checking-evidence-checking-cm','checking-evidence-checking-loc',
               'checking-evidence-checking-mmb'],
           y=[ltv_checking_no_evidence,ltv_checking_evidence_checking_cm, ltv_checking_evidence_checking_loc,
              ltv_checking_evidence_checking_mmb])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick) 


ltv_checking_no_evidence_expect, ltv_checking_evidence_checking_cm_expect,ltv_checking_evidence_checking_mmb_expect, ltv_checking_evidence_checking_loc_expect  = expected_ltv(avg_cust_percent_checking_no_evidence,avg_weekly_rev_checking_no_evidence,
   avg_cust_percent_checking_evidence_checking_cm, avg_weekly_rev_checking_evidence_checking_cm,
   avg_cust_percent_checking_evidence_checking_mmb, avg_weekly_rev_checking_evidence_checking_mmb,
   avg_cust_percent_checking_evidence_checking_loc, avg_weekly_rev_checking_evidence_checking_loc)

plt.figure(figsize=(12,8))
plt.title('LTV checking - weighted by product probabilities')
fig = sns.barplot(x=['checking-no-evidence','checking-evidence-checking-cm','checking-evidence-checking-loc',
               'checking-evidence-checking-mmb'],
           y=[ltv_checking_no_evidence_expect,ltv_checking_evidence_checking_cm_expect, ltv_checking_evidence_checking_loc_expect,
              ltv_checking_evidence_checking_mmb_expect])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick)

print('ltv checking no evidence',ltv_checking_no_evidence,'checking evidece = checking cm',ltv_checking_evidence_checking_cm, 
      'ltv checking evidence=checking loc',ltv_checking_evidence_checking_loc,
           'ltv checking evidence =checking mmb',   ltv_checking_evidence_checking_mmb)

# Get average GP and percent of customer across simulation for checking
avg_cust_percent_loc_no_evidence, avg_cust_percent_loc_evidence_checking_cm,avg_cust_percent_loc_evidence_checking_mmb, avg_cust_percent_loc_evidence_checking_loc = get_average_per_time_period(cumulative_loc_customers_percent_no_evidence_final,
                            cumulative_loc_customers_percent_evidence_checking_cm_final,
                           cumulative_loc_customers_percent_evidence_checking_mmb_final,
                           cumulative_loc_customers_percent_evidence_checking_loc_final)

avg_weekly_rev_loc_no_evidence, avg_weekly_rev_loc_evidence_checking_cm,avg_weekly_rev_loc_evidence_checking_mmb, avg_weekly_rev_loc_evidence_checking_loc = get_average_per_time_period(cumulative_loc_rev_per_customer_no_evidence_final,
                            cumulative_loc_rev_per_customer_evidence_checking_cm_final,
                           cumulative_loc_rev_per_customer_evidence_checking_mmb_final,
                           cumulative_loc_rev_per_customer_evidence_checking_loc_final)




ltv_loc_no_evidence, ltv_loc_evidence_checking_cm,ltv_loc_evidence_checking_mmb, ltv_loc_evidence_checking_loc  = ltv(avg_cust_percent_loc_no_evidence,avg_weekly_rev_checking_no_evidence,
   avg_cust_percent_loc_evidence_checking_cm, avg_weekly_rev_loc_evidence_checking_cm,
   avg_cust_percent_loc_evidence_checking_mmb, avg_weekly_rev_loc_evidence_checking_mmb,
   avg_cust_percent_loc_evidence_checking_loc, avg_weekly_rev_loc_evidence_checking_loc)


plt.figure(figsize=(12,8))
plt.title('LTV loc - average revenues')
fig = sns.barplot(x=['loc-no-evidence','loc-evidence-checking-cm','loc-evidence-checking-loc',
               'loc-evidence-checking-mmb'],
           y=[ltv_loc_no_evidence,ltv_loc_evidence_checking_cm, ltv_loc_evidence_checking_loc,
              ltv_loc_evidence_checking_mmb])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick) 


ltv_loc_no_evidence_expect, ltv_loc_evidence_checking_cm_expect,ltv_loc_evidence_checking_mmb_expect, ltv_loc_evidence_checking_loc_expect  = expected_ltv(avg_cust_percent_loc_no_evidence,avg_weekly_rev_loc_no_evidence,
   avg_cust_percent_loc_evidence_checking_cm, avg_weekly_rev_loc_evidence_checking_cm,
   avg_cust_percent_loc_evidence_checking_mmb, avg_weekly_rev_loc_evidence_checking_mmb,
   avg_cust_percent_loc_evidence_checking_loc, avg_weekly_rev_loc_evidence_checking_loc)

plt.figure(figsize=(12,8))
plt.title('LTV loc - weighted by product probabilities')
fig = sns.barplot(x=['loc-no-evidence','loc-evidence-checking-cm','loc-evidence-checking-loc',
               'loc-evidence-checking-mmb'],
           y=[ltv_loc_no_evidence_expect,ltv_loc_evidence_checking_cm_expect, ltv_loc_evidence_checking_loc_expect,
              ltv_loc_evidence_checking_mmb_expect])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick)

# Get average GP and percent of customer across simulation for checking
avg_cust_percent_cmma_no_evidence, avg_cust_percent_cmma_evidence_checking_cm,avg_cust_percent_cmma_evidence_checking_mmb, avg_cust_percent_cmma_evidence_checking_loc = get_average_per_time_period(cumulative_cmma_customers_percent_no_evidence_final,
                            cumulative_cmma_customers_percent_evidence_checking_cm_final,
                           cumulative_cmma_customers_percent_evidence_checking_mmb_final,
                           cumulative_cmma_customers_percent_evidence_checking_loc_final)

avg_weekly_rev_cmma_no_evidence, avg_weekly_rev_cmma_evidence_checking_cm,avg_weekly_rev_cmma_evidence_checking_mmb, avg_weekly_rev_cmma_evidence_checking_loc = get_average_per_time_period(cumulative_cmma_rev_per_customer_no_evidence_final,
                            cumulative_cmma_rev_per_customer_evidence_checking_cm_final,
                           cumulative_cmma_rev_per_customer_evidence_checking_mmb_final,
                           cumulative_cmma_rev_per_customer_evidence_checking_loc_final)




ltv_cmma_no_evidence, ltv_cmma_evidence_checking_cm,ltv_cmma_evidence_checking_mmb, ltv_cmma_evidence_checking_loc  = ltv(avg_cust_percent_cmma_no_evidence,avg_weekly_rev_cmma_no_evidence,
   avg_cust_percent_cmma_evidence_checking_cm, avg_weekly_rev_cmma_evidence_checking_cm,
   avg_cust_percent_cmma_evidence_checking_mmb, avg_weekly_rev_cmma_evidence_checking_mmb,
   avg_cust_percent_cmma_evidence_checking_loc, avg_weekly_rev_cmma_evidence_checking_loc)


plt.figure(figsize=(12,8))
plt.title('LTV cmma ')
fig = sns.barplot(x=['cmma-no-evidence','cmma-evidence-checking-cm','cmma-evidence-checking-loc','cmma-evidence-checking-mmb'],
           y=[ltv_cmma_no_evidence,ltv_cmma_evidence_checking_cm, ltv_cmma_evidence_checking_loc,
              ltv_cmma_evidence_checking_mmb])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick) 

ltv_cmma_no_evidence_expect, ltv_cmma_evidence_checking_cm_expect,ltv_cmma_evidence_checking_mmb_expect, ltv_cmma_evidence_checking_loc_expect  = expected_ltv(avg_cust_percent_cmma_no_evidence,avg_weekly_rev_cmma_no_evidence,
   avg_cust_percent_cmma_evidence_checking_cm, avg_weekly_rev_cmma_evidence_checking_cm,
   avg_cust_percent_cmma_evidence_checking_mmb, avg_weekly_rev_cmma_evidence_checking_mmb,
   avg_cust_percent_cmma_evidence_checking_loc, avg_weekly_rev_cmma_evidence_checking_loc)


plt.figure(figsize=(12,8))
plt.title('LTV cmma - weighted by product probabilities')
fig = sns.barplot(x=['cmma-no-evidence','cmma-evidence-checking-cm','cmma-evidence-checking-loc','cmma-evidence-checking-mmb'],
           y=[ltv_cmma_no_evidence_expect,ltv_cmma_evidence_checking_cm_expect, ltv_cmma_evidence_checking_loc_expect,
              ltv_cmma_evidence_checking_mmb_expect])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick)

ltv_cmma_no_evidence_expect, ltv_cmma_evidence_checking_cm_expect,ltv_cmma_evidence_checking_mmb_expect, ltv_cmma_evidence_checking_loc_expect  = expected_ltv(avg_cust_percent_cmma_no_evidence,avg_weekly_rev_cmma_no_evidence,
   avg_cust_percent_cmma_evidence_checking_cm, avg_weekly_rev_cmma_evidence_checking_cm,
   avg_cust_percent_cmma_evidence_checking_mmb, avg_weekly_rev_cmma_evidence_checking_mmb,
   avg_cust_percent_cmma_evidence_checking_loc, avg_weekly_rev_cmma_evidence_checking_loc)


plt.figure(figsize=(14,8))
plt.title('Expected LTV product 3 ')
fig = sns.barplot(x=['product 3-no-evidence','product 3-evidence-product 4-product 7',
               'product 3-evidence-product 4-product 1','product 3-evidence-product 4-product 5'],
           y=[ltv_cmma_no_evidence_expect,ltv_cmma_evidence_checking_cm_expect, ltv_cmma_evidence_checking_loc_expect,
              ltv_cmma_evidence_checking_mmb_expect])
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
fig.yaxis.set_major_formatter(tick)

# Get average GP and percent of customer across simulation for checking
avg_cust_percent_cmma_no_evidence, avg_cust_percent_cmma_evidence_checking_cm,avg_cust_percent_cmma_evidence_checking_mmb, avg_cust_percent_cmma_evidence_checking_loc = get_average_per_time_period(cumulative_cmma_customers_percent_no_evidence_final,
                            cumulative_cmma_customers_percent_evidence_checking_cm_final,
                           cumulative_cmma_customers_percent_evidence_checking_mmb_final,
                           cumulative_cmma_customers_percent_evidence_checking_loc_final)

avg_weekly_rev_cmma_no_evidence, avg_weekly_rev_cmma_evidence_checking_cm,avg_weekly_rev_cmma_evidence_checking_mmb, avg_weekly_rev_cmma_evidence_checking_loc = get_average_per_time_period(cumulative_cmma_rev_per_customer_no_evidence_final,
                            cumulative_cmma_rev_per_customer_evidence_checking_cm_final,
                           cumulative_cmma_rev_per_customer_evidence_checking_mmb_final,
                           cumulative_cmma_rev_per_customer_evidence_checking_loc_final)




ltv_cmma_no_evidence, ltv_cmma_evidence_checking_cm,ltv_cmma_evidence_checking_mmb, ltv_cmma_evidence_checking_loc  = ltv(avg_cust_percent_cmma_no_evidence,avg_weekly_rev_cmma_no_evidence,
   avg_cust_percent_cmma_evidence_checking_cm, avg_weekly_rev_cmma_evidence_checking_cm,
   avg_cust_percent_cmma_evidence_checking_mmb, avg_weekly_rev_cmma_evidence_checking_mmb,
   avg_cust_percent_cmma_evidence_checking_loc, avg_weekly_rev_cmma_evidence_checking_loc)


plt.figure(figsize=(15,8))
plt.title('LTV product 3 ')
sns.barplot(x=['product 3-no-evidence','product 3-evidence-product 4-product 7',
               'product 3-evidence-product 4-product 1','product 3-evidence-product 4-product 5'],
           y=[ltv_cmma_no_evidence,ltv_cmma_evidence_checking_cm, ltv_cmma_evidence_checking_loc,
              ltv_cmma_evidence_checking_mmb])

ltv_table_dict = {
    'ltv_cmma_AvgRev':[ltv_cmma_no_evidence,ltv_cmma_evidence_checking_cm, ltv_cmma_evidence_checking_loc,
              ltv_cmma_evidence_checking_mmb],
    'ltv_mmb_AvgRev':[ltv_mmb_no_evidence,ltv_mmb_evidence_checking_cm, ltv_mmb_evidence_checking_loc,
              ltv_mmb_evidence_checking_mmb],
    'ltv_loc_AvgRev':[np.mean(ltv_loc_no_evidence),np.mean(ltv_loc_evidence_checking_cm),np.mean(ltv_loc_evidence_checking_loc),
              np.mean(ltv_loc_evidence_checking_mmb)],
    'ltv_checking_AvgRev':[ltv_checking_no_evidence,ltv_checking_evidence_checking_cm, ltv_checking_evidence_checking_loc,
              ltv_checking_evidence_checking_mmb],
    'ltv_fx_AvgRev':[ltv_fx_no_evidence,ltv_fx_evidence_checking_cm, ltv_fx_evidence_checking_loc,
              ltv_fx_evidence_checking_mmb],
    'ltv_cm_AvgRev':[ltv_cm_no_evidence,ltv_cm_evidence_checking_cm, ltv_cm_evidence_checking_loc,
              ltv_cm_evidence_checking_mmb],
    'ltv_es_AvgRev':[ltv_es_no_evidence,ltv_es_evidence_checking_cm, ltv_es_evidence_checking_loc,
              ltv_es_evidence_checking_mmb]
    
    
    
    
    
}

ltv_table_expect_percent_df = pd.DataFrame.from_dict(ltv_table_dict)

ltv_table_expect_percent_df.describe()

ltv_table_expect_percent_df.describe().iloc[1,:]

ltv_table_dict_prodprob = {
    'ltv_cmma_ProdProb':[ltv_cmma_no_evidence_expect,ltv_cmma_evidence_checking_cm_expect, ltv_cmma_evidence_checking_loc_expect,
              ltv_cmma_evidence_checking_mmb_expect],
    'ltv_mmb_ProdProb':[ltv_mmb_no_evidence_expect,ltv_mmb_evidence_checking_cm_expect, ltv_mmb_evidence_checking_loc_expect,
              ltv_mmb_evidence_checking_mmb_expect],
    'ltv_loc_ProdProb':[np.mean(ltv_loc_no_evidence_expect),np.mean(ltv_loc_evidence_checking_cm_expect),np.mean(ltv_loc_evidence_checking_loc_expect),
              np.mean(ltv_loc_evidence_checking_mmb_expect)],
    'ltv_checking_ProdProb':[ltv_checking_no_evidence_expect,ltv_checking_evidence_checking_cm_expect, ltv_checking_evidence_checking_loc_expect,
              ltv_checking_evidence_checking_mmb_expect],
    'ltv_fx_ProdProb':[ltv_fx_no_evidence_expect,ltv_fx_evidence_checking_cm_expect, ltv_fx_evidence_checking_loc_expect,
              ltv_fx_evidence_checking_mmb_expect],
    'ltv_cm_ProdProb':[ltv_cm_no_evidence_expect,ltv_cm_evidence_checking_cm_expect, ltv_cm_evidence_checking_loc_expect,
              ltv_cm_evidence_checking_mmb_expect],
    'ltv_es_ProdProb':[ltv_es_no_evidence_expect,ltv_es_evidence_checking_cm_expect, ltv_es_evidence_checking_loc_expect,
              ltv_es_evidence_checking_mmb_expect]
    
    

    
    
}

prob_prob_weight_ltv = pd.DataFrame.from_dict(ltv_table_dict_prodprob)

prob_prob_weight_ltv .describe()























































































































































































































































































































































































































































































































































































































































































































































































































































































