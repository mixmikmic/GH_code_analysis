import time
import itertools
import time, os, fnmatch, shutil
import pickle
import re
# Python 2 & 3 Compatibility
from __future__ import print_function, division

# Necessary imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
import seaborn as sns
from seaborn import plt
import matplotlib

matplotlib.rcParams.update({'font.size': 22})

get_ipython().magic('matplotlib inline')

from __future__ import print_function, division
import requests

folder = '/Users/torrie/Documents/Metis_Project_3_(McNulty)/'
pkl_filename_master = folder + "IP_OP_Hosp_df.pkl"

with open(pkl_filename_master, 'rb') as picklefile: 
    df_alldata = pickle.load(picklefile)

df_alldata.dtypes

#df_IPandProvider['Average_Total_Payments'].iloc[1] - df_IPandProvider['Average_Medicare_Payments'].iloc[1]
df_alldata['Patient_payment'] = df_alldata['Average_Total_Payments'] - df_alldata['Average_Medicare_Payments']

stats = df_alldata[df_alldata.DRG_Definition == '885 - PSYCHOSES'].mean()

stats

medians = df_alldata[df_alldata.DRG_Definition == '885 - PSYCHOSES'].median()
medians

df_Psych = df_alldata[df_alldata.DRG_Definition == '885 - PSYCHOSES']
df_Psych['Average_Covered_Charges'].hist(bins = 200)
df_Psych['Average_Covered_Charges'].mean()

df_Psych['log_Average_Covered_Charges'] = np.log(df_Psych['Average_Covered_Charges'])
df_Psych['log_Average_Covered_Charges'].hist(bins = 200)

df_vasc = df_alldata[df_alldata.DRG_Definition == '252 - OTHER VASCULAR PROCEDURES W MCC']
df_vasc['Average_Covered_Charges'].hist(bins = 200)
df_vasc['Average_Covered_Charges'].mean()

df_vasc['log_Average_Covered_Charges'] = np.log(df_vasc['Average_Covered_Charges'])
df_vasc['log_Average_Covered_Charges'].hist(bins = 200)

df_debride = df_alldata[df_alldata.DRG_Definition == '0012 - Level I Debridement & Destruction']
df_debride['Average_Covered_Charges'].hist(bins = 200)
df_debride['Average_Covered_Charges'].mean()

df_debride['log_Average_Covered_Charges'] = np.log(df_debride['Average_Covered_Charges'])
df_debride['log_Average_Covered_Charges'].hist(bins = 200)

df_alldata.columns

df_alldata['Average_Medicare_Payments'] = df_alldata['Average_Medicare_Payments'].replace('0',int(1))

df_alldata['Patient_payment'] = df_alldata['Patient_payment'].replace('0',int(1))

df_alldata['log_Average_Covered_Charges'] = np.log(df_alldata['Average_Covered_Charges'])

df_alldata['log_Average_Medicare_Payments'] = np.log(df_alldata['Average_Medicare_Payments'])

df_alldata['log_Average_Total_Payments'] = np.log(df_alldata['Average_Total_Payments'])

df_alldata['log_Patient_payment'] = np.log(df_alldata['Patient_payment'])

df_alldata['Average_Covered_Charges'].groupby([df_alldata['DRG_Definition'],df_alldata['Provider_State']]).mean()

stats_calc = ['Average_Covered_Charges','Average_Medicare_Payments','Average_Total_Payments',              'Patient_payment','log_Average_Covered_Charges','log_Average_Medicare_Payments',              'log_Average_Total_Payments', 'log_Patient_payment']
    
for column in stats_calc:
    df_alldata = df_alldata.join(df_alldata.groupby('DRG_Definition')[column].mean(), on='DRG_Definition', rsuffix='_mean')
    df_alldata = df_alldata.join(df_alldata.groupby('DRG_Definition')[column].median(), on='DRG_Definition', rsuffix='_median')
    df_alldata =df_alldata.join(df_alldata.groupby('DRG_Definition')[column].std(), on='DRG_Definition', rsuffix='_std')

for column in stats_calc:
    df_alldata = df_alldata.join(df_alldata.groupby(['DRG_Definition','Provider_State'])[column].mean(), on=['DRG_Definition','Provider_State'], rsuffix='_ST_mean')
    df_alldata = df_alldata.join(df_alldata.groupby(['DRG_Definition','Provider_State'])[column].mean(), on=['DRG_Definition','Provider_State'], rsuffix='_ST_median')
    df_alldata = df_alldata.join(df_alldata.groupby(['DRG_Definition','Provider_State'])[column].mean(), on=['DRG_Definition','Provider_State'], rsuffix='_ST_std')

df_alldata.columns

df_alldata['Average_Covered_Charges_mean_3_bins'] = '1'
df_alldata['Average_Covered_Charges_mean_3_bins'][df_alldata['Average_Covered_Charges'] >= (df_alldata['Average_Covered_Charges_mean']+0.5*df_alldata['Average_Covered_Charges_std'])] = '2'
df_alldata['Average_Covered_Charges_mean_3_bins'][df_alldata['Average_Covered_Charges'] <= (df_alldata['Average_Covered_Charges_mean']-0.5*df_alldata['Average_Covered_Charges_std'])] = '0'


df_alldata['Average_Covered_Charges_mean_3_bins'].value_counts()

df_alldata.columns

charge_list = ['Average_Covered_Charges', 'Average_Medicare_Payments',               'Average_Total_Payments','Patient_payment']
  
#df_alldata['Ones'] = int(1)    

for charge_type in charge_list:

    charge_type_mean = charge_type + "_mean"
    charge_type_median = charge_type + "_median"
    charge_type_std = charge_type + '_std'

    charge_type_ST_mean = charge_type + "_ST_mean"
    charge_type_ST_median = charge_type + "_ST_median"
    charge_type_ST_std = charge_type + '_ST_std'
    
    column_name = charge_type_mean +'_3_bins'
    #df_alldata[column_name] = df_alldata['Ones'].where(df_alldata[charge_type] >= df_alldata[charge_type_mean],0)
    df_alldata[column_name] = '1'
    df_alldata[column_name][df_alldata[charge_type] >= (df_alldata[charge_type_mean]+0.4*df_alldata[charge_type_std])] = '2'
    df_alldata[column_name][df_alldata[charge_type] <= (df_alldata[charge_type_mean]-0.4*df_alldata[charge_type_std])] = '0'

    
    column_name = charge_type_mean +'_ST_3_bins'
    #df_alldata[column_name] = df_alldata['Ones'].where(df_alldata[charge_type] >= df_alldata[charge_type_ST_mean],0)
    df_alldata[column_name] = '1'
    df_alldata[column_name][df_alldata[charge_type] >= (df_alldata[charge_type_ST_mean]+0.15*df_alldata[charge_type_ST_std])] = '2'
    df_alldata[column_name][df_alldata[charge_type] <= (df_alldata[charge_type_ST_mean]-0.15*df_alldata[charge_type_ST_std])] = '0'

    column_name = charge_type_median +'_3_bins'
    #df_alldata[column_name] = df_alldata['Ones'].where(df_alldata[charge_type] >= df_alldata[charge_type_median],0)
    df_alldata[column_name] = '1'
    df_alldata[column_name][df_alldata[charge_type] >= (df_alldata[charge_type_median]+0.4*df_alldata[charge_type_std])] = '2'
    df_alldata[column_name][df_alldata[charge_type] <= (df_alldata[charge_type_median]-0.4*df_alldata[charge_type_std])] = '0'

    column_name = charge_type_median +'_ST_3_bins'
    #df_alldata[column_name] = df_alldata['Ones'].where(df_alldata[charge_type] >= df_alldata[charge_type_ST_median],0)
    df_alldata[column_name] = '1'
    df_alldata[column_name][df_alldata[charge_type] >= (df_alldata[charge_type_ST_median]+0.15*df_alldata[charge_type_ST_std])] = '2'
    df_alldata[column_name][df_alldata[charge_type] <= (df_alldata[charge_type_ST_median]-0.15*df_alldata[charge_type_ST_std])] = '0'

    


        

charge_list = ['log_Average_Covered_Charges', 'log_Average_Medicare_Payments',               'log_Average_Total_Payments','log_Patient_payment']
  
#df_alldata['Ones'] = int(1)    

for charge_type in charge_list:

    charge_type_mean = charge_type + "_mean"
    charge_type_median = charge_type + "_median"
    charge_type_std = charge_type + '_std'

    charge_type_ST_mean = charge_type + "_ST_mean"
    charge_type_ST_median = charge_type + "_ST_median"
    charge_type_ST_std = charge_type + '_ST_std'
    
    column_name = charge_type_mean +'_3_bins'
    #df_alldata[column_name] = df_alldata['Ones'].where(df_alldata[charge_type] >= df_alldata[charge_type_mean],0)
    df_alldata[column_name] = '1'
    df_alldata[column_name][df_alldata[charge_type] >= (df_alldata[charge_type_mean]+0.5*df_alldata[charge_type_std])] = '2'
    df_alldata[column_name][df_alldata[charge_type] <= (df_alldata[charge_type_mean]-0.5*df_alldata[charge_type_std])] = '0'

    
    column_name = charge_type_mean +'_ST_3_bins'
    #df_alldata[column_name] = df_alldata['Ones'].where(df_alldata[charge_type] >= df_alldata[charge_type_ST_mean],0)
    df_alldata[column_name] = '1'
    df_alldata[column_name][df_alldata[charge_type] >= (df_alldata[charge_type_ST_mean]+0.5*df_alldata[charge_type_ST_std])] = '2'
    df_alldata[column_name][df_alldata[charge_type] <= (df_alldata[charge_type_ST_mean]-0.5*df_alldata[charge_type_ST_std])] = '0'

    column_name = charge_type_median +'_3_bins'
    #df_alldata[column_name] = df_alldata['Ones'].where(df_alldata[charge_type] >= df_alldata[charge_type_median],0)
    df_alldata[column_name] = '1'
    df_alldata[column_name][df_alldata[charge_type] >= (df_alldata[charge_type_median]+0.5*df_alldata[charge_type_std])] = '2'
    df_alldata[column_name][df_alldata[charge_type] <= (df_alldata[charge_type_median]-0.5*df_alldata[charge_type_std])] = '0'

    column_name = charge_type_median +'_ST_3_bins'
    #df_alldata[column_name] = df_alldata['Ones'].where(df_alldata[charge_type] >= df_alldata[charge_type_ST_median],0)
    df_alldata[column_name] = '1'
    df_alldata[column_name][df_alldata[charge_type] >= (df_alldata[charge_type_ST_median]+0.5*df_alldata[charge_type_ST_std])] = '2'
    df_alldata[column_name][df_alldata[charge_type] <= (df_alldata[charge_type_ST_median]-0.5*df_alldata[charge_type_ST_std])] = '0'

    


        

df_alldata.columns

df_alldata['log_Patient_payment_mean_3_bins'].value_counts()

df_alldata['log_Average_Total_Payments_median_3_bins'].value_counts()

df_alldata['log_Patient_payment_median_3_bins'].value_counts()

df_alldata['log_Average_Covered_Charges_median_ST_3_bins'].value_counts()

df_alldata['Average_Covered_Charges_median_ST_3_bins'].value_counts()



df_alldata.head()

df_alldata.shape

df_alldata['log_Average_Medicare_Payments_median_3_bins'].value_counts()

df_alldata = df_alldata.join(df_alldata.groupby('DRG_Definition')['DRG_Definition'].count(), on='DRG_Definition', rsuffix='_count')
df_alldata.head()

folder = '/Users/torrie/Documents/Metis_Project_3_(McNulty)/'

with open(folder +'IP_OP_Hosp_binned_3_df.pkl', 'wb') as picklefile:
    pickle.dump(df_alldata, picklefile)

df_alldata.Average_Covered_Charges_median_ST_3_bins.value_counts()







