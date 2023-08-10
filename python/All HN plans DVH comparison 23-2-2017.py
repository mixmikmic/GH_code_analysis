import os
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyEclipseDVH import eclipse_DVH
from scipy import interpolate
from scipy import stats
from scipy.stats import wilcoxon  # must import explicitly

files = os.listdir()   # return a list of files
AAA_files = [file for file in files if file.endswith('AAA.txt')]
AXB_files = [file for file in files if file.endswith('Dm.txt')]

AXB_files

# d = {value: foo(value) for value in sequence if bar(value)}
AAA_dict = {file.strip('.txt'): eclipse_DVH(file) for file in AAA_files}    # load all AAA DVH into dict
AXB_dict = {file.strip('.txt'): eclipse_DVH(file) for file in AXB_files}    # load all AAA DVH into dict

case_list = AAA_dict.keys()
case_list = [entry.strip('_AAA') for entry in case_list]
print(len(case_list))
case_list

AAA_dict['Case1_AAA'].structures_names_list

structure = 'PTV1'         # The structure to plot
#structure = 'PTV2'
xlimits = [50,75]

for i, key in enumerate(case_list):
    temp_AAA_DVH_df = pd.DataFrame({key : AAA_dict.get(key + '_AAA').DVH_df[structure]})       # place in a df
    temp_AXB_DVH_df = pd.DataFrame({key : AXB_dict.get(key + '_Dm').DVH_df[structure] })       # place in a df

    if i == 0:
        AAA_df = temp_AAA_DVH_df  # create the dataframe
        AXB_df  = temp_AXB_DVH_df  # create the dataframe
    else:       
        AAA_df  = pd.concat([AAA_df, temp_AAA_DVH_df], axis=1)   # if df exists, populate    
        AXB_df  = pd.concat([AXB_df, temp_AXB_DVH_df], axis=1)   # if df exists, populate
    
   # AAA_df = AAA_df.fillna(value=0)
   # AXB_df = AXB_df.fillna(value=0)
    
    AAA_df = AAA_df.fillna(method='pad')  # pad the data to fill in NaN, this is OK as difference in index is 0.01%
    AXB_df = AXB_df.fillna(method='pad')   

AAA_df.head()

height=4
width=8           # wwidth of figs
plt.figure(figsize=(width, height))
    
plt.fill_between(AAA_df.mean(axis = 1).index.values, (AAA_df.mean(axis = 1) - AAA_df.std(axis = 1)).values, (AAA_df.mean(axis = 1) + AAA_df.std(axis = 1)).values, color = 'm', alpha=0.55, interpolate=True)
plt.plot(AAA_df.mean(axis = 1), color = 'm', label = 'AAA Mean', ls='--') # ls = '--', 

plt.fill_between(AXB_df.mean(axis = 1).index.values, (AXB_df.mean(axis = 1) - AXB_df.std(axis = 1)).values, (AXB_df.mean(axis = 1) + AXB_df.std(axis = 1)).values, color = 'g', alpha=0.35, interpolate=True)
plt.plot(AXB_df.mean(axis = 1), color = 'g', label = 'AXB Mean', ls='--') # ls = '--'

#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend()
plt.ylim([0,105])
plt.xlim(xlimits)
plt.title(structure + ' comparison of mean and standard deviation')
plt.xlabel('Dose (Gy)')
plt.ylabel('Ratio of total structure volume (%)')

width=7
height=4
plt.figure(figsize=(width, height))

case = 'Case10'
structure = 'PTV1'
plt.plot(AAA_dict[case + '_AAA'].DVH_df[structure], label=structure+" AAA", color='m', ls='--')
plt.plot(AXB_dict[case +'_Dm'].DVH_df[structure], label=structure+" AXB",  color='g', ls='--' )

#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend()
plt.title(case + ' comparison of AAA and AXB DVH')
plt.xlabel('Dose (Gy)')
plt.ylabel('Ratio of total structure volume (%)')
plt.ylim([0,105])
plt.xlim(xlimits)



