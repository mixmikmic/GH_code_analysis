import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
from scipy import stats

get_ipython().magic('matplotlib inline')

dframe = pd.read_csv('data/sa2_cor_features.csv') #Reading the new csv file
dframe.columns = dframe.columns.str.strip()
dframe.drop('Unnamed: 0',axis=1,inplace=True)
dframe.set_index('area_name',inplace=True) #Setting the index to SA2 name

dframe

dframe.corr(method='pearson') #Using Pearson's coefficient of correlation

r_r2 = DataFrame(dframe.corr(method='pearson')['Walkability Index'])

r_r2.columns = ['Correlation'] 
r_r2['Coefficient of Determination'] = r_r2['Correlation'] ** 2

r_r2

#For Health Indices
for feature in dframe.columns[1:3]:
    scatter_plot = sns.lmplot('Walkability Index',feature,data=dframe) #Creating inline figures
    scatter_plot.savefig("figures/"+feature+'_walk_corr.png') #Saving figures
    

#For Average Life Satisfaction
for feature in dframe.columns[3:4]:
    scatter_plot = sns.lmplot('Walkability Index',feature,data=dframe) #Creating inline figures
    scatter_plot.savefig("figures/"+feature+'_walk_corr.png') #Saving figures
    

#For Median Age
for feature in dframe.columns[6:7]:
    scatter_plot = sns.lmplot('Walkability Index',feature,data=dframe) #Creating inline figures
    scatter_plot.savefig("figures/"+feature+'_walk_corr.png') #Saving figures
    

#For % private dwellings without a vehicle 
for feature in dframe.columns[4:5]:
    scatter_plot = sns.lmplot('Walkability Index',feature,data=dframe) #Creating inline figures
    scatter_plot.savefig("figures/"+feature+'_walk_corr.png') #Saving figures

#For Median Rent
for feature in dframe.columns[8:9]:
    scatter_plot = sns.lmplot('Walkability Index',feature,data=dframe) #Creating inline figures
    scatter_plot.savefig("figures/"+feature+'_walk_corr.png') #Saving figures

#For Income Variable and 
for feature in dframe.columns[5:6]:
    scatter_plot = sns.lmplot('Walkability Index',feature,data=dframe) #Creating inline figures
    scatter_plot.savefig("figures/"+feature+'_walk_corr.png') #Saving figures

#For Median Household Income
for feature in dframe.columns[7:8]:
    scatter_plot = sns.lmplot('Walkability Index',feature,data=dframe) #Creating inline figures
    scatter_plot.savefig("figures/"+feature+'_walk_corr.png') #Saving figures

#For public transport stop and station counts
for feature in dframe.columns[9:]:
    scatter_plot = sns.lmplot('Walkability Index',feature,data=dframe) #Creating inline figures
    scatter_plot.savefig("figures/"+feature+'_walk_corr.png') #Saving figures

