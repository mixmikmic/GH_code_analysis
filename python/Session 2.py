import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('c:/Users/arpie71/Dropbox/Columbia/workshop')
print(os.listdir('data/.'))
df=pd.read_pickle('data/preselect')
df.head(n=5)

df.head(n=5)

DEV = df['Dem_EV'].sum()
print(DEV)
df['Dem_EV'].sum()

print("Clinton received", df['Dem_EV'].sum(), "electoral votes and Trump received", df['Rep_EV'].sum())
df['DEVshare']=df['Dem_EV']/DEV

print(df['Median_Income'].mean())
df['DEVshare'].mean()

print(df['Median_Income'][df['Dem_EV']!=0].mean())
print(df['DEVshare'][df['Dem_EV']!=0].mean())
print(df['DEVshare'][30].mean())
print(df[['DEVshare','Median_Income']][df['Dem_EV']!=0].mean())
print(df[['DEVshare','Median_Income']][df['State']=='California'].mean())

print(df['Median_Income'][df['Dem_EV']==0].mean())
print(df['Median_Income'][df['Rep_EV']!=0].mean())

print(round(df['Median_Income'][df['Dem_EV']==0].mean(),2))

print(round(df['Median_Income'][df['Rep_EV']!=0].mean(),2))


print(df['Unalloc_EV'].sum())
print(df['Median_Income'][df['Unalloc_EV']!=0].mean())
print(df['State'][df['Unalloc_EV']!=0])
print(df[['State','Unalloc_EV','Dem_EV','Rep_EV']][df['Unalloc_EV']!=0])

print(df['Median_Income'][df['Dem_EV']!=0].median())
print(df['Median_Income'][df['Dem_EV']!=0].std())
print(df['Median_Income'][df['Dem_EV']!=0].max())
print(df['Median_Income'][df['Dem_EV']!=0].min())
print(df['Median_Income'][df['Dem_EV']!=0].skew())
print(df['Median_Income'][df['Dem_EV']!=0].quantile([.25,.75]))

df.describe()
df[['Dem_EV','Rep_EV','Unalloc_EV']].describe()
df[['Dem_EV','Rep_EV','Unalloc_EV']][df['Dem_EV']!=0].describe()

print(df[['State','Dem_EV']][df['Rep_EV']!=0])





