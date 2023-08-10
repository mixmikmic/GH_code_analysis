get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set(style="whitegrid", palette="pastel", color_codes=True)

results_df = pd.read_csv('MU check data RC 12-4-2017.csv')

results_df[['Site', 'By', 'Using', 'MU_check_mins']].head()

results_df.shape[0]  # 77 patients

sites = results_df['Site'].value_counts()
print(len(sites))
sites

mult_sites = sites[sites > 1].index.values   # get the sites with more than 1 entry
print(len(mult_sites))
mult_sites

mult_sites_df = results_df[results_df['Site'].isin(mult_sites)]       # for plots of sites with more than 1 entry
notin_mult_sites_df = results_df[~results_df['Site'].isin(mult_sites)]

plt.figure(figsize=(8, 8))  
sns.countplot(y="Site", hue="Using", data=mult_sites_df, palette="BuPu");  # , palette="Greens_d", , order=reversed(sites.index.values)
plt.ylabel(' ');
plt.xlabel('Counts');

print(len(results_df['Using']))
results_df['Using'].value_counts()

results_df['By'].value_counts()

#plt.figure(figsize=(3, 3)) 
results_df['By'].value_counts().plot.pie(figsize=(5, 5), title = 'Proportion of entries by physicist');
plt.ylabel(' ');

results_df['MU_check_mins'].plot.hist(figsize=(5, 5), bins=20);   # all results
#mult_sites_df['MU_check_mins'].plot.hist(figsize=(5, 5), bins=20);   # all results
#notin_mult_sites_df['MU_check_mins'].plot.hist(figsize=(5, 5), bins=20);   # all results
plt.title('Histogram of times for all sites');
plt.xlim([0, 50])
plt.xlabel('Time (minutes)');
plt.ylabel('Counts');

plt.figure(figsize=(10, 8))  
sns.boxplot(x="MU_check_mins", y="Site", data=mult_sites_df, hue="Using",palette="BuPu");  #  hue="By",
plt.xlabel('Time (minutes)');
plt.ylabel(' ');

plt.figure(figsize=(10, 8)) 
results_pivot = results_df.groupby(['Site', 'By'],as_index=False).mean().pivot(index='Site', columns='By', values='MU_check_mins')
sns.heatmap(results_pivot, annot=True, cmap='YlOrRd');
plt.title('Mean time (minutes) to perform the test');
plt.ylabel(' ');
plt.xlabel('Physicist');

res = results_pivot.mean(axis=1)
res.sort()
res.plot.barh(figsize=(5, 5)); # , labeldistance=2
plt.xlabel('Mean time (minutes)');

from scipy.stats import norm
sns.distplot(results_df['MU_check_mins'], color="m", bins=20,  fit=norm)  # ,  kde=False
plt.title('Histogram of times')
plt.xlim([0,60])
plt.xlabel('Time (minutes)')



