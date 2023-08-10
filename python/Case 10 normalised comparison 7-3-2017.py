import os
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyEclipseDVH_v2 import List_txt, Load_patient, get_dmin, get_dmax, get_d_metric, Load_files_to_df

from scipy import interpolate
from scipy import stats
from scipy.stats import wilcoxon  # must import explicitly
import seaborn as sns
import re

files = List_txt()
files

df = Load_files_to_df(files)

df.head()

structures_ls = df.columns.levels[2]
structures_ls

PTV1 = structures_ls[15]
PTV1    # string key for this structure

height=6
width=12           # wwidth of figs

df.xs(PTV1, level='Structure', axis=1).plot(figsize=(width, height))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.legend()
#plt.ylim([0,105])
plt.axhline(y=50, c='k', ls = '--')
plt.xlim([62,68])

df.plot()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

df = pd.read_csv('DVH_metrics.csv')

df.head()

metrics = df['metric'].unique()
print(len(metrics))
metrics

df['structure'].unique()

format = lambda x: re.sub('_P9', '', x)   # more reliable than strip
format2 = lambda x: re.sub('_1', '', x)   # more reliable than strip

df['structure'] = df['structure'].map(format)
df['patID'] = df['patID'].map(format2)

df['structure'].unique()

df[df['patID'] == 'AAA'].head()

df2 = pd.merge(df[df['patID'] == 'AAA'], df[df['patID'] == 'AXB'], how='inner', on=['metric',  'structure'])  # Get merged set

df2['AXB-AAA'] = df2['observed_y'] - df2['observed_x'] 

df2.head()

diff_data_df2 = df2.pivot(index='structure', columns='metric', values='AXB-AAA')

sns.heatmap(diff_data_df2, annot=True, linewidths=.5, center=0,  cmap='PRGn') #  



metrics_to_plot = ['D5%', 'D50%', 'D95%']

# Set up the matplotlib figure
height=12
width=8  
palette="Set3"

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(width, height), sharex=True)

metric = metrics_to_plot[0]
sns.barplot(x = df2[df2['metric'] == metric]['structure'].values, y = df2[df2['metric'] == metric]['AXB-AAA'].values, palette = palette, ax=ax1)
ax1.set_ylabel(metric)
ax1.set_ylim([-0.5, 0.5])

metric = metrics_to_plot[1]
sns.barplot(x = df2[df2['metric'] == metric]['structure'].values, y = df2[df2['metric'] == metric]['AXB-AAA'].values, palette = palette, ax=ax2)
ax2.set_ylabel(metric)
ax2.set_ylim([-0.5, 0.5])

metric = metrics_to_plot[2]
sns.barplot(x = df2[df2['metric'] == metric]['structure'].values, y = df2[df2['metric'] == metric]['AXB-AAA'].values, palette = palette,ax=ax3)
ax3.set_ylabel(metric)
ax3.set_ylim([-0.5, 0.5])

