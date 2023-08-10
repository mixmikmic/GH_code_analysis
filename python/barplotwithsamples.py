get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd

# PARAMETERS - EDIT ME
groups = [9, 10]  # set the number of items/values for each group
groups_labels = ['MCSminus', 'MCSplus']  # set names for each group
rex_data_filepath = 'seed_mask_mcsminus_mcsplus.ROIs.rex.data.txt'

# Loading data from Rex csv
dfraw = pd.read_csv(rex_data_filepath, index_col=False, header=None, squeeze=True)
dfraw

# Extract the values for each group in a separate dataframe
df_g = []
start = 0
for g in groups:
    df_g.append(dfraw[start:start+g])
    start = g
df_g

# Helper functions
import numpy as np
import scipy.stats

def comp_ci(a):
    '''Calculates the 90% confidence interval from a vector.
    From the excellent SO answer by Ulrich Stern: https://stackoverflow.com/a/34474255/1121352'''
    return scipy.stats.t.interval(0.90, len(a)-1, loc=np.mean(a), scale=scipy.stats.sem(a))

# Plot!

# Plotting parameters
ylim = [-0.2, 0.3]  # limit y axis to these values. Set to None to use default limits automatically detected by matplotlib.
width = 1  # width of the bars
colors = ['b', 'g', 'r', 'y', 'c', 'b']
ylabel = 'Effect sizes'
ticks = np.arange(1, 1+(width*len(groups)), width)  # do not modify this

# Plotting each bar
fig, ax = plt.subplots()
for i, dg in enumerate(df_g):
    ax.bar(ticks[i], dg.mean(), width=width, yerr=(dg.mean() - comp_ci(dg)[1]), alpha=0.5, color=colors[i], error_kw={'ecolor': 'k', 'elinewidth': 1, 'capsize': 15, 'capthick': 1, 'barsabove': False})
    ax.scatter([ticks[i]+(float(width)/2)] * len(dg), dg, color=colors[i], marker='x', s=30)
# Change the ticks to set the group name (and place the labels nicely)
ax.set_xticks([t + float(width)/2 for t in ticks])  # place in the middle of each bar (position tick t + half of bar width)
ax.set_xticklabels(('MCSminus', 'MCSplus'))
# Force draw the plot
plt.tight_layout()
if ylim:
    ax.set_ylim(ylim)
ax.set_xlim([ticks[0], ticks[-1]+width])
plt.ylabel(ylabel)
plt.show()



