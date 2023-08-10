__depends__ = ["/Volumes/Seagate Expansion Drive/seq1_jcole_gene-editing_gene-editing_multiple/multiple"]
__dest__ = []

get_ipython().magic('matplotlib inline')

import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.mpl_style', 'default')
import statsmodels.api as sm
import itertools
import numpy as np
import seaborn as sns

plt.rcdefaults()
# Typeface sizes
from matplotlib import rcParams
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
#rcParams['font.family'] = 'serif'
#rcParams['font.serif'] = ['Computer Modern Roman']
#rcParams['text.usetex'] = True

# Optimal figure size
WIDTH = 350.0  # the number latex spits out
FACTOR = 0.90  # the fraction of the width you'd like the figure to occupy
fig_width_pt  = WIDTH * FACTOR

inches_per_pt = 1.0 / 72.27
golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good

fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
fig_height_in = fig_width_in * golden_ratio   # figure height in inches
fig_dims      = [fig_width_in, fig_height_in] # fig dims as a list

rcParams['figure.figsize'] = fig_dims

methods = ['crispr', 'noedits', 'perfect', 'talen', 'zfn']
for method in methods:
    print method
    # We have 10 replicates for each simulation
    for sim in xrange(1,11):
        if sim == 1: print '\tReplicate: ', sim,
        elif sim < 10: print ', ', sim,
        else: print ', ', sim, ''
        # Load the individual history files
        lc = pd.read_csv('multiple/01_00/%s/%s/cows_history_%s_20.txt'%(method,sim,method), sep='\t')
        dc = pd.read_csv('multiple/01_00/%s/%s/dead_cows_history_%s_20.txt'%(method,sim,method), sep='\t')
        lb = pd.read_csv('multiple/01_00/%s/%s/bulls_history_%s_20.txt'%(method,sim,method), sep='\t')
        db = pd.read_csv('multiple/01_00/%s/%s/dead_bulls_history_%s_20.txt'%(method,sim,method), sep='\t')
        # Stack the individual animal datasets
        all_animals = pd.concat([lc, dc, lb, db], axis=0)
        all_animals['replicate'] = sim
        all_animals['method'] = method
    if method == methods[0]:
        grouped = pd.crosstab(all_animals['died'], all_animals['cause']).apply(lambda r: r/r.sum(), axis=1)
        grouped['method'] = method
    else:
        temp_grouped = pd.crosstab(all_animals['died'], all_animals['cause']).apply(lambda r: r/r.sum(), axis=1)
        temp_grouped['method'] = method
        grouped = pd.concat([grouped, temp_grouped])

grouped.head()

grouped.reset_index(inplace=True)

fig = plt.figure(figsize=(12, 9), dpi=300, facecolor='white')

# Set nicer limits
ymin ,ymax = 0, 0.10
xmin, xmax = 0, 31

colors = itertools.cycle(['r', 'g', 'b','k'])
markers = itertools.cycle(['o', 'v', 's'])

ax = fig.add_subplot(1, 1, 1)
ax.set_title('Embryonic losses (1% of bulls and 0% of cows edited)')
ax.set_xlabel('Birth year')
ax.set_ylabel('Proportion of embryonic losses')
ax.plot(grouped['died'][grouped['method']=='crispr'], grouped['R'][grouped['method']=='crispr'], label='crispr', linewidth=2, marker=markers.next(), c=colors.next())
ax.plot(grouped['died'][grouped['method']=='noedits'], grouped['R'][grouped['method']=='noedits'], label='noedits', linewidth=2, marker=markers.next(), c=colors.next())
ax.plot(grouped['died'][grouped['method']=='perfect'], grouped['R'][grouped['method']=='perfect'], label='perfect', linewidth=2, marker=markers.next(), c=colors.next())
ax.plot(grouped['died'][grouped['method']=='talen'], grouped['R'][grouped['method']=='talen'], label='talen', linewidth=2, marker=markers.next(), c=colors.next())
ax.plot(grouped['died'][grouped['method']=='zfn'], grouped['R'][grouped['method']=='zfn'], label='zfn', linewidth=2, marker=markers.next(), c=colors.next())
ax.legend(loc='best')

ax.set_ylim(ymin, ymax)

# Despine
ax = fig.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Plot and save
fig.tight_layout(pad=0.1)  # Make the figure use all available whitespace
fig.savefig('multiple/01_00/embryonic losses_01_00.png', dpi=300)
plt.show()

methods = ['crispr', 'noedits', 'perfect', 'talen', 'zfn']
for method in methods:
    print method
    # We have 10 replicates for each simulation
    for sim in xrange(1,11):
        if sim == 1: print '\tReplicate: ', sim,
        elif sim < 10: print ', ', sim,
        else: print ', ', sim, ''
        # Load the individual history files
        lc = pd.read_csv('multiple/01_00/%s/%s/cows_history_%s_20.txt'%(method,sim,method), sep='\t')
        dc = pd.read_csv('multiple/01_00/%s/%s/dead_cows_history_%s_20.txt'%(method,sim,method), sep='\t')
        lb = pd.read_csv('multiple/01_00/%s/%s/bulls_history_%s_20.txt'%(method,sim,method), sep='\t')
        db = pd.read_csv('multiple/01_00/%s/%s/dead_bulls_history_%s_20.txt'%(method,sim,method), sep='\t')
        # Stack the individual animal datasets
        all_animals = pd.concat([lc, dc, lb, db], axis=0)
        all_animals['replicate'] = sim
        all_animals['method'] = method
        # Select only the rows where the animal was either the sire or a dam of another animal in the dataset.
        all_animals = all_animals[(all_animals['animal'].isin(all_animals['sire'])) | (all_animals['animal'].isin(all_animals['dam']))]            
        all_animals['r_count'] = all_animals['recessives'].apply(np.count_nonzero, axis=1)
    if method == methods[0]:
        grouped = pd.crosstab(all_animals['born'], [all_animals['r_count'], all_animals['sex']])#.apply(lambda r: r/r.sum(), axis=1)
        grouped['method'] = method
    else:
        temp_grouped = pd.crosstab(all_animals['born'], [all_animals['r_count'], all_animals['sex']])#.apply(lambda r: r/r.sum(), axis=1)
        temp_grouped['method'] = method
        grouped = pd.concat([grouped, temp_grouped])

grouped.reset_index(drop=True, inplace=True)

grouped.columns = grouped.columns.get_level_values(0)
grouped.columns = ['born', 'carrier_dams', 'carrier_sires', 'method']
grouped.head()

grouped['at_risk'] = grouped['carrier_dams'] * grouped['carrier_sires']

grouped.head(10)

grouped['risk_pct'] = grouped['at_risk'] / grouped['at_risk'].sum(axis=0)

grouped.head(10)

fig = plt.figure(figsize=(12, 9), dpi=300, facecolor='white')

# Set nicer limits
#ymin ,ymax = 0, 0.10
xmin, xmax = 0, 31

colors = itertools.cycle(['r', 'y', 'g', 'b', 'indigo'])
markers = itertools.cycle(['o', 'v', 's'])
bwidth = 7.5

ax = fig.add_subplot(1, 1, 1)
ax.set_title('Number of at-risk matings (1% of bulls and 0% of cows edited)')
ax.set_xlabel('Birth year')
ax.set_ylabel('Number of matings')
r1 = ax.bar(grouped['born'][grouped['method']=='crispr'], grouped['at_risk'][grouped['method']=='crispr'], 
       width=bwidth, label='crispr', color=colors.next())
r2 = ax.bar(grouped['born'][grouped['method']=='noedits']+bwidth, grouped['at_risk'][grouped['method']=='noedits'],
       width=bwidth, label='noedits', color=colors.next())
r3 = ax.bar(grouped['born'][grouped['method']=='perfect']+(2*bwidth), grouped['at_risk'][grouped['method']=='perfect'], 
       width=bwidth, label='perfect', color=colors.next())
r4 = ax.bar(grouped['born'][grouped['method']=='talen']+(3*bwidth), grouped['at_risk'][grouped['method']=='talen'],
       width=bwidth, label='talen', color=colors.next())
r5 = ax.bar(grouped['born'][grouped['method']=='zfn']+(4*bwidth), grouped['at_risk'][grouped['method']=='zfn'],
       width=bwidth, label='zfn', color=colors.next())
ax.legend(loc='best')

#ax.set_ylim(ymin, ymax)

# Despine
ax = fig.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Plot and save
fig.tight_layout(pad=0.1)  # Make the figure use all available whitespace
#fig.savefig('./results/embryonic losses_01_00.png', dpi=300)
plt.show()



