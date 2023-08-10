#__depends__ = ["/Volumes/Seagate Expansion Drive/seq1_jcole_gene-editing_gene-editing_multiple/multiple"]
#__dest__ = ["./results"]

get_ipython().magic('matplotlib inline')

import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.mpl_style = 'default'
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

pop_edited = '10_01'
methods = ['crispr', 'noedits', 'perfect', 'talen', 'zfn']
for method in methods:
    print method
    # We have 10 replicates for each simulation
    for sim in xrange(1,11):
        if sim == 1: print '\tReplicate: ', sim,
        elif sim < 10: print ', ', sim,
        else: print ', ', sim, ''
        # Load the individual history files
        lc = pd.read_csv('multiple/10_01/%s/%s/cows_history_%s_20.txt'%(method,sim,method), sep='\t')
        dc = pd.read_csv('multiple/10_01/%s/%s/dead_cows_history_%s_20.txt'%(method,sim,method), sep='\t')
        lb = pd.read_csv('multiple/10_01/%s/%s/bulls_history_%s_20.txt'%(method,sim,method), sep='\t')
        db = pd.read_csv('multiple/10_01/%s/%s/dead_bulls_history_%s_20.txt'%(method,sim,method), sep='\t')
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

grouped.tail()

grouped = grouped.reset_index()

grouped.columns

grouped['R'].fillna(0.0, inplace=True)

fig = plt.figure(figsize=(12, 9), dpi=300, facecolor='white')

# Set nicer limits
ymin ,ymax = 0, 0.10
xmin, xmax = 0, 31

colors = itertools.cycle(['r', 'g', 'b','k'])
markers = itertools.cycle(['o', 'v', 's'])

ax = fig.add_subplot(1, 1, 1)
ax.set_title('Embryonic losses (10% of bulls and 1% of cows edited)')
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
fig.savefig('multiple/10_01/embryonic losses_10_01.png', dpi=300)
plt.show()

margins



