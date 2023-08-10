get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import seaborn as sns

# include and import place_emu/util/eval.py
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path) 
from place_emu.util import eval

sns.set(font_scale=1.5)
sns.set_style("white")

# read all results
vnf_df = pd.DataFrame()
chain_df = pd.DataFrame()

for net in ['Abilene', 'Airtel', 'Cogentco']:
    for alg in ['bjointsp', 'greedy', 'random']:
        tmp_chain_df, tmp_aux_chain_df, tmp_vnf_df, tmp_aux_vnf_df = eval.prepare_eval(net, alg)

        vnf_df = pd.concat([vnf_df, tmp_vnf_df])
        chain_df = pd.concat([chain_df, tmp_chain_df])

chain_df.head()

chain_df.tail()

vnf_df.head()

vnf_df.tail()

# calc distance in 1000 km between connected VNFs
c = 299792458   # speed of light in m/s
prop = 0.77   # propagation factor
# divide by 1000 for ms to s and by another 1000 for m to km and another 1000 for km to 1000 km
vnf_df['distance'] = ((vnf_df['sim_rtt'] * c) / (prop * 1000 * 1000 * 1000)).astype('int')
vnf_df.head()

# tuned figure for paper
fig, ax = plt.subplots()

#black_palette = sns.color_palette(['black', 'black', 'black'])
sns.pointplot(x='distance', y='rtt_diff', data=vnf_df, ax=ax,
              hue='network', join=False, markers=['o', 'x', '^'])
sns.despine()
ax.set_xlabel('Path distance [1000 km]')
ax.set_xlim(0, 19)
ax.set_ylabel('Emu.-Sim. RTT difference [ms]')

# fewer x-axis ticks
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

# legend outside, custom legend title
#legend = plt.legend(title='Network', bbox_to_anchor=(1.05, 1))
#plt.setp(legend.get_title(),fontsize='small')

fig.savefig('plots/combined_vnf_rtt.pdf', bbox_inches='tight')

# only consider greedy placements
greedy_chain = chain_df.loc[chain_df['algorithm'] == 'greedy']

fig, ax = plt.subplots()

grayscale = sns.color_palette(['darkgray', 'lightgray', 'white'])
sns.boxplot(x='num_vnfs', y='rtt_diff', hue='network', data=greedy_chain, palette=grayscale, ax=ax)
sns.despine()
ax.set_xlabel('Num. VNFs')
ax.set_ylabel('End-to-end RTT \nmodel vs. emulation diff. [ms]')

small = mpatches.Patch(color='darkgray', label='small')
medium = mpatches.Patch(color='lightgray', label='medium')
large = mpatches.Patch(facecolor='white', edgecolor='black', label='large')
legend = plt.legend(title='Network size', handles=[small, medium, large])
plt.setp(legend.get_title(),fontsize='small')

fig.savefig('plots/combined_chain_rtt.pdf', bbox_inches='tight')

