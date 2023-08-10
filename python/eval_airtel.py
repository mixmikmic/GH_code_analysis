get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


bjointsp_chain_df, bjointsp_aux_chain_df, bjointsp_vnf_df, bjointsp_aux_vnf_df = eval.prepare_eval('Airtel', 'bjointsp')
greedy_chain_df, greedy_aux_chain_df, greedy_vnf_df, greedy_aux_vnf_df = eval.prepare_eval('Airtel', 'greedy')
random_chain_df, random_aux_chain_df, random_vnf_df, random_aux_vnf_df = eval.prepare_eval('Airtel', 'random')

# combined dfs for easier eval
aux_vnf_df = pd.concat([bjointsp_aux_vnf_df, greedy_aux_vnf_df, random_aux_vnf_df])
vnf_df = pd.concat([bjointsp_vnf_df, greedy_vnf_df, random_vnf_df])
chain_df = pd.concat([bjointsp_chain_df, greedy_chain_df, random_chain_df])

bjointsp_chain_df.head()

bjointsp_aux_chain_df.head()

bjointsp_vnf_df.head()

sns.boxplot(x='type', y='rtt', hue='algorithm', data=aux_vnf_df).set_title('Inter-VNF RTT')

sns.boxplot(x='algorithm', y='rtt_diff', data=vnf_df).set_title('Inter-VNF RTT difference')

# plot RTT difference as a variable of the node distance = link delay = sim delay/RTT
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
sns.pointplot(x='sim_rtt', y='rtt_diff', data=bjointsp_vnf_df, ax=ax1).set_title('B-JointSP')
sns.pointplot(x='sim_rtt', y='rtt_diff', data=greedy_vnf_df, ax=ax2).set_title('Greedy')
sns.pointplot(x='sim_rtt', y='rtt_diff', data=random_vnf_df, ax=ax3).set_title('Random placement')

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
              hue='algorithm', join=False, markers=['o', 'x', '^'])
sns.despine()
ax.set_xlabel('Path distance [1000 km]')
ax.set_ylabel('Inter-VNF RTT \nmodel vs. emulation diff. [ms]')

# fewer x-axis ticks
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

# legend outside, custom legend title
legend = plt.legend(title='Placement alg.', bbox_to_anchor=(1.05, 1))
plt.setp(legend.get_title(),fontsize='small')

fig.savefig('plots/airtel_vnf_rtt.pdf', bbox_inches='tight')

# set infinite ratios to NaN, so they are ignored when plotting
print('Entries with inifinite ratio: {}'.format(vnf_df['rtt_ratio'].loc[vnf_df['rtt_ratio'] == np.inf].count()))
vnf_df = vnf_df.replace(np.inf, np.nan)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(x='algorithm', y='rtt_ratio', hue='num_vnfs', data=vnf_df, ax=ax1).set_title('Inter-VNF RTT ratio')
# splitting the plot up for different chain lengths doesn't help so much
sns.boxplot(x='algorithm', y='rtt_ratio', data=vnf_df, ax=ax2).set_title('Inter-VNF RTT ratio')

# comparison of bjointsp and random placement
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(x='num_vnfs', y='sim_rtt', hue='algorithm', data=chain_df, ax=ax1).set_title('Simulation chain RTT')
sns.boxplot(x='num_vnfs', y='emu_rtt', hue='algorithm', data=chain_df, ax=ax2).set_title('Emulation chain RTT')

# comparison of simulation and emulation delays
# this is what aux_chain_df is for
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(x='num_vnfs', y='rtt', hue='type', data=bjointsp_aux_chain_df, ax=ax1).set_title('B-JointSP chain RTT')
sns.boxplot(x='num_vnfs', y='rtt', hue='type', data=greedy_aux_chain_df, ax=ax2).set_title('Greedy chain RTT')
sns.boxplot(x='num_vnfs', y='rtt', hue='type', data=random_aux_chain_df, ax=ax3).set_title('Random chain RTT')

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(x='num_vnfs', y='rtt_diff', data=bjointsp_chain_df, ax=ax1).set_title('B-JointSP chain RTT difference')
sns.boxplot(x='num_vnfs', y='rtt_diff', data=greedy_chain_df, ax=ax2).set_title('Greedy chain RTT difference')
sns.boxplot(x='num_vnfs', y='rtt_diff', data=random_chain_df, ax=ax3).set_title('Random chain RTT difference')

# tuned figure for paper
fig, ax = plt.subplots()
grayscale = sns.color_palette(['darkgray', 'lightgray', 'white'])
sns.boxplot(x='num_vnfs', y='rtt_diff', hue='algorithm', data=chain_df, ax=ax, palette=grayscale)
#ax.set_title('Difference emulated vs simulated chain RTT')
ax.set_xlabel('Chain length (#VNFs)')
ax.set_ylabel('Chain RTT difference (ms)')
ax.set_facecolor('white')
fig.savefig('plots/airtel_chain_rtt_diff.pdf', bbox_inches='tight')

# calc distance in 1000 km between connected VNFs
c = 299792458   # speed of light in m/s
prop = 0.77   # propagation factor
# divide by 1000 for ms to s and by another 1000 for m to km and another 1000 for km to 1000 km
chain_df['distance'] = ((chain_df['sim_rtt'] * c) / (prop * 1000 * 1000 * 1000)).astype('int')
chain_df.head()

# tuned figure for paper
fig, ax = plt.subplots()
#black_palette = sns.color_palette(['black', 'black', 'black'])
sns.pointplot(x='distance', y='rtt_diff', data=chain_df, ax=ax,
              hue='num_vnfs', join=False, markers=['o', 'x', '^'])
sns.despine()
ax.set_xlabel('Path distance [1000 km]')
ax.set_ylabel('End-to-end RTT \nmodel vs. emulation diff. [ms]')

# fewer x-axis ticks
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

# legend outside, custom legend title
legend = plt.legend(title='Num. VNFs', bbox_to_anchor=(1.05, 1))
plt.setp(legend.get_title(),fontsize='small')

fig.savefig('plots/airtel_chain_rtt.pdf', bbox_inches='tight')

sns.boxplot(x='num_vnfs', y='rtt_ratio', hue='algorithm', data=chain_df).set_title('Chain RTT ratio')

