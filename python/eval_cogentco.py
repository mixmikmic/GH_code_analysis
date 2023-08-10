get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# include and import place_emu/util/eval.py
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path) 
from place_emu.util import eval


bjointsp_chain_df, bjointsp_aux_chain_df, bjointsp_vnf_df, bjointsp_aux_vnf_df = eval.prepare_eval('Cogentco', 'bjointsp')
greedy_chain_df, greedy_aux_chain_df, greedy_vnf_df, greedy_aux_vnf_df = eval.prepare_eval('Cogentco', 'greedy')
random_chain_df, random_aux_chain_df, random_vnf_df, random_aux_vnf_df = eval.prepare_eval('Cogentco', 'random')

# combined dfs for easier eval
aux_vnf_df = pd.concat([bjointsp_aux_vnf_df, greedy_aux_vnf_df, random_aux_vnf_df])
vnf_df = pd.concat([bjointsp_vnf_df, greedy_vnf_df, random_vnf_df])
chain_df = pd.concat([bjointsp_chain_df, greedy_chain_df, random_chain_df])

bjointsp_chain_df.head()

bjointsp_aux_chain_df.head()

bjointsp_vnf_df.head()

sns.boxplot(x='type', y='rtt', hue='algorithm', data=aux_vnf_df).set_title('Inter-VNF RTT')

# tuned figure for paper: focus on sim_rtt (emu_rtt similar and checked later)
fig, ax = plt.subplots()
sns.set(font_scale=1.5)
sns.boxplot(x='algorithm', y='sim_rtt', data=vnf_df, color='lightgrey', ax=ax)
#ax.set_title('Sim. inter-VNF RTT')
ax.set_xlabel('Placement algorithm')
ax.set_ylabel('Inter-VNF RTT (ms)')
ax.set_facecolor('white')
fig.savefig('plots/cogentco_sim_vnf_rtt.pdf', bbox_inches='tight')

sns.boxplot(x='algorithm', y='rtt_diff', data=vnf_df).set_title('Inter-VNF RTT difference')

# tuned figure for paper
fig, ax = plt.subplots(figsize=(5,5))
sns.set(font_scale=1.5)
sns.boxplot(x='algorithm', y='rtt_diff', data=vnf_df, ax=ax, color='lightgrey')
ax.set_title('Difference emulated vs simulated inter-VNF RTT')
ax.set_xlabel('Placement algorithm')
ax.set_ylabel('Inter-VNF RTT difference (ms)')
ax.set_facecolor('white')
fig.savefig('plots/cogentco_vnf_rtt_diff.pdf', bbox_inches='tight')

# plot RTT difference as a variable of the node distance = link delay = sim delay/RTT
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
sns.pointplot(x='sim_rtt', y='rtt_diff', data=bjointsp_vnf_df, ax=ax1).set_title('B-JointSP')
sns.pointplot(x='sim_rtt', y='rtt_diff', data=greedy_vnf_df, ax=ax2).set_title('Greedy')
sns.pointplot(x='sim_rtt', y='rtt_diff', data=random_vnf_df, ax=ax3).set_title('Random placement')

# tuned figure for paper
fig, ax = plt.subplots()
sns.set(font_scale=1.5)
#black_palette = sns.color_palette(['black', 'black', 'black'])
sns.pointplot(x='sim_rtt', y='rtt_diff', data=vnf_df, ax=ax,
              hue='algorithm', join=False, markers=['o', 'x', '^'], errwidth=1.5, capsize=0.2)
#ax.set_title('Difference emulated vs simulated inter-VNF RTT')
ax.set_xlabel('Path RTT based on inter-VNF distance (ms)')
ax.set_ylabel('Inter-VNF RTT difference (ms)')
ax.set_facecolor('white')
fig.savefig('plots/cogentco_vnf_rtt_diff_sim_rtt.pdf', bbox_inches='tight')

# set infinite ratios to NaN, so they are ignored when plotting
vnf_df = vnf_df.replace(np.inf, np.nan)
for alg in ['bjointsp', 'greedy', 'random']:
    num_invalid = vnf_df.loc[(vnf_df['algorithm'] == alg) & (vnf_df['rtt_ratio'].isnull())].shape[0]
    num_total = vnf_df.loc[vnf_df['algorithm'] == alg].shape[0]
    print('Entries with invalid ratio for {}: {} of {}'.format(alg, num_invalid, num_total))

sns.boxplot(x='algorithm', y='rtt_ratio', hue='num_vnfs', data=vnf_df).set_title('Inter-VNF RTT ratio')

# comparison of bjointsp and random placement
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(x='num_vnfs', y='sim_rtt', hue='algorithm', data=chain_df, ax=ax1).set_title('Simulation chain RTT')
sns.boxplot(x='num_vnfs', y='emu_rtt', hue='algorithm', data=chain_df, ax=ax2).set_title('Emulation chain RTT')

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(x='num_vnfs', y='rtt_diff', data=bjointsp_chain_df, ax=ax1).set_title('B-JointSP chain RTT difference')
sns.boxplot(x='num_vnfs', y='rtt_diff', data=greedy_chain_df, ax=ax2).set_title('Greedy chain RTT difference')
sns.boxplot(x='num_vnfs', y='rtt_diff', data=random_chain_df, ax=ax3).set_title('Random chain RTT difference')

# set infinite ratios to NaN, so they are ignored when plotting
chain_df = chain_df.replace(np.inf, np.nan)
for alg in ['bjointsp', 'greedy', 'random']:
    num_invalid = chain_df.loc[(chain_df['algorithm'] == alg) & (chain_df['rtt_ratio'].isnull())].shape[0]
    num_total = chain_df.loc[chain_df['algorithm'] == alg].shape[0]
    print('Entries with invalid ratio for {}: {} of {}'.format(alg, num_invalid, num_total))

sns.boxplot(x='num_vnfs', y='rtt_ratio', hue='algorithm', data=chain_df).set_title('Chain RTT ratio')

