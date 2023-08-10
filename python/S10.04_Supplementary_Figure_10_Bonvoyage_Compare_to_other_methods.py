import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import six

sns.set(style='ticks', context='talk', rc={'font.sans-serif':'Arial', 'pdf.fonttype': 42})


import bonvoyage

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Figures in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Set random seed
np.random.seed(sum(map(ord, 'bonvoyage')))


# Define folder to save figures
folder = 'pdf'
get_ipython().system('mkdir -p $folder')


data = pd.read_csv('data.csv', index_col=0)
data.head()

mean = data.mean()
variance = data.var()
stddev = data.std()

fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(mean, variance, 'o', alpha=0.1)

fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(mean, stddev, 'o', alpha=0.1)

fig, ax = plt.subplots(figsize=(2.5, 2))
ax.hexbin(mean, variance)
ax.set(xlabel='Mean', ylabel='Variance')
sns.despine()
fig.tight_layout()
fig.savefig('{}/mean_vs_var_hexbin.pdf'.format(folder))

# fig, ax = plt.subplots(figsize=(2, 2))
g = sns.jointplot(mean, variance, alpha=0.1, color='#262626', size=3, stat_func=None, joint_kws=dict(rasterized=True))
g.ax_joint.set(xlabel='Mean', ylabel='Variance', xticks=(0, 0.5, 1), yticks=(0, 0.1, 0.2, 0.3), xlim=(0, 1), ylim=(0, 0.3))
# sns.despine()
fig.tight_layout()
fig.savefig('{}/mean_vs_var_scatter.pdf'.format(folder), dpi=600)

# fig, ax = plt.subplots(figsize=(2, 2))
g = sns.jointplot(mean, variance, alpha=0.1, color='#262626', size=3, stat_func=None, kind='hex')
g.ax_joint.set(xlabel='Mean', ylabel='Variance', xticks=(0, 0.5, 1), yticks=(0, 0.1, 0.2, 0.3), xlim=(0, 1), ylim=(0, 0.3))
# sns.despine()
fig.tight_layout()
fig.savefig('{}/mean_vs_var_hexbin.pdf'.format(folder))

np.random.seed(sum(map(ord, 'beyonce')))
shuffled_data = data.copy()
shuffled_columns = np.random.permutation(data.columns)
shuffled_data = shuffled_data[shuffled_columns]
shuffled_data.columns = data.columns
shuffled_data.head()

shuffled_data.to_csv('shuffled_data.csv')

# Initialize the waypoints transformer
ws = bonvoyage.Waypoints()

shuffled_waypoints = ws.fit_transform(shuffled_data)
six.print_(shuffled_waypoints.shape)
shuffled_waypoints.head()

shuffled_waypoints.to_csv('shuffled_waypoints.csv')

bins = np.arange(0, 1.1, .1)
six.print_(bins)


import anchor
jsd = anchor.infotheory.binify_and_jsd(data, shuffled_data, 'data vs shuffled', bins)

jsd.head()

waypoints = pd.read_csv('waypoints.csv', index_col=0)
waypoints.head()

waypoints.columns

shuffled_waypoints.columns

waypoints.columns = shuffled_waypoints.columns

waypoints['phenotype'] = 'original'
shuffled_waypoints['phenotype'] = 'shuffled'

waypoints_combined = pd.concat([waypoints, shuffled_waypoints])
waypoints_combined = waypoints_combined.set_index('phenotype', append=True)
waypoints_combined.index = waypoints_combined.index.swaplevel(0, 1)
waypoints_combined = waypoints_combined.sort_index()
waypoints_combined.head()

v = bonvoyage.Voyages()
voyages = v.voyages(waypoints_combined, [('original', 'shuffled')])

voyages.head()

voyages = voyages.set_index('event_id')
voyages.head()

voyages['magnitude'].corr(jsd)

jsd.corr(voyages['magnitude'])

voyages['magnitude'].describe()

g = sns.jointplot(voyages['magnitude'], jsd,)

g = sns.jointplot(voyages['magnitude'], jsd, kind='hex', color='#262626', size=3, stat_func=None)
g.ax_joint.set(xlabel='Voyages', ylabel='JSD', xticks=(0, 0.4, 0.8, 1.2), yticks=(0, 0.5, 1))
g.savefig('{}/jsd_vs_voyages.pdf'.format(folder))



