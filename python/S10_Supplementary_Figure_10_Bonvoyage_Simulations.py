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
folder = 'figures/anchor/sfig_10'
get_ipython().system('mkdir -p $folder')

size = 100

perfectly1 = np.ones(size).reshape(size, 1)
perfectly0 = np.zeros(size).reshape(size, 1)
perfectly_middle = 0.5*np.ones(size).reshape(size, 1)
maybe_middles_0 = np.vstack([np.concatenate([np.zeros(i), np.ones(size-i)*0.5]) for i in range(1, size)]).T
maybe_middles_1 = np.vstack([np.concatenate([np.ones(i), np.ones(size-i)*0.5]) for i in range(1, size)]).T
maybe_bimodals = np.vstack([np.concatenate([np.zeros(i), np.ones(size-i)]) for i in range(1, size)]).T

columns = ['perfect_middle'.format(str(i).zfill(2)) for i in range(perfectly_middle.shape[1])]     + ['middle0_{}'.format(str(i).zfill(2)) for i in range(maybe_middles_0.shape[1])]     + ['middle1_{}'.format(str(i).zfill(2)) for i in range(maybe_middles_1.shape[1])]     + ['bimodal_{}'.format(str(i).zfill(2)) for i in range(maybe_bimodals.shape[1])]     + ['perfect_included', 'perfect_excluded']

data = np.hstack([perfectly_middle, maybe_middles_0, maybe_middles_1, maybe_bimodals, perfectly1, perfectly0])
maybe_everything = pd.DataFrame(data, columns=columns)
six.print_(maybe_everything.shape)
maybe_everything.head()

from anchor.simulate import add_noise

maybe_everything_noisy = add_noise(maybe_everything, iteration_per_noise=10, 
                                   noise_percentages=np.arange(0, 101, 5), plot=False)
six.print_(maybe_everything_noisy.shape)
maybe_everything_noisy.head()

maybe_everything_noisy.to_csv('data.csv')

tidy = maybe_everything_noisy.unstack().reset_index()
tidy = tidy.rename(columns={'level_0':'Feature ID', 'level_1': "Sample ID", 0:'$\Psi$'})
six.print_(tidy.shape)
tidy.head()

tidy['Iteration'] = tidy['Feature ID'].str.extract('iter(\d+)').astype(int)
tidy['% Noise'] = tidy['Feature ID'].str.extract('noise(\d+)').astype(int)
tidy.head()

get_ipython().run_cell_magic('time', '', "split_id = tidy['Feature ID'].str.split('_').apply(pd.Series)\ntidy = pd.concat([tidy, split_id], axis=1)\ntidy.head()")

tidy.to_csv('metadata.csv', index=False)

tidy.head()

tidy.columns

noise_levels = tidy['% Noise'].isin([0, 25, 50, 75])

perfects = tidy['Feature ID'].str.contains('perfect')
middles = tidy['Feature ID'].str.startswith('middle') & tidy[1].isin(['25', '50', '75'])
bimodals = tidy['Feature ID'].str.startswith('bimodal') & tidy[1].isin(['25', '50', '75'])

row_subsets = perfects, bimodals, middles

dfs = []

for rows in row_subsets:
    df = tidy.loc[rows & noise_levels]
    dfs.append(df)
tidy_subset = pd.concat(dfs, ignore_index=True)
six.print_(tidy_subset.shape)
tidy_subset.head()

tidy_subset.groupby([0, 1, '% Noise']).size()

from anchor import MODALITY_TO_COLOR

figure_folder = 'pdf'
get_ipython().system(' mkdir $figure_folder')

sns.set(context='paper', style='ticks')

for group, df in tidy_subset.groupby(0):
    palette = None
    six.print_(group)
    
    if group == 'bimodal':
        palette = "RdBu"
    elif group == 'middle0':
        palette = 'YlGnBu'
    elif group == 'middle1':
        palette = 'YlOrRd'
    elif group == 'perfect':
        palette = [MODALITY_TO_COLOR[m] for m in df[1].unique()]
    
    g = sns.factorplot(x=1, y='$\Psi$', col='% Noise', data=df, aspect=1, size=1.5,
                   kind='violin', bw=0.2, inner=None, scale='width', sharex=False, palette=palette)
    g.set(ylim=(0, 1), yticks=(0, 0.5, 1), xlabel='')
    if group == 'perfect':
        for ax in g.axes.flat:
            plt.setp(ax.get_xticklabels(), rotation=30)
    g.savefig('{}/data_{}.pdf'.format(figure_folder, group))

# Initialize the waypoints transformer
ws = bonvoyage.Waypoints()

waypoints = ws.fit_transform(maybe_everything_noisy)
six.print_(waypoints.shape)
waypoints.head()

waypoints.to_csv('waypoints.csv')



