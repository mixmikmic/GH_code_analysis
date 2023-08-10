import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import six


sns.set(style='ticks', context='talk', rc={'font.sans-serif':'Arial', 'pdf.fonttype': 42})


import anchor


from anchor import MODALITY_ORDER, MODALITY_PALETTE, MODALITY_TO_COLOR, MODALITY_TO_CMAP

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Figures in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Set random seed
np.random.seed(sum(map(ord, 'anchor')))



# Define folder to save figures
folder = 'figures/dip_test'
get_ipython().system('mkdir -p $folder')

dataset_name_to_folder = {'Perfect Modalities': '../S05._Supplementary_Figure5',
                          'Maybe Bimodals': '../S06._Supplementary_Figure6'}

datatypes = 'data', 'metadata'
datasets = {name: {datatype: pd.read_csv('{}/{}.csv'.format(folder, datatype), index_col=0) 
              for datatype in datatypes} for name, folder in dataset_name_to_folder.items()}

def name_to_filename(name):
    return name.lower().replace(' ', '_')

for name in datasets:
    dataset_folder = '{}/{}'.format(folder, name_to_filename(name))
    get_ipython().system(' mkdir $dataset_folder')

bimodal_palette = sns.diverging_palette(247, 0, 85, 58, 10, n=99, center='dark')


import six

get_ipython().run_line_magic('pinfo', 'six.moves.range')

from diptest.diptest import diptest

for name, datas in datasets.items():
    dataset_folder = '{}/{}'.format(folder, name.lower().replace(' ', '_'))
    data = datas['data']
    metadata = datas['metadata']
    
    diptest_results = data.apply(lambda x: diptest(x.values))
    diptest_results = diptest_results.apply(lambda x: pd.Series(x, index=['Dip Statistic', '$p$-value']))
    diptest_results['log10_p_value'] = np.log10(diptest_results['$p$-value'])

    g = sns.jointplot(x='Dip Statistic', y='$p$-value', data=diptest_results, stat_func=None, 
                  size=4, ylim=(0, 1), xlim=(0, 0.25))
    g.savefig('{}/diptest_statistic_vs_p_value.pdf'.format(dataset_folder))

    diptest_results['Predicted Bimodal'] = diptest_results['$p$-value'] < 0.05

    diptest_with_metadata = metadata.join(diptest_results)
    diptest_with_metadata.head()

    if name == 'Perfect Modalities':
        # Plot raw dipstatistic values vs noise
        g = sns.factorplot(x='% Noise', y='Dip Statistic', data=diptest_with_metadata, 
                           scale=0.5, size=3, aspect=1.5, hue='Modality', 
                           palette=MODALITY_PALETTE[:-1], hue_order=MODALITY_ORDER[:-1])
        g.set(title=name)
        g.savefig('{}/dip_statistic_in_all_modalities.pdf'.format(dataset_folder))

        # PLot how often this method predicted an event as bimodal, versus noise
        g = sns.factorplot(x='% Noise', y='Predicted Bimodal', data=diptest_with_metadata, 
                       scale=0.5, size=3, aspect=1.5, hue='Modality', 
                       palette=MODALITY_PALETTE[:-1], hue_order=MODALITY_ORDER[:-1])
        g.set(title=name)
        for ax in g.axes.flat:
            ax.set(ylim=(0, 1), ylabel='% Predicted Bimodal')
            ax.set(yticklabels=[int(100*tick) for tick in ax.get_yticks()])

        g.savefig('{}/dip_statistic_predicted_bimodal.pdf'.format(dataset_folder))
    else:
        g = sns.factorplot(x='% 1s', y='Predicted Bimodal', 
                       data=diptest_with_metadata, scale=0.5, color='#262626',
                       aspect=1.5, ci=None)
        for ax in g.axes.flat:
            ymin, ymax = ax.get_ylim()
            ax.vlines([10, 90], ymin, ymax, linestyle='--')
        g.set(xticks=(0, 20, 40, 60, 80, 100), xticklabels=(0, 20, 40, 60, 80, 100), ylim=(ymin, ymax))
        g.savefig('{}_bimodals_percent_predicted_bimodal.pdf'.format(dataset_folder))

        g = sns.factorplot(x='% 1s', y='Predicted Bimodal', 
                       data=diptest_with_metadata, scale=0.5, dodge=False,
                       aspect=1.5, ci=None, hue='% Noise', palette='GnBu_r', hue_order=np.arange(0, 101, 5)[::-1])
        g.set(xticks=(0, 20, 40, 60, 80, 100), xticklabels=(0, 20, 40, 60, 80, 100))
        # g.map_dataframe(sns.pointplot,  x='% 1s', y='Predicted Bimodal', scale=0.5, ci=None, dodge=False)
        g.savefig('{}_bimodals_percent_predicted_bimodal_with_noise.pdf'.format(dataset_folder))

        g = sns.factorplot(x='% Noise', y='Predicted Bimodal', 
                       data=diptest_with_metadata, scale=0.5, dodge=False, legend=False,
                       aspect=1.5, ci=None, color='#262626')
        # g.set(xticks=(0, 20, 40, 60, 80, 100), xticklabels=(0, 20, 40, 60, 80, 100))
        g.savefig('{}_bimodals_percent_predicted_bimodal_with_noise.pdf'.format(dataset_folder))

        g = sns.factorplot(x='% Noise', y='Predicted Bimodal', 
                       data=diptest_with_metadata, scale=0.5, dodge=False, legend=False,
                       aspect=1.5, ci=None, hue='% 1s', palette=bimodal_palette, hue_order=np.arange(1, 100)[::-1])
        # g.set(xticks=(0, 20, 40, 60, 80, 100), xticklabels=(0, 20, 40, 60, 80, 100))
        g.savefig('{}_bimodals_percent_predicted_bimodal_with_noise_per_percent_1.pdf'.format(dataset_folder))

sns.choose_diverging_palette()

g = sns.factorplot(x='% 1s', y='Predicted Bimodal', 
               data=diptest_with_metadata, scale=0.5, color='#262626',
               aspect=1.5, ci=None)
for ax in g.axes.flat:
    ymin, ymax = ax.get_ylim()
    ax.vlines([10, 90], ymin, ymax, linestyle='--')
g.set(xticks=(0, 20, 40, 60, 80, 100), xticklabels=(0, 20, 40, 60, 80, 100), ylim=(ymin, ymax))
g.savefig('{}/bimodals_percent_predicted_bimodal.pdf'.format(dataset_folder))

g = sns.factorplot(x='% 1s', y='Predicted Bimodal', 
               data=diptest_with_metadata, scale=0.5, dodge=False,
               aspect=1.5, ci=None, hue='% Noise', palette='GnBu_r', hue_order=np.arange(0, 101, 5)[::-1])
g.set(xticks=(0, 20, 40, 60, 80, 100), xticklabels=(0, 20, 40, 60, 80, 100))
# g.map_dataframe(sns.pointplot,  x='% 1s', y='Predicted Bimodal', scale=0.5, ci=None, dodge=False)
g.savefig('{}/bimodals_percent_predicted_bimodal_with_noise.pdf'.format(dataset_folder))

g = sns.factorplot(x='% Noise', y='Predicted Bimodal', 
               data=diptest_with_metadata, scale=0.5, dodge=False, legend=False,
               aspect=1.5, ci=None, color='#262626')
# g.set(xticks=(0, 20, 40, 60, 80, 100), xticklabels=(0, 20, 40, 60, 80, 100))
g.savefig('{}/bimodals_percent_predicted_bimodal_with_noise.pdf'.format(dataset_folder))

g = sns.factorplot(x='% Noise', y='Predicted Bimodal', 
               data=diptest_with_metadata, scale=0.5, dodge=False, legend=False,
               aspect=1.5, ci=None, hue='% 1s', palette='RdBu_r', hue_order=np.arange(1, 100)[::-1])
# g.set(xticks=(0, 20, 40, 60, 80, 100), xticklabels=(0, 20, 40, 60, 80, 100))
g.savefig('{}/bimodals_percent_predicted_bimodal_with_noise_per_percent_1.pdf'.format(dataset_folder))

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(1, 2))
cmap = mpl.cm.RdBu_r
norm = mpl.colors.Normalize(vmin=1, vmax=99)
mpl.colorbar.ColorbarBase(ax, cmap=cmap, label='% 1s', norm=norm, ticks=[0, 20, 40, 60, 80, 100])
fig.tight_layout()
fig.savefig('maybe_bimodals_percent_ones_colorbar_cmap.pdf')

diptest_with_metadata.head()



metadata.head()

diptest_with_predicted = bayesian_predictions.join(diptest_results, on='Feature ID')
diptest_with_predicted.head()

g = sns.factorplot(x='% Noise', y='Dip Statistic', data=diptest_with_predicted, 
                   scale=0.5, size=3, aspect=1.5, hue='Original Modality', 
                   palette=MODALITY_PALETTE[:-1], hue_order=MODALITY_ORDER[:-1])
g.set(title='Diptest')
for ax in g.axes.flat:
    ax.set(yticks=[0, 0.04, .08, 0.12, 0.16])
g.savefig('{}/dip_statistic.pdf'.format(dataset_folder))

g = sns.factorplot(x='% Noise', y='Predicted Bimodal', data=diptest_with_predicted, 
                   scale=0.5, size=3, aspect=1.5, hue='Original Modality', 
                   palette=MODALITY_PALETTE[:-1], hue_order=MODALITY_ORDER[:-1])
g.set(title='Diptest')
for ax in g.axes.flat:
    ax.set(ylim=(0, 1), ylabel='% Predicted Bimodal')
    ax.set(yticklabels=[int(100*tick) for tick in ax.get_yticks()])

g.savefig('{}/dip_statistic_predicted_bimodal.pdf'.format(dataset_folder))







