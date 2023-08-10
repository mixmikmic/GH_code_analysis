from __future__ import print_function

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from anchor.visualize import MODALITY_TO_COLOR, MODALITY_ORDER, MODALITY_PALETTE
modality_order = MODALITY_ORDER

sns.set(style='ticks', context='talk', rc={'font.sans-serif':'Arial', 'pdf.fonttype': 42})

get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

folder = 'figures'

import flotilla
study = flotilla.embark('singlecell_pnm_figure4_voyages', 
                        flotilla_dir='/projects/ps-yeolab/obotvinnik/flotilla_projects/')
# study = flotilla.embark('singlecell_pnm_figure1_supplementary_post_splicing_filtering')

study.splicing.maybe_renamed_to_feature_id('PKM')

pkm_event = 'exon:chr15:72499069-72499221:-@exon:chr15:72495363-72495529:-@exon:chr15:72494795-72494961:-@exon:chr15:72492815-72492996:-'

study.splicing.maybe_renamed_to_feature_id('SNAP25')

snap25_event = 'exon:chr20:10265372-10265420:+@exon:chr20:10273530-10273647:+@exon:chr20:10273809-10273926:+@exon:chr20:10277573-10277698:+'

study.splicing.maybe_renamed_to_feature_id('SMARCE1')

study.plot_event('SMARCE1')

smarce1_event = 'exon:chr17:38801828-38801871:-@exon:chr17:38798707-38798811:-@exon:chr17:38793744-38793824:-'

motor_neurons_metadata = study.metadata.data.query('(phenotype == "MN") & (single == True)')
print(motor_neurons_metadata.shape)
motor_neurons_metadata.head()

motor_neurons_metadata['outlier'].sum()

outlier_colors = pd.Series(['lightgrey' if x else 'black' for x in motor_neurons_metadata['outlier']], 
                           index=motor_neurons_metadata.index)
outlier_colors[:5]

motor_neurons_expression = study.expression.data.loc[motor_neurons_metadata.index]
print(motor_neurons_expression.shape)
motor_neurons_expression.head()

gene_filter = (motor_neurons_expression > 1).sum() >= 3
gene_filter.sum()

motor_neurons_expression = motor_neurons_expression.loc[:, gene_filter]
print(motor_neurons_expression.shape)
motor_neurons_expression.head()

pkm = study.splicing.data.loc[motor_neurons_metadata.index, pkm_event].dropna()
print(pkm.shape)
pkm.head()

snap25 = study.splicing.data.loc[motor_neurons_metadata.index, snap25_event].dropna()
print(snap25.shape)
snap25.head()



smarce1 = study.splicing.data.loc[motor_neurons_metadata.index, smarce1_event].dropna()
print(smarce1.shape)
smarce1.head()

motor_neurons_expression_subset = motor_neurons_expression.loc[snap25.index]
print(motor_neurons_expression_subset.shape)
motor_neurons_expression_subset.head()

get_ipython().run_line_magic('time', 'snap25_expression_corr = motor_neurons_expression_subset.apply(lambda x: x.corr(snap25))')
print(snap25_expression_corr.shape)
snap25_expression_corr.head()

sns.distplot(snap25_expression_corr.dropna())

splicing_events = {'PKM': pkm_event, 'SNAP25': snap25_event, 'SMARCE1': smarce1_event}
splicing_to_correlate = pd.DataFrame(dict((gene, study.splicing.data.loc[motor_neurons_metadata.index, event_id].dropna()) 
                                          for gene, event_id in splicing_events.items()))
print(splicing_to_correlate.shape)
splicing_to_correlate.head()

get_ipython().run_line_magic('time', 'splicing_expression_corr = splicing_to_correlate.apply(lambda x: motor_neurons_expression.apply(lambda y: y.corr(x)))')
print(splicing_expression_corr.shape)
splicing_expression_corr.head()

sns.set(style='white', context='paper')

figure_folder = 'figures/004_background_correlations'
get_ipython().system(' mkdir -p $figure_folder')

fig, ax = plt.subplots(figsize=(3, 2))
sns.violinplot(splicing_expression_corr)
xmin, xmax = ax.get_xlim()
ax.hlines([-0.5,  0.5], xmin, xmax, linestyle='--', color='darkgrey')
sns.despine()
fig.savefig('{}/expression_correlation_to_bimodal_splicing_violinplots_pearson.pdf'.format(figure_folder))

np.random.seed(int(1e6))
n_permuations = 1000

random_seeds = pd.Series(np.random.randint(low=0, high=1e6, size=n_permuations))
assert len(random_seeds.unique()) == n_permuations
random_seeds.head()

permuted = motor_neurons_expression.apply(np.random.permutation)
permuted.head()

# import joblib

# def permute_and_correlate(i, random_seed, expression, splicing):
#     np.random.seed(random_seed)
#     permuted = expression.apply(np.random.permutation)
#     df = splicing.apply(lambda x: permuted.apply(lambda y: y.corr(x)))
#     df['iteration'] = i
#     return df

# dfs = joblib.Parallel(n_jobs=-1)(joblib.delayed(
#         permute_and_correlate(i, random_seed, motor_neurons_expression, splicing_to_correlate) 
#         for i, random_seed in list(enumerate(random_seeds))))

get_ipython().run_cell_magic('time', '', "\ndfs = []\n\nfor i, random_seed in random_seeds.iteritems():\n    np.random.seed(random_seed)\n    permuted = motor_neurons_expression.apply(np.random.permutation)\n    df = splicing_to_correlate.apply(lambda x: permuted.apply(lambda y: y.corr(x)))\n    df['iteration'] = i\n    dfs.append(df)\npermuted_correlations = pd.concat(dfs, ignore_index=True)\n# permuted_correlations['dataset'] = 'Permuted'\nprint(permuted_correlations.shape)\npermuted_correlations.head()")

column_renamer = {'level_0': 'Splicing Gene', 0:'Pearson R'}

permuted_correlations_tidy = permuted_correlations.unstack().reset_index()
permuted_correlations_tidy = permuted_correlations_tidy.rename(columns=column_renamer)
permuted_correlations_tidy['dataset'] = 'Permuted'
permuted_correlations_tidy.head()

splicing_expression_corr_tidy = splicing_expression_corr.unstack().reset_index()
splicing_expression_corr_tidy = splicing_expression_corr_tidy.rename(columns=column_renamer)
splicing_expression_corr_tidy['dataset'] = 'Actual'
splicing_expression_corr_tidy.head()

correlations_tidy = pd.concat([permuted_correlations_tidy, splicing_expression_corr_tidy])
print(correlations_tidy.shape)
correlations_tidy.head()



study.supplemental.splicing_expression_corr_with_permuted = correlations_tidy

study.save('singlecell_pnm_figure6_splicing_correlation_permuted', 
                        flotilla_dir='/projects/ps-yeolab/obotvinnik/flotilla_projects/')

correlations_tidy.groupby('Splicing Gene')['Pearson R'].describe()

# correlations_tidy.to_csv('{}/splicing_expression_correlation_with_permutation.csv'.format(figure_folder))

g = sns.FacetGrid(correlations_tidy, hue='dataset', col='Splicing Gene', palette=['DarkTurquoise', 'LightGrey'], 
                  hue_order=['Permuted', 'Actual'])
g.map(sns.distplot, 'Pearson R')
g.set(xlim=(-1, 1))
g.savefig('{}/pearson_correlation_distplot.pdf'.format(folder))

# fig, ax = plt.subplots(figsize=(3, 2))
# sns.violinplot(x='Splicing Gene', y='Pearson R', data=correlations_tidy, hue='dataset', 
#                palette=['DarkTurquoise', 'lightGrey'], order=splicing_events.keys(),
#                hue_order=['Actual', 'Permuted'], cut=True)
# sns.despine()
# fig.savefig('{}/pearson_correlation_violinplots.pdf'.format(folder))

# g = sns.FacetGrid(correlations_tidy, col='Splicing Gene', hue='dataset',
#                   palette=['DarkTurquoise', 'lightGrey'], hue_order=['Actual', 'Permuted'])
# g.map(sns.distplot, 'Pearson R')

splicing_expression_corr_filtered = splicing_expression_corr[splicing_expression_corr.abs() > 0.5].dropna(how='all')
print(splicing_expression_corr_filtered.shape)
splicing_expression_corr_filtered.head()

import matplotlib as mpl

cmap = mpl.cm.RdYlBu_r

cmap

values = 0, 0.1, 0.5, 0.9, 0.99, 1, 1.0, 1.1
sns.palplot([cmap(v) for v in values])

cmap(1.1)

cmap(1.0)

get_ipython().system(' mkdir $folder')

for splicing_gene, col in splicing_expression_corr_filtered.iteritems():
    col = col.dropna()
    splicing = splicing_to_correlate[splicing_gene].dropna()
    
    subset = motor_neurons_expression.loc[splicing.index, col.index]
    subset.columns = study.expression.feature_data.loc[subset.columns, 'gene_name']
    subset = subset.T
    print(subset.shape)
    
    psi_color = [cmap(float(splicing[sample_id])) for sample_id in subset.columns]
    side_colors = [psi_color, outlier_colors[subset.columns]]
    
    g = sns.clustermap(subset, col_colors=side_colors, method='ward')
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    g.fig.suptitle(splicing_gene)
    g.savefig('{folder}/{splicing_gene}_correlated_genes_clustermap.pdf'.format(
            folder=folder, splicing_gene=splicing_gene))

for splicing_gene, col in splicing_expression_corr_filtered.iteritems():
    col = col.dropna()
    splicing = splicing_to_correlate[splicing_gene].dropna()
    
    subset = motor_neurons_expression.loc[splicing.index, col.index]
    subset.columns = study.expression.feature_data.loc[subset.columns, 'gene_name']
    subset = subset.T
    print(subset.shape)
    
    psi_color = [cmap(float(splicing[sample_id])) for sample_id in subset.columns]
    side_colors = [psi_color, outlier_colors[subset.columns]]
    
    g = sns.clustermap(subset, col_colors=side_colors, method='ward', z_score=0, cmap='PRGn')
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    g.fig.suptitle(splicing_gene)
    g.savefig('{folder}/{splicing_gene}_correlated_genes_clustermap_zscores.pdf'.format(
            folder=folder, splicing_gene=splicing_gene))

range(3)

snap25_correlated = splicing_expression_corr_filtered['SNAP25'].dropna()
snap25_correlated.head()

for x in snap25_correlated[snap25_correlated > 0].index:
    print(x)

for x in snap25_correlated[snap25_correlated < 0].index:
    print(x)

for splicing_gene, col in splicing_expression_corr_filtered.iteritems():
    col = col.dropna()
    splicing = splicing_to_correlate[splicing_gene].dropna()
    
    study.plot_pca(sample_subset='MN', feature_subset=col.index)
    fig = plt.gcf()
#     subset = motor_neurons_expression.loc[splicing.index, col.index]
#     subset.columns = study.expression.feature_data.loc[subset.columns, 'gene_name']
#     subset = subset.T
#     print(subset.shape)
    
#     psi_color = [cmap(float(splicing[sample_id])) for sample_id in subset.columns]
#     side_colors = [psi_color, outlier_colors[subset.columns]]
    
#     g = sns.clustermap(subset, col_colors=side_colors, method='ward')
#     plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
#     g.fig.suptitle(splicing_gene)
    fig.savefig('{folder}/{splicing_gene}_correlated_genes_pca_all_mn.pdf'.format(
            folder=folder, splicing_gene=splicing_gene))

for splicing_gene, col in splicing_expression_corr_filtered.iteritems():
    col = col.dropna()
    splicing = splicing_to_correlate[splicing_gene].dropna()
    
    study.plot_pca(sample_subset=splicing.index, feature_subset=col.index)
    fig = plt.gcf()
#     subset = motor_neurons_expression.loc[splicing.index, col.index]
#     subset.columns = study.expression.feature_data.loc[subset.columns, 'gene_name']
#     subset = subset.T
#     print(subset.shape)
    
#     psi_color = [cmap(float(splicing[sample_id])) for sample_id in subset.columns]
#     side_colors = [psi_color, outlier_colors[subset.columns]]
    
#     g = sns.clustermap(subset, col_colors=side_colors, method='ward')
#     plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
#     g.fig.suptitle(splicing_gene)
    fig.savefig('{folder}/{splicing_gene}_correlated_genes_pca_only_with_detected_splicing.pdf'.format(
            folder=folder, splicing_gene=splicing_gene))

get_ipython().run_line_magic('time', "splicing_expression_spearman = splicing_to_correlate.apply(lambda x: motor_neurons_expression.apply(lambda y: y.corr(x, method='spearman')))")
print(splicing_expression_spearman.shape)
splicing_expression_spearman.head()

fig, ax = plt.subplots(figsize=(3, 2))
sns.violinplot(splicing_expression_spearman)
xmin, xmax = ax.get_xlim()
ax.hlines([-0.5,  0.5], xmin, xmax, linestyle='--', color='darkgrey')

motor_neurons_splicing = study.splicing.data.loc[motor_neurons_expression.index]
print(motor_neurons_splicing.shape)
motor_neurons_splicing.head()

range(3)

var = motor_neurons_splicing.var()
var.mean()

motor_neurons_splicing = motor_neurons_splicing.loc[:, var > var.mean()]
print(motor_neurons_splicing.shape)
motor_neurons_splicing.head()

get_ipython().run_line_magic('time', 'splicing_expression_corr = splicing_to_correlate.apply(lambda x: motor_neurons_splicing.apply(lambda y: y.corr(x)))')
print(splicing_expression_corr.shape)
splicing_expression_corr.head()

sns.violinplot(splicing_expression_corr)

splicing_expression_corr.describe()

splicing_expression_corr_filtered = splicing_expression_corr[splicing_expression_corr.abs() > 0.5].dropna(how='all')
print(splicing_expression_corr_filtered.shape)
splicing_expression_corr_filtered.head()

splicing_expression_corr_filtered.notnull().sum()

splicing_events

for splicing_gene, col in splicing_expression_corr_filtered.iteritems():
    col = col.dropna()
    splicing = splicing_to_correlate[splicing_gene].dropna()
    
    subset = motor_neurons_splicing.loc[splicing.index, col.index]
    subset.columns = study.splicing.feature_data.loc[subset.columns, 'gene_name']
    subset = subset.T
    
    # Remove the gene itself
    subset = subset.drop(splicing_events[splicing_gene])
    print(subset.shape)
    
    psi_color = [cmap(float(splicing[sample_id])) for sample_id in subset.columns]
    side_colors = [psi_color, outlier_colors[subset.columns]]
    
    g = sns.clustermap(subset.fillna(subset.mean()), mask=subset.isnull(), 
                       col_colors=side_colors, method='ward', cmap='RdYlBu_r')
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    g.fig.suptitle(splicing_gene)
    g.savefig('{folder}/{splicing_gene}_correlated_events_clustermap.pdf'.format(
            folder=folder, splicing_gene=splicing_gene))

# pooled_mns = study.pooled & study.sample_subset_to_sample_ids('MN')

# outlier_colors[pooled_mns] = 'white'

thresh = 15

for splicing_gene, col in splicing_expression_corr_filtered.iteritems():
    col = col.dropna()
    splicing = splicing_to_correlate[splicing_gene].dropna()
    
    subset = motor_neurons_splicing.loc[splicing.index, col.index]
    subset = subset.drop(splicing_events[splicing_gene], axis=1)
    subset.columns = study.splicing.feature_data.loc[subset.columns, 'gene_name']
    subset = subset.T.dropna(thresh=thresh, axis=0).dropna(how='all', axis=1)
    
    # Remove the gene itself
    print(subset.shape)
    
    psi_color = [cmap(float(splicing[sample_id])) for sample_id in subset.columns]
    side_colors = [psi_color, outlier_colors[subset.columns]]
    
    mask = subset.isnull()
    plot_data = subset.fillna(subset.mean())
    
    g = sns.clustermap(plot_data, mask=mask, 
                       col_colors=side_colors, method='ward', cmap='RdYlBu_r')
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    g.fig.suptitle(splicing_gene)
    g.savefig('{folder}/{splicing_gene}_correlated_events_clustermap_dropna_thresh{thresh}.pdf'.format(
            folder=folder, splicing_gene=splicing_gene, thresh=thresh))









