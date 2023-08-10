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

# outlier_colors = pd.Series(['lightgrey' if x else 'black' for x in motor_neurons_metadata['outlier']], 
#                            index=motor_neurons_metadata.index)
# outlier_colors[:5]

single_metadata = study.metadata.data.query('(single == True)')
print(single_metadata.shape)
single_metadata.head()

single_expression = study.expression.singles
print(single_expression.shape)
single_expression.head()

gene_filter = (single_expression > 1).sum() >= 30
gene_filter.sum()

single_expression = single_expression.loc[:, gene_filter]
print(single_expression.shape)
single_expression.head()

splicing_events = (('BRD8-event1', 'exon:chr5:137495758-137495862:- exon:chr5:137495244-137495288:- exon:chr5:137492571-137492956:-'),
                   ('BRD8-event2', 'exon:chr5:137500009-137500102:- exon:chr5:137499776-137499822:- exon:chr5:137498819-137499033:-'),
                   ("DYNC1I2", 'exon:chr2:172563743-172563887:+ exon:chr2:172569277-172569336:+ exon:chr2:172571838-172571878:+'),
                   ('EIF5', 'exon:chr14:103800339-103800597:+ exon:chr14:103800726-103800934:+ exon:chr14:103801990-103802269:+'),
                   ('EIF6', 'exon:chr20:33871979-33872295:- exon:chr20:33868457-33868632:- exon:chr20:33867745-33867921:-'),
                   ("MDM4", 'exon:chr1:204501319-204501374:+ exon:chr1:204506558-204506625:+ exon:chr1:204507337-204507436:+'),
                   ("MEAF6", 'exon:chr1:37967405-37967597:- exon:chr1:37962308-37962337:- exon:chr1:37961475-37961519:-'),
                   ("RPN2", 'exon:chr20:35864983-35865112:+ exon:chr20:35866805-35866852:+ exon:chr20:35869706-35869820:+'),
                   ('SUGT1', 'exon:chr13:53233314-53233384:+ exon:chr13:53235610-53235705:+ exon:chr13:53236784-53236837:+'))

splicing_events = list(((x, '@'.join(y.split())) for x, y in splicing_events))
splicing_events


splicing_to_correlate = pd.DataFrame(dict((gene, study.splicing.singles[event_id].dropna()) 
                                          for gene, event_id in splicing_events))
print(splicing_to_correlate.shape)
splicing_to_correlate.head()

figure_folder = 'figures/012_background_correlations_track_examples'
get_ipython().system(' mkdir -p $figure_folder')

np.random.seed(int(1e6))
n_permuations = 1000

random_seeds = pd.Series(np.random.randint(low=0, high=1e6, size=n_permuations))
assert len(random_seeds.unique()) == n_permuations
random_seeds.head()

get_ipython().run_cell_magic('time', '', "\ndfs = []\n\nfor i, random_seed in random_seeds.iteritems():\n    np.random.seed(random_seed)\n    permuted = single_expression.apply(np.random.permutation)\n    df = splicing_to_correlate.groupby(study.sample_id_to_phenotype).apply(\n        lambda df: df.apply(lambda x: permuted.apply(lambda y: y.corr(x))))\n#     df = splicing_to_correlate.apply(lambda x: permuted.apply(lambda y: y.corr(x)))\n    df['iteration'] = i\n    dfs.append(df)\npermuted_correlations = pd.concat(dfs, ignore_index=True)\n# permuted_correlations['dataset'] = 'Permuted'\nprint(permuted_correlations.shape)\npermuted_correlations.head()")

column_renamer = {'level_0': 'Splicing Gene', 0:'Pearson R', 'level_1': 'iteration'}

permuted_correlations_tidy = permuted_correlations.unstack(levels=[0, 1]).reset_index()
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









