from __future__ import print_function

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from anchor.visualize import MODALITY_TO_COLOR, MODALITY_ORDER, MODALITY_PALETTE
modality_order = MODALITY_ORDER

sns.set(style='ticks', context='paper', rc={'font.sans-serif':'Arial', 'pdf.fonttype': 42})

get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import flotilla
study = flotilla.embark('singlecell_pnm_figure6_splicing_correlation_track_examples', 
                        flotilla_dir='/projects/ps-yeolab/obotvinnik/flotilla_projects/')
# study = flotilla.embark('singlecell_pnm_figure1_supplementary_post_splicing_filtering')

folder = 'figures'

figure_folder = 'figures/016_bimodal_correlations_pca'
get_ipython().system(' mkdir -p $figure_folder')

splicing_expression_corr_tidy = study.supplemental.splicing_expression_corr
splicing_expression_corr_tidy = splicing_expression_corr_tidy.loc[splicing_expression_corr_tidy['Pearson R'].abs() > 0.5]
splicing_expression_corr_tidy = splicing_expression_corr_tidy.join(study.expression.feature_data['gene_name'], on='Expression Gene')
splicing_expression_corr_tidy['Correlation Direction'] = splicing_expression_corr_tidy['Pearson R'].map(
    lambda x: '(+) Positive' if x > 0 else '(-) Negative')
print(splicing_expression_corr_tidy.shape)
splicing_expression_corr_tidy.head()

splicing_events = (('BRD8-event1', 'exon:chr5:137495758-137495862:- exon:chr5:137495244-137495288:- exon:chr5:137492571-137492956:-'),
                   ('BRD8-event2', 'exon:chr5:137500009-137500102:- exon:chr5:137499776-137499822:- exon:chr5:137498819-137499033:-'),
                   ("DYNC1I2", 'exon:chr2:172563743-172563887:+ exon:chr2:172569277-172569336:+ exon:chr2:172571838-172571878:+'),
                   ('EIF5', 'exon:chr14:103800339-103800597:+ exon:chr14:103800726-103800934:+ exon:chr14:103801990-103802269:+'),
                   ('EIF6', 'exon:chr20:33871979-33872295:- exon:chr20:33868457-33868632:- exon:chr20:33867745-33867921:-'),
                   ("MDM4", 'exon:chr1:204501319-204501374:+ exon:chr1:204506558-204506625:+ exon:chr1:204507337-204507436:+'),
                   ("MEAF6", 'exon:chr1:37967405-37967597:- exon:chr1:37962308-37962337:- exon:chr1:37961475-37961519:-'),
                   ("RPN2", 'exon:chr20:35864983-35865112:+ exon:chr20:35866805-35866852:+ exon:chr20:35869706-35869820:+'),
                   ('SUGT1', 'exon:chr13:53233314-53233384:+ exon:chr13:53235610-53235705:+ exon:chr13:53236784-53236837:+'),
                   ('PKM', 'exon:chr15:72499069-72499221:-@exon:chr15:72495363-72495529:-@exon:chr15:72494795-72494961:-@exon:chr15:72492815-72492996:-'),
                   ('SNAP25', 'exon:chr20:10265372-10265420:+@exon:chr20:10273530-10273647:+@exon:chr20:10273809-10273926:+@exon:chr20:10277573-10277698:+'),
                   ('SMARCE1', 'exon:chr17:38801828-38801871:- exon:chr17:38798707-38798811:- exon:chr17:38793744-38793824:-'))

splicing_events = list(((x, '@'.join(y.split())) for x, y in splicing_events))
splicing_events

single_metadata = study.metadata.data.query('single == True')

outlier_colors = pd.Series(['lightgrey' if x else 'black' for x in single_metadata['outlier']], 
                           index=single_metadata.index)
outlier_colors[:5]

single_expression = study.expression.singles
print(single_expression.shape)
single_expression.head()

# gene_filter = (single_expression > 1).sum() >= 30
# gene_filter.sum()

# single_expression = single_expression.loc[:, gene_filter]
# print(single_expression.shape)
# single_expression.head()

splicing_to_correlate = pd.DataFrame(dict((gene, study.splicing.singles[event_id].dropna()) 
                                          for gene, event_id in splicing_events))
print(splicing_to_correlate.shape)
splicing_to_correlate.head()

splicing_expression_corr_tidy.head()

splicing_corr_phenotypes = splicing_expression_corr_tidy.drop(['Expression Gene', 'Pearson R', 'gene_name', 'Correlation Direction'], axis=1)
splicing_corr_phenotypes = splicing_corr_phenotypes.drop_duplicates()
print(splicing_corr_phenotypes.shape)
splicing_corr_phenotypes.head()

splicing_corr_phenotypes

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
single_expression_decomposed = pd.DataFrame(pca.fit_transform(single_expression), index=single_expression.index)
print(single_expression_decomposed.shape)
single_expression_decomposed.head()

pcas = {}
dfs = []

for phenotype in study.phenotype_order:
    print(phenotype)
    samples = study.sample_subset_to_sample_ids(phenotype)
    single_samples = single_expression.index & samples
    print('\t', len(single_samples))
    pca = PCA(n_components=2)
    smushed = pd.DataFrame(pca.fit_transform(single_expression.loc[single_samples]), 
                           index=single_samples)
    smushed['phenotype'] = phenotype
    dfs.append(smushed.reset_index())
#     smusheds[phenotype] = smushed
    pcas[phenotype] = pca
smushed_phenotype = pd.concat(dfs, ignore_index=True)
print(smushed_phenotype.shape)
smushed_phenotype = smushed_phenotype.rename(columns={'index': 'sample_id'})
smushed_phenotype.head()

import matplotlib as mpl

cmap = mpl.cm.RdYlBu_r
cmap.set_under('white')
# cmap.se

for i, row in splicing_corr_phenotypes.iterrows():
    
    phenotype = row['phenotype']
    splicing_gene = row['Splicing Gene']
    
    fig, ax = plt.subplots(figsize=(2, 1.5))
    
    psi = splicing_to_correlate[row['Splicing Gene']].dropna()
#     psi = splicing_to_correlate[splicing_gene].fillna(-1)
    samples = study.sample_subset_to_sample_ids(phenotype)
    samples = psi.index & samples
    psi = psi[samples]
#     colors = cmap(psi)

    smushed = smushed_phenotype.query('phenotype == @phenotype')
    smushed = smushed.set_index('sample_id')
    
    pca = pcas[phenotype]
    percent_explained = 100 * pca.explained_variance_ratio_
    xlabel = 'PC1 ({:.1f}%)'.format(percent_explained[0])
    ylabel = 'PC2 ({:.1f}%)'.format(percent_explained[1])
    
    ax.scatter(smushed.loc[psi.index, 0], smushed.loc[psi.index, 1], 
               c=psi.values, cmap=cmap, vmin=0, vmax=1, linewidths=0.5, s=20)
    ax.set(title='{phenotype} {splicing_gene}'.format(phenotype=phenotype, splicing_gene=splicing_gene), 
           xticks=[], yticks=[], xlabel=xlabel, ylabel=ylabel)
#     ax.locator_params(nbins=3)
    sns.despine()
    fig.savefig('{folder}/{splicing_gene}_{phenotype}_full_dataset_pca_colored_by_{splicing_gene}_psi.pdf'.format(
            phenotype=phenotype, splicing_gene=splicing_gene, folder=figure_folder))

correlated_smusheds = {}

for (splicing_gene, phenotype), df in splicing_expression_corr_tidy.groupby(['Splicing Gene', 'phenotype']):
    print(splicing_gene, phenotype, df.shape)
    if df.shape[0] < 2:
        continue
    psi = splicing_to_correlate[splicing_gene].dropna()
    samples = study.sample_subset_to_sample_ids(phenotype)
    samples = psi.index & samples
    psi = psi[samples]
#     colors = cmap(psi)
    
    subset = single_expression.loc[samples, df['Expression Gene']]
    
    pca = PCA(n_components=2)
    smushed = pd.DataFrame(pca.fit_transform(subset), index=subset.index)
    correlated_smusheds[(splicing_gene, phenotype)] = smushed
    
    percent_explained = 100 * pca.explained_variance_ratio_
    xlabel = 'PC1 ({:.1f}%)'.format(percent_explained[0])
    ylabel = 'PC2 ({:.1f}%)'.format(percent_explained[1])

    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.scatter(smushed.loc[psi.index, 0], smushed.loc[psi.index, 1], 
               c=psi.values, cmap=cmap, vmin=0, vmax=1, linewidths=0.5, s=20)
    ax.set(title='{phenotype} {splicing_gene}'.format(phenotype=phenotype, splicing_gene=splicing_gene),
           xticks=[], yticks=[], xlabel=xlabel, ylabel=ylabel)
    ax.locator_params(nbins=3)
    sns.despine()
    
    fig.savefig('{folder}/{splicing_gene}_{phenotype}_correlated_pca_colored_by_psi.pdf'.format(
            phenotype=phenotype, splicing_gene=splicing_gene, folder=figure_folder))



