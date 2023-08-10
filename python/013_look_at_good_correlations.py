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


import flotilla
study = flotilla.embark('singlecell_pnm_figure6_splicing_correlation_track_examples', 
                        flotilla_dir='/projects/ps-yeolab/obotvinnik/flotilla_projects/')
# study = flotilla.embark('singlecell_pnm_figure1_supplementary_post_splicing_filtering')

folder = 'figures'

figure_folder = 'figures/013_look_at_good_correlations'
get_ipython().system(' mkdir -p $figure_folder')

splicing_expression_corr_tidy = study.supplemental.splicing_expression_corr
splicing_expression_corr_tidy = splicing_expression_corr_tidy.loc[splicing_expression_corr_tidy['Pearson R'].abs() > 0.5]
splicing_expression_corr_tidy = splicing_expression_corr_tidy.join(study.expression.feature_data['gene_name'], on='Expression Gene')
splicing_expression_corr_tidy['Correlation Direction'] = splicing_expression_corr_tidy['Pearson R'].map(
    lambda x: '(+) Positive' if x > 0 else '(-) Negative')
print(splicing_expression_corr_tidy.shape)
splicing_expression_corr_tidy.head()

study.expression.feature_data.head()

splicing_expression_corr_tidy_gene_types = splicing_expression_corr_tidy.join(study.expression.feature_data['gene_type'], on='Expression Gene')
print(splicing_expression_corr_tidy_gene_types.shape)
splicing_expression_corr_tidy_gene_types.head()

splicing_expression_corr_tidy_protein_coding = splicing_expression_corr_tidy_gene_types.query('gene_type == "protein_coding"')
print(splicing_expression_corr_tidy_protein_coding.shape)
splicing_expression_corr_tidy_protein_coding.head()

for (splicing_gene, phenotype), df in splicing_expression_corr_tidy_protein_coding.groupby(['Splicing Gene', 'phenotype']):
    print(splicing_gene, df.shape)
    df.to_csv('{}/{}_{}_correlations.csv'.format(figure_folder, splicing_gene, phenotype))

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

single_splicing = study.splicing.singles

single_metadata = study.metadata.data.query('(single == True)')
print(single_metadata.shape)
single_metadata.head()

outlier_colors = pd.Series(['lightgrey' if x else 'black' for x in single_metadata['outlier']], 
                           index=single_metadata.index)
outlier_colors[:5]

single_expression = study.expression.singles
print(single_expression.shape)
single_expression.head()

gene_filter = (single_expression > 1).sum() >= 30
gene_filter.sum()

single_expression = single_expression.loc[:, gene_filter]
print(single_expression.shape)
single_expression.head()

splicing_to_correlate = pd.DataFrame(dict((gene, study.splicing.singles[event_id].dropna()) 
                                          for gene, event_id in splicing_events))
print(splicing_to_correlate.shape)
splicing_to_correlate.head()

# fig, ax = plt.subplots(figsize=(3, 2))
# sns.violinplot(x='Splicing Gene', y='Pearson R', data=correlations_tidy, hue='dataset', 
#                palette=['DarkTurquoise', 'lightGrey'], order=splicing_events.keys(),
#                hue_order=['Actual', 'Permuted'], cut=True)
# sns.despine()
# fig.savefig('{}/pearson_correlation_violinplots.pdf'.format(folder))

# g = sns.FacetGrid(correlations_tidy, col='Splicing Gene', hue='dataset',
#                   palette=['DarkTurquoise', 'lightGrey'], hue_order=['Actual', 'Permuted'])
# g.map(sns.distplot, 'Pearson R')

import matplotlib as mpl

cmap = mpl.cm.RdYlBu_r

get_ipython().system(' mkdir $folder')

figure_folder = '{}/013_look_at_good_correlations'.format(folder)
get_ipython().system(' mkdir $figure_folder')

sns.set(context='paper', style='white')

for (splicing_gene, phenotype), df in splicing_expression_corr_tidy_protein_coding.groupby(['Splicing Gene', 'phenotype']):
    splicing = splicing_to_correlate[splicing_gene].dropna()
    
    samples = single_metadata.query('phenotype == @phenotype').index
    samples = splicing.index.intersection(samples)
    splicing = splicing[samples]
    
    genes = df['Expression Gene']
    gene_names = df['gene_name']
    
    subset = single_expression.loc[samples, genes]
    subset.columns = gene_names
    subset = subset.T
    print(splicing_gene, phenotype, subset.shape)
    if subset.shape[0] < 2:
        continue

    psi_color = [cmap(float(splicing[sample_id])) for sample_id in subset.columns]
    phenotype_color = [study.phenotype_to_color[study.sample_id_to_phenotype[sample_id]]
                       for sample_id in subset.columns]
    side_colors = [phenotype_color, psi_color, outlier_colors[subset.columns]]
    
    g = sns.clustermap(subset, col_colors=side_colors, method='ward')
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    g.fig.suptitle(splicing_gene)
    g.savefig('{folder}/{splicing_gene}_{phenotype}_correlated_genes_clustermap.pdf'.format(
            folder=figure_folder, splicing_gene=splicing_gene, phenotype=phenotype))
    
    subset.iloc[g.dendrogram_row.reordered_ind, :0].to_csv(
        '{folder}/{splicing_gene}_{phenotype}_correlated_genes_clustered_order.csv'.format(
            folder=figure_folder, splicing_gene=splicing_gene, phenotype=phenotype))

figure_folder

rows = (splicing_expression_corr_tidy_protein_coding['Splicing Gene'] == 'EIF5') & (splicing_expression_corr_tidy_protein_coding['phenotype'] == 'NPC')
splicing_expression_corr_tidy_protein_coding.loc[rows]

folder

figure_folder

for (splicing_gene, phenotype), df in splicing_expression_corr_tidy_protein_coding.groupby(['Splicing Gene', 'phenotype']):
    splicing = splicing_to_correlate[splicing_gene].dropna()
    
    samples = single_metadata.query('phenotype == @phenotype').index
    samples = splicing.index.intersection(samples)
    splicing = splicing[samples]
    
    genes = df['Expression Gene']
    gene_names = df['gene_name']
    print(splicing_gene, '\t', phenotype, '\tsamples:', len(samples), '\tgenes:', len(genes))
    if len(genes) < 2:
        continue
        
    pcaviz = study.plot_pca(sample_subset=samples, feature_subset=genes)
    pcaviz.ax_components.set(title=splicing_gene)
    fig = plt.gcf()

    fig.savefig('{folder}/{splicing_gene}_{phenotype}_correlated_genes_pca.pdf'.format(
            folder=figure_folder, splicing_gene=splicing_gene, phenotype=phenotype))









