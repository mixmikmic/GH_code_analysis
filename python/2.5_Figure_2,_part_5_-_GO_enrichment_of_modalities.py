get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import six

sns.set(style='ticks', context='paper', rc={'font.sans-serif':'Arial', 'pdf.fonttype': 42})

get_ipython().run_line_magic('matplotlib', 'inline')


folder = 'figures'

import flotilla

flotilla_dir = '/projects/ps-yeolab/obotvinnik/flotilla_projects'
study = flotilla.embark('singlecell_pnm_figure2_modalities_bayesian', flotilla_dir=flotilla_dir)
not_outliers = study.splicing.singles.index.difference(study.splicing.outliers.index)

psi = study.splicing.singles.ix[not_outliers]
grouped = psi.groupby(study.sample_id_to_phenotype)
psi_filtered = grouped.apply(lambda x: x.dropna(axis=1, thresh=20))

figure_folder = '{}/gene_ontology'.format(folder)
get_ipython().system(' mkdir $figure_folder')

study.splicing.feature_expression_id_col = 'ensembl_id'

np.finfo(np.float128).eps

all_bimodal_events = study.supplemental.modalities_tidy.query('modality == "bimodal"')['event_id'].unique()
len(all_bimodal_events)

sns.set(style='white', context='paper')

from flotilla.visualize.gene_ontology import plot_go_enrichment
gos = []
for (phenotype), phenotype_df in study.supplemental.modalities_tidy.groupby(['phenotype']):
    print phenotype
    background = study.splicing.splicing_to_expression_id(phenotype_df.event_id)
    for modality, modality_df in phenotype_df.groupby('modality'):
        print '\t', modality
        modality_genes = study.splicing.splicing_to_expression_id(modality_df.event_id)
        go = study.go_enrichment(modality_genes, background, #p_value_cutoff=0.01, 
                                 min_feature_size=5, min_background_size=10,
                                 domain='biological_process')
        if go is None or go.empty:
            print '... empty GO enrichment!'
            continue
        go = go.iloc[:10, :]
        fig, ax = plt.subplots(figsize=(1, 1))
        ax = plot_go_enrichment(data=go, color='grey', zorder=-1)
        ax.set_title('{} {}'.format(phenotype, modality))
        ax.grid(axis='x', color='white', zorder=100, linewidth=0.5)
        fig = plt.gcf()
#         fig.tight_layout()
        suffix = 'modality_go_enrichment_within_celltype_{}_{}'.format(modality, phenotype)
        go.to_csv('{}/{}.csv'.format(figure_folder, suffix))
        fig.savefig('{}/{}.pdf'.format(figure_folder, suffix))

from flotilla.visualize.gene_ontology import plot_go_enrichment
gos = []
for (phenotype), phenotype_df in study.supplemental.modalities_tidy.groupby(['phenotype']):
    print phenotype
    background = study.splicing.splicing_to_expression_id(phenotype_df.event_id)
    for modality, modality_df in phenotype_df.groupby('modality'):
        print '\t', modality
        modality_genes = study.splicing.splicing_to_expression_id(modality_df.event_id)
        go = study.go_enrichment(modality_genes, background, #p_value_cutoff=0.01, 
                                 min_feature_size=5, min_background_size=10)
        if go is None or go.empty:
            print '... empty GO enrichment!'
            continue
        go = go.iloc[:10, :]
        fig, ax = plt.subplots(figsize=(1, 1))
        ax = plot_go_enrichment(data=go, color='grey', zorder=-1)
        ax.set_title('{} {}'.format(phenotype, modality))
        fig = plt.gcf()
#         fig.tight_layout()
        prefix = '{}/modality_go_enrichment_within_celltype_{}_{}_all_domains'.format(figure_folder, modality, phenotype)
        go.to_csv('{}.csv'.format(prefix))
        fig.savefig('{}.pdf'.format(prefix))

import itertools

modalities_all_celltypes = study.supplemental.modalities_tidy.groupby('event_id').filter(
    lambda x: len(x)==len(study.phenotype_order))

go_dfs = []

for (modality), modality_df in modalities_all_celltypes.groupby(['modality']):
    print modality
    background = study.splicing.splicing_to_expression_id(modality_df.event_id)
    for phenotype, phenotype_df in modality_df.groupby('phenotype'):
        print '\t', phenotype
        phenotype_genes = study.splicing.splicing_to_expression_id(phenotype_df.event_id)
        go = study.go_enrichment(phenotype_genes, background, #p_value_cutoff=0.01, 
#                                  min_feature_size=5, min_background_size=10,
                                domain='biological_process')
        if go is None or go.empty:
            print '\t... empty GO enrichment!'
            continue
            
        go['modality'] = modality
        go['phenotype'] = phenotype
        go_dfs.append(go.reset_index())
        fig, ax = plt.subplots(figsize=(2, 2))
        ax = plot_go_enrichment(data=go, color='grey')
        ax.set_title('{} {}'.format(phenotype, modality))
        fig = plt.gcf()
#         fig.tight_layout()
        prefix = '{}/modality_go_enrichment_across_celltypes_{}_{}'.format(figure_folder, phenotype, modality)
        go.to_csv('{}.csv'.format(prefix))
        fig.savefig('{}.pdf'.format(prefix))

go_df = pd.concat(go_dfs, ignore_index=True)

modalities_all_celltypes = study.supplemental.modalities_tidy.groupby('event_id').filter(
    lambda x: len(x) == len(study.phenotype_order))

import itertools

background = study.splicing.splicing_to_expression_id(modalities_all_celltypes.event_id)


for (group), df1 in modalities_all_celltypes.groupby(['modality']):
    print group
    foreground = study.splicing.splicing_to_expression_id(df1.event_id)
    go = study.go_enrichment(foreground, background, #p_value_cutoff=0.1, 
#                              min_feature_size=5, min_background_size=10,
                             domain='biological_process')
    if go is None or go.empty:
        print '\t... empty GO enrichment!'
        continue

    fig, ax = plt.subplots(figsize=(2, 2))
    ax = plot_go_enrichment(data=go, color='grey')
    ax.set_title(group)
    fig = plt.gcf()
#     fig.tight_layout()
    prefix = '{}/modality_go_enrichment_across_celltypes_{}.pdf'.format(figure_folder, group)
    go.to_csv('{}.csv'.format(prefix))
    fig.savefig('{}.pdf'.format(prefix))



