get_ipython().run_cell_magic('time', '', "from __future__ import print_function\n\nimport matplotlib as mpl\nimport pandas as pd\nimport numpy as np\nimport seaborn as sns\nimport six\n%matplotlib inline\n\nimport flotilla\n\n\nfolder = '/projects/ps-yeolab/obotvinnik/singlecell_pnms'\n\ncsv_folder = '{}/csvs_for_paper'.format(folder)\noutrigger_rmdup_folder = '{}/outrigger_v2_bam_stranded'.format(csv_folder)\n\nmetadata = pd.read_csv('{}/metadata.csv'.format(csv_folder), index_col=0)\nexpression = pd.read_csv('{}/expression.csv'.format(csv_folder), index_col=0)\nmapping_stats = pd.read_csv('{}/mapping_stats.csv'.format(csv_folder), index_col=0)\n\nsplicing = pd.read_csv('{}/psi.csv'.format(csv_folder), index_col=0)\n\n\nprint('metadata.shape', metadata.shape)\nprint('expression.shape', expression.shape)\nprint('mapping_stats.shape', mapping_stats.shape)\nprint('splicing.shape', splicing.shape)")

get_ipython().system(' ls -lha $csv_folder/psi.csv')

# psi_rmdup = pd.read_csv('{}/psi/outrigger_psi.csv'.format(outrigger_rmdup_folder))

outrigger_folder = '/projects/ps-yeolab/obotvinnik/singlecell_pnms/outrigger_v2'

mutually_exclusive_exon_feature_data = pd.read_csv('{}/index/mxe/manual_metadata.csv'.format(outrigger_folder), index_col=0)
print(mutually_exclusive_exon_feature_data.shape)
mutually_exclusive_exon_feature_data.head()

skipped_exon_feature_data = pd.read_csv('{}/index/se/manual_metadata.csv'.format(outrigger_folder), index_col=0)
print(skipped_exon_feature_data.shape)
skipped_exon_feature_data.head()

splicing_feature_data = pd.concat([skipped_exon_feature_data, mutually_exclusive_exon_feature_data])
print(splicing_feature_data.shape)
splicing_feature_data.head()

splicing_feature_data_subset = splicing_feature_data.loc[splicing.columns]
print(splicing_feature_data_subset.shape)
splicing_feature_data_subset.head()

splicing_feature_data_subset = splicing_feature_data_subset.drop_duplicates()
print(splicing_feature_data_subset.shape)
splicing_feature_data_subset.head()

skipped_exon_feature_data.columns

splicing.head()

splicing.index = splicing.index.map(lambda x: '_'.join(x.split('_')[:2]))
splicing.head()

import re

pattern = 'exon:chr[0-9A-Z]+:\d+-\d+:[-=]'

splicing_feature_data['alternative_exons'] = splicing_feature_data.index.map(lambda x: '\n'.join(re.findall(pattern, x)))
splicing_feature_data['shortened_id'] = splicing_feature_data['alternative_exons'] + '\n' + splicing_feature_data['event_location']
splicing_feature_data.head()

pkm_event = 'isoform1=junction:chr15:72494962-72499068:-@exon:chr15:72494795-72494961:-@junction:chr15:72492997-72494794:-|isoform2=junction:chr15:72495530-72499068:-@exon:chr15:72495363-72495529:-@junction:chr15:72492997-72495362:-'

splicing_feature_data.query('transcript_name == "SNAP25"').index.unique()

splicing_feature_data.loc[pkm_event, 'alternative_exons']

splicing_feature_data.loc[pkm_event, 'event_location']

pkm_shortened = splicing_feature_data.loc[pkm_event, 'shortened_id'].iloc[0]

gene_ontology = pd.read_csv('/projects/ps-yeolab/obotvinnik/flotilla_projects/hg19/gene_ontology.csv', index_col=0)
print(gene_ontology.shape)
gene_ontology.head()

get_ipython().run_line_magic('matplotlib', 'inline')

greens = map(mpl.colors.rgb2hex, sns.color_palette('Greens', n_colors=3))
sns.palplot(greens)

lightgreen, mediumgreen, darkgreen = greens

phenotype_to_color = {'MN': greens[2],
                      'NPC': greens[1],
                      'iPSC': greens[0]}
phenotype_to_marker = {'MN': 's',    # square
                       'NPC': '^',   # Triangle
                       'iPSC': 'o'}  # circle
metadata_phenotype_order = ('iPSC', 'NPC', 'MN')

min_samples = 10

study = flotilla.Study(metadata, expression_data=expression, expression_log_base=2, 
                       expression_plus_one=True,
                       expression_thresh=1,
                       splicing_data=splicing,
#                        splicing_feature_data=splicing_feature_data,
#                        splicing_feature_rename_col='gene_name',
#                        splicing_feature_expression_id_col='ensembl_id',
                       # At least 10 samples per feature (either gene or splicing event)
                       splicing_feature_shortener_col='shortened_id',
                       metadata_minimum_samples=min_samples, 
                       mapping_stats_data=mapping_stats,
                       mapping_stats_min_reads=1e6,
                       mapping_stats_number_mapped_col='Uniquely mapped reads number',
                       metadata_phenotype_order=('iPSC', 'NPC', 'MN'), 
                       metadata_phenotype_to_marker=phenotype_to_marker,
                       metadata_phenotype_to_color=phenotype_to_color,
                       species='hg19')

# Set the curated splicing feature data separately because setting species as 'hg19' overrides
study.splicing.feature_data = splicing_feature_data

study.plot_gene('RBFOX2')

study.splicing.maybe_renamed_to_feature_id('PKM')

study.plot_event("PKM")
# fig = plt.gcf()

# ax = plt.gca()

import matplotlib.pyplot as plt

study.plot_event(pkm_event)
ax = plt.gca()
ax.set_title('PKM')
fig = plt.gcf()
fig.tight_layout()

ax.text(0.5, -0.4, u'\n'.join(pkm_event.split('@')), transform=ax.transAxes, 
        horizontalalignment='center', verticalalignment='top')



study.plot_event(pkm_event)
ax = plt.gca()
ax.set_title('PKM')
fig = plt.gcf()
fig.tight_layout()

ax.text(0.5, -0.4, u'\n'.join(pkm_event.split('@')), transform=ax.transAxes, 
        horizontalalignment='center', verticalalignment='top')

flotilla_dir = '/projects/ps-yeolab/obotvinnik/flotilla_projects'
study.save('singlecell_pnm', flotilla_dir=flotilla_dir)



