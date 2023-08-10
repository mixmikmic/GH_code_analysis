import pandas as pd

se_events = pd.read_csv('/home/obotvinnik/projects/singlecell_pnms/analysis/outrigger_v2/index/se/events.csv', index_col=0)
print(se_events.shape)
se_events.head(2)

non_isoform_cols = [x for x in se_events.columns if 'isoform' not in x]
se_events_non_isoform = se_events[non_isoform_cols]
print(se_events_non_isoform.shape)
se_events_non_isoform.head()

se_gencode_filename = '/home/obotvinnik/projects/singlecell_pnms/analysis/outrigger_v2/index/se/event.sorted.gencode.v19.bed'
se_gencode = pd.read_table(se_gencode_filename, header=None)
se_gencode.head()

def split_gtf_attributes(attributes):
    split = attributes.split('; ')
    pairs = [x.split(' ') for x in split]
    no_quotes = [map(lambda x: x.strip('";'), pair) for pair in pairs]
    mapping = dict(no_quotes)
    return mapping

get_ipython().run_cell_magic('time', '', '\nattributes = se_gencode[14].map(split_gtf_attributes).apply(pd.Series)\nprint(attributes.shape)\nattributes.head()')

attributes.index = se_gencode[3]
attributes.index.name = 'event_id'
attributes.head()

attributes['ensembl_id'] = attributes['gene_id'].str.split('.').str[0]
attributes.head()

attributes_grouped = attributes.groupby(level=0, axis=0).apply(lambda df: df.apply(
        lambda x: ','.join(map(str, set(x.dropna().values)))))
print(attributes_grouped.shape)
attributes_grouped.head(2)

se_metadata = se_events_exons.join(attributes_grouped)
print(se_metadata.shape)
se_metadata.head()

se_metadata.query('gene_name == "RPS24"').index.unique()

se_metadata.query('gene_name == "RBFOX2"').index.unique()

s = se_metadata['gene_name'].str.contains('SNAP25').dropna()

se_metadata.loc[s[s].index.unique()]

se_metadata['gene_name'].isnull().sum()

se_metadata.to_csv('/projects/ps-yeolab/obotvinnik/singlecell_pnms/outrigger_v2/index/se/events_with_metadata.csv')



