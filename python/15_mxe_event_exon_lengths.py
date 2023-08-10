import pandas as pd

mxe_events = pd.read_csv('/home/obotvinnik/projects/singlecell_pnms/analysis/outrigger_v2/index/mxe/events.csv', index_col=0)
print(mxe_events.shape)
mxe_events.head(2)

# mxe_events_not_duplicated = mxe_events.drop_duplicates()
# print(mxe_events_not_duplicated.shape)
# mxe_events_not_duplicated.head(2)

# ! head -n 5 /home/obotvinnik/projects/singlecell_pnms/analysis/outrigger_v2/index/mxe/exon*.bed



from outrigger.region import Region

mxe_exon_bed_template = '/home/obotvinnik/projects/singlecell_pnms/analysis/outrigger_v2/index/mxe/exon{}.bed'

exons = {}

for i in range(1, 5):
    exon = 'exon{}'.format(i)
    bed = mxe_exon_bed_template.format(i)
    exon_df = pd.read_table(bed, names=['chrom', 'start', 'stop', 'name', 'score', 'strand'])
    exon_regions = exon_df.apply(lambda row: Region('exon:{chrom}:{start}-{stop}:{strand}'.format(
                chrom=row.chrom, start=row.start+1, stop=row.stop, strand=row.strand)), axis=1)
    exons[exon] = exon_regions
exon_regions_df = pd.DataFrame(exons)
exon_regions_df.head()

exon_regions_df.shape

exon_regions_df.index = mxe_events.index
print(exon_regions_df.shape)
exon_regions_df.head(2)

def make_introns(row):
    exon1 = row.iloc[0]
    exonN = row.iloc[-1]
    
    if exon1.strand == "+":
        start = exon1.stop
        stop = exonN.start
    elif exon1.strand == '-':
        start = exonN.stop
        stop = exon1.start
    return Region('intron:{chrom}:{start}-{stop}:{strand}'.format(
                chrom=exon1.chrom, start=start, stop=stop, strand=exon1.strand))

def event_location(row):
    exon1 = row.iloc[0]
    exonN = row.iloc[-1]
    
    if exon1.strand == "+":
        start = exon1.start
        stop = exonN.stop
    elif exon1.strand == '-':
        start = exonN.start
        stop = exon1.stop
    return Region('event:{chrom}:{start}-{stop}:{strand}'.format(
                chrom=exon1.chrom, start=start, stop=stop, strand=exon1.strand))


exon_regions_df['intron_location'] = exon_regions_df.apply(make_introns, axis=1)
exon_regions_df['event_location'] = exon_regions_df.apply(event_location, axis=1)
exon_regions_df.head(2)

lengths = exon_regions_df.applymap(len)
lengths.columns = [x.split('_')[0] + '_length' for x in lengths.columns]
print(lengths.shape)
lengths.head()

exon_str = exon_regions_df.applymap(lambda x: x.name)
exon_str.head()

print(exon_str.shape)

event_location.shape

mxe_events_exons = pd.concat([mxe_events, exon_str], axis=1)
print(mxe_events_exons.shape)
mxe_events_exons.head()

mxe_gencode_filename = '/home/obotvinnik/projects/singlecell_pnms/analysis/outrigger_v2/index/mxe/event.sorted.gencode.v19.bed'
mxe_gencode = pd.read_table(mxe_gencode_filename, header=None)
mxe_gencode.head()

def split_gtf_attributes(attributes):
    split = attributes.split('; ')
    pairs = [x.split(' ') for x in split]
    no_quotes = [map(lambda x: x.strip('";'), pair) for pair in pairs]
    mapping = dict(no_quotes)
    return mapping

get_ipython().run_cell_magic('time', '', '\nattributes = mxe_gencode[14].map(split_gtf_attributes).apply(pd.Series)\nprint(attributes.shape)\nattributes.head()')

attributes.index = mxe_gencode[3]
attributes.index.name = 'event_id'
attributes.head()

attributes['ensembl_id'] = attributes['gene_id'].str.split('.').str[0]
attributes.head()

attributes_grouped = attributes.groupby(level=0, axis=0).apply(lambda df: df.apply(
        lambda x: ','.join(map(str, set(x.dropna().values)))))
print(attributes_grouped.shape)
attributes_grouped.head(2)

mxe_metadata = mxe_events_exons.join(attributes_grouped)
print(mxe_metadata.shape)
mxe_metadata.head()

mxe_metadata.query('gene_name == "PKM"').index.unique()

s = mxe_metadata['gene_name'].str.contains('SNAP25').dropna()

mxe_metadata.loc[s[s].index.unique()]

mxe_metadata['gene_name'].isnull().sum()

mxe_metadata.to_csv('/projects/ps-yeolab/obotvinnik/singlecell_pnms/outrigger_v2/index/mxe/events_with_metadata.csv')



