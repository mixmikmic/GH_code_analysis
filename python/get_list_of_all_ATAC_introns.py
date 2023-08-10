import pandas as pd
import numpy as np
import os
from Bio import SeqIO

intron_sequences = '/projects/ps-yeolab3/bay001/annotations/hg19.introns.fa'

from Bio import SeqIO
handle = open(intron_sequences, "rU")
atac_sequences = []
for record in SeqIO.parse(handle, "fasta"):
    if record.seq[:2].upper() == 'AT' and record.seq[-2:] == 'AC' and '_f::' in record.name:
        atac_sequences.append(record)
    elif record.seq[:2].upper() == 'GT' and record.seq[-2:] == 'AT' and '_r::' in record.name:
        atac_sequences.append(record)
        
handle.close()

# should print 955
outfile = '/projects/ps-yeolab3/bay001/annotations/hg19.gencode.v19.atac_introns.fa'
SeqIO.write(atac_sequences,outfile,"fasta")

def get_strand(name):
    if name.endswith('_r'):
        return '-'
    elif name.endswith('_f'):
        return '+'

bedfile = '/projects/ps-yeolab3/bay001/annotations/hg19.gencode.v19.atac_introns.bed'

o = open(bedfile, 'w')

for sequence in atac_sequences:
    name, _, chrom, pos = sequence.name.split(':')
    start, end = pos.split('-')
    o.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
        chrom, start, end, name, 0, get_strand(name)
    ))
o.close()

gff = pd.read_table(
    '/projects/ps-yeolab3/bay001/annotations/hg17.u12db.annotations.atac_introns.gff',
    names=['chrom','src','featuretype','start','end','.','strand','.','attr']
)
gff.head()

def gff_to_bed(gff_file):
    df = pd.read_table(
        gff_file, names=['chrom','src','featuretype','start','end','.','strand','.','attr']
    )
    df['score'] = 0
    bed_df = df[['chrom','start','end','attr','score','strand']]
    bed_df['start'] = bed_df['start'] - 1
    return bed_df

hg17_bed = gff_to_bed('/projects/ps-yeolab3/bay001/annotations/hg17.u12db.annotations.atac_introns.gff')
hg17_bed.to_csv('/projects/ps-yeolab3/bay001/annotations/hg17.u12db.annotations.atac_introns.bed', sep='\t', index=False, header=False)

hg19_liftover = pd.read_table(
    '/projects/ps-yeolab3/bay001/annotations/hg19.u12db.annotations.atac_introns.hg17liftover.bed',
    sep='\t', names=['chrom','start','end','name','score','strand']
)
hg19_liftover.head(5)

eric_names = ['intron_type','chrom','strand','low_exon','hi_exon','name']
example_file = '/home/elvannostrand/data/ENCODE/RNAseq/scripts/exon_junction_counts/gencodev19.CIandRIlist.txt'
atac_intron_bedfile = '/projects/ps-yeolab3/bay001/annotations/hg19.gencode.v19.atac_introns.bed'
pd.read_table(example_file, names=eric_names).head()

atac_introns = pd.read_table(
    atac_intron_bedfile, names=['intron_chrom','intron_start','intron_stop','intron_name','intron_score','intron_strand']
) # .drop_duplicates(['intron_chrom','intron_start','intron_stop','intron_strand'])
print("{} introns found.".format(atac_introns.shape[0]))
atac_introns['intron_tx_name'] = atac_introns['intron_name'].apply(lambda x: x.split('_')[0])
atac_introns.head()

# intersect with exons (can be more than one exon per intron)
exons = pd.read_table(
    '/projects/ps-yeolab3/bay001/annotations/gencode.v19.annotations.exons.bed',
    names=['exon_chrom','exon_start','exon_stop','exon_name','exon_score','exon_strand']
) # .drop_duplicates(['exon_chrom','exon_start','exon_stop','exon_strand'])
print("{} exons found.".format(exons.shape[0]))
exons['exon_tx_name'] = exons['exon_name'].apply(lambda x: x.split('_')[0])
exons[exons['exon_tx_name']=='ENST00000465537.1']

# merge on the upstream junction

merged_upstream = pd.merge(
    atac_introns, exons, 
    how='left', 
    left_on=['intron_chrom','intron_start','intron_strand','intron_tx_name'], 
    right_on=['exon_chrom','exon_stop','exon_strand','exon_tx_name']
)
merged_upstream.columns = [
    'intron_chrom','intron_start','intron_stop','intron_name',
    'intron_score','intron_strand','intron_tx_name',
    'upstream_exon_chrom','upstream_exon_start','upstream_exon_stop',
    'upstream_exon_name','upstream_exon_score','upstream_exon_strand',
    'upstream_exon_tx_name'
]
# x = merged_upstream.fillna(0)
# x[x['upstream_exon_start']==0]


# merge on the downstream junctions

merged = pd.merge(
    merged_upstream, exons, 
    how='left',
    left_on=['intron_chrom','intron_stop','intron_strand','intron_tx_name'],
    right_on=['exon_chrom','exon_start','exon_strand','exon_tx_name']
)
merged.head()

merged = merged[['intron_chrom','upstream_exon_start','upstream_exon_stop','exon_start','exon_stop','intron_strand','intron_tx_name']]
merged.head()

def format_upstream_pos(row):
    return '{}-{}'.format(row['upstream_exon_start'], row['upstream_exon_stop'])
def format_downstream_pos(row):
    return '{}-{}'.format(row['exon_start'], row['exon_stop'])

merged['upstream'] = merged.apply(format_upstream_pos, axis=1)
merged['downstream'] = merged.apply(format_downstream_pos, axis=1)
merged['intron_label'] = 'atac_intron'
merged = merged[['intron_label','intron_chrom','intron_strand','upstream','downstream','intron_tx_name']]
merged

merged.to_csv(
    '/projects/ps-yeolab3/bay001/annotations/hg19.gencode.v19.atac_introns.flanking_exons.tab',
    sep='\t', header=False, index=False
)



