import pandas as pd
import pybedtools

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


folder = '/projects/ps-yeolab/obotvinnik/singlecell_pnms'
csv_folder = '{}/csvs_for_paper/'.format(folder)

splicing_feature_folder = '{}/splicing_feature_data'.format(csv_folder)
alternative_feature_folder = '{}/alternative'.format(splicing_feature_folder)
constitutive_feature_folder = '{}/constitutive'.format(splicing_feature_folder)

alt_exons_bedfile = '{}/exons.bed'.format(alternative_feature_folder)
constitutive_bedfile = '{}/exons.bed'.format(constitutive_feature_folder)
bedfiles = alt_exons_bedfile, constitutive_bedfile

get_ipython().system(" grep 'exon:chr10:102286732-102286831:-@exon:chr10:102286156-102286311:-@exon:chr10:102283497-102283686:-' $alt_exons_bedfile")

get_ipython().system(' wc -l $alt_exons_bedfile')

constitutive_bed = pybedtools.BedTool(constitutive_bedfile)
names = [x.name for x in constitutive_bed]


hg19_fasta = '/projects/ps-yeolab/genomes/hg19/gencode/v19/GRCh37.p13.genome.fa'

constitutive_df = pd.read_table(constitutive_bedfile, header=None)
constitutive_df[2] += 1
constitutive_df.to_csv('{}/exons_stop_plus1.bed'.format(constitutive_feature_folder), sep='\t', index=False, header=False)
print(constitutive_df.shape)
constitutive_df.head()

alt_exons_df = pd.read_table(alt_exons_bedfile, header=None)
alt_exons_df.head()
alt_exons_df[2] += 1
alt_exons_df.to_csv('{}/exons_stop_plus1.bed'.format(alternative_feature_folder), sep='\t', index=False, header=False)
print(alt_exons_df.shape)
alt_exons_df.head()
# alt_exons = pybedtools.BedTool(alt_exons_bedfile)

# exon2_seq = alt_exons.sequence(fi=hg19_fasta)
# exon2_seq

filename = '{}/exons_stop_plus1.bed'.format(alternative_feature_folder)

get_ipython().system(' head $filename')

from Bio import SeqIO
from Bio.SeqUtils import GC

alt_exons = pybedtools.BedTool('{}/exons_stop_plus1.bed'.format(alternative_feature_folder))

exon2_seq = alt_exons.sequence(fi=hg19_fasta)


with open(exon2_seq.seqfn) as f:
    records = SeqIO.parse(f, 'fasta')
    records = pd.Series([str(x.seq) for x in records], index=alt_exons_df[3])
exon2_gc = records.apply(GC)
exon2_gc.name = 'exon2_gc_content'
print(exon2_gc.shape)
exon2_gc.head()

exon2_gc.shape

constitutive_exons = pybedtools.BedTool('{}/exons_stop_plus1.bed'.format(constitutive_feature_folder))
 
constitutive_seq = constitutive_exons.sequence(fi=hg19_fasta)


with open(constitutive_seq.seqfn) as f:
    records = SeqIO.parse(f, 'fasta')
    records = pd.Series([str(x.seq) for x in records], index=constitutive_df[3])
constitutive_gc = records.apply(GC)
constitutive_gc.name = 'gc_content'
print(constitutive_gc.shape)
constitutive_gc.head()

csv = 'gc_content.csv'

constitutive_gc.to_csv('{}/{}'.format(constitutive_feature_folder, csv))

exon2_gc.to_csv('{}/{}'.format(alternative_feature_folder, csv))

# if constitutive_gc.name not in constitutive_feature_data:
#     constitutive_feature_data = constitutive_feature_data.join(constitutive_gc)
# constitutive_feature_data.to_csv('{}/constitutive_feature_data.csv'.format(csv_folder))

# if exon2_gc.name not in splicing_feature_data:
#     splicing_feature_data = splicing_feature_data.join(exon2_gc)

# splicing_feature_data.to_csv('{}/splicing_feature_data.csv'.format(csv_folder))



