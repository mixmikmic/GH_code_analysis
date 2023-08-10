get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set(style='ticks', context='talk', rc={'font.sans-serif':'Arial', 'pdf.fonttype': 42})

get_ipython().run_line_magic('matplotlib', 'inline')


import flotilla
flotilla_dir = '/projects/ps-yeolab/obotvinnik/flotilla_projects'

study = flotilla.embark('singlecell_pnm_figure2_modalities_bayesian', flotilla_dir=flotilla_dir)
not_outliers = study.splicing.singles.index.difference(study.splicing.outliers.index)

psi = study.splicing.singles.ix[not_outliers]
grouped = psi.groupby(study.sample_id_to_phenotype)
psi_filtered = grouped.apply(lambda x: x.dropna(axis=1, thresh=20))

folder = '/home/obotvinnik/Dropbox/figures2/singlecell_pnm/figure2_modalities/bayesian'
get_ipython().system('mkdir $folder')

figure_folder = '{}/kmer_counting'.format(folder)
get_ipython().system(' mkdir -p $figure_folder')

study.supplemental.modalities_tidy.groupby(['phenotype', 'modality']).size()



bed_folder = '/projects/ps-yeolab/obotvinnik/singlecell_pnms/figure2_modalities/bayesian'
get_ipython().system(' mkdir $bed_folder')
import pybedtools

import pyhomer

all_events = study.supplemental.modalities_tidy.event_id.unique()

from outrigger.region import Region

exon2s = map(lambda x: x.split('@')[1], all_events)
exon2_regions = map(Region, exon2s)
exon2_bed_table = pd.DataFrame(map(lambda x: [x.chrom, x._start, x._stop, x.name, 1000, x.strand], exon2_regions))
exon2_bed_table[3] = all_events
exon2_bed_table.head()

exon2_bed_table = exon2_bed_table.sort_values(by=3)
exon2_bed_table.head()

exon2_bed_table.to_csv('{}/background_events.bed'.format(bed_folder), index=False, header=False, sep='\t')

from Bio import SeqIO
import kvector
import pybedtools

DIRECTIONS = 'upstream', 'downstream'



DIR = '/projects/ps-yeolab/obotvinnik/singlecell_pnms'

exon_bedfile = '{}/background_events.bed'.format(bed_folder)
exon_bed = pybedtools.BedTool(exon_bedfile)

commands = []

findMotifsGenome = '/home/yeo-lab/software/homer/bin/findMotifsGenome.pl'
n_processors = 4
homer_flags = '-rna -len 4,5,6 -mset vertebrates -mis 0 -p {} -noweight'.format(n_processors)


primate_filename = '/projects/ps-yeolab/genomes/hg19/database/phastConsElements46wayPrimates.bed'
primate = pybedtools.BedTool(primate_filename)
placental_filename = '/projects/ps-yeolab/genomes/hg19/database/phastConsElements46wayPlacental.bed'
placental = pybedtools.BedTool(placental_filename)
conserved_regions = {'primate': primate, 'placental': placental}

conservation_bed = placental

genome = 'hg19'

nt = 400


genome_fasta = '/projects/ps-yeolab/genomes/hg19/chromosomes/all.fa'

kmer_zscores = []

for phenotype, phenotype_df in study.supplemental.modalities_tidy.groupby('phenotype'):
    background_events = set(phenotype_df.event_id)
    for modality, modality_df in phenotype_df.groupby('modality'):
        print '---\n', phenotype, modality
        event_ids = set(modality_df.event_id)
        format_args = bed_folder, phenotype, modality
        foreground_table = exon2_bed_table.loc[exon2_bed_table[3].isin(event_ids)]
        foreground_filename = '{}/exon2_{}_{}_foreground.bed'.format(*format_args)
        foreground_table.to_csv(foreground_filename, index=False, header=False, sep='\t')
        foreground = pybedtools.BedTool(foreground_filename)
        
        
        background_table = exon2_bed_table.loc[exon2_bed_table[3].isin(background_events)]
        background_filename = '{}/exon2_{}_{}_background.bed'.format(*format_args)
        background_table.to_csv(background_filename, index=False, header=False, sep='\t')
        background = pybedtools.BedTool(background_filename)
        
        
        pair = pyhomer.ForegroundBackgroundPair(foreground, background)
#         print '\n', pair 
        
        for direction in DIRECTIONS:
            print '\n\t', direction
            intron_pair = pair.flanking_intron(direction, 'hg19', 400)

            conserved_introns = intron_pair.intersect(conservation_bed, 'placental')
#             print '\n', conserved_introns
            get_ipython().run_line_magic('time', 'seqs = conserved_introns.foreground.sequence(fi=genome_fasta, s=True)')
            get_ipython().run_line_magic('time', 'foreground_kmers = kvector.count_kmers(seqs.seqfn)')

            get_ipython().run_line_magic('time', 'seqs = conserved_introns.background.sequence(fi=genome_fasta, s=True)')
            get_ipython().run_line_magic('time', 'background_kmers = kvector.count_kmers(seqs.seqfn)')
            kmer_zscore = (foreground_kmers.mean() - background_kmers.mean())/background_kmers.std()
            kmer_zscore.name = '{}{}nt_{}_{}_placental'.format(direction, nt, phenotype, modality)
            kmer_zscores.append(kmer_zscore)
kmer_zscores_all = pd.concat(kmer_zscores, axis=1)
kmer_zscores_all.head()

kmer_zscores_all = pd.concat(kmer_zscores, axis=1)
print kmer_zscores_all.shape
kmer_zscores_all = kmer_zscores_all.dropna(axis=1, how='all').dropna(how='all', axis=0)
print kmer_zscores_all.shape

# Replace remaining NAs with zero since they aren't enriched
kmer_zscores_all = kmer_zscores_all.fillna(0)
kmer_zscores_all.head()

study.supplemental.kmer_zscores = kmer_zscores_all

kmer_zscores_metadata = pd.DataFrame.from_records(
    study.supplemental.kmer_zscores.columns.map(lambda x: pd.Series(x.split('_'))), 
    index=study.supplemental.kmer_zscores.columns)
kmer_zscores_metadata.columns = ['direction', 'phenotype', 'modality', 'clade']
study.supplemental.kmer_zscores_metadata = kmer_zscores_metadata

kmer_zscores_metadata.head()

study.save('singlecell_pnm_figure2_modalities_bayesian_kmers', flotilla_dir=flotilla_dir)

