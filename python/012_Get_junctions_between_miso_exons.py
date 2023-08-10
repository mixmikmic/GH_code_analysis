from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


import flotilla
sns.set(style='ticks', context='talk')

folder = 'figures'

flotilla_dir = '/projects/ps-yeolab/obotvinnik/flotilla_projects/'

study = flotilla.embark('singlecell_pnm_figure1_supplementary_post_splicing_filtering', flotilla_dir=flotilla_dir)

figure_folder = 'figures/004_Comparison_to_miso'
get_ipython().system(' mkdir -p $figure_folder')

from __future__ import print_function

study.splicing.maybe_renamed_to_feature_id("PKM")

# miso_psi_filename = '/projects/ps-yeolab/obotvinnik/flotilla_projects/singlecell_pnms/splicing.csv.gz'
# miso_psi = pd.read_csv(miso_psi_filename, index_col=0, compression='gzip')
# print(miso_psi.shape)
# miso_psi.head()

# miso_psi_tidy = miso_psi.unstack().reset_index()
# miso_psi_tidy = miso_psi_tidy.rename(columns={'level_0':'miso_id', 'level_1':'sample_id', 0:'miso_psi'})
# print(miso_psi_tidy.shape)
# miso_psi_tidy = miso_psi_tidy.dropna()
# print(miso_psi_tidy.shape)
# miso_psi_tidy.head()

# miso_psi_tidy['n_exons'] = miso_psi_tidy['miso_id'].map(lambda x: len(x.split('@')))
# miso_psi_tidy.head()

# miso_psi_tidy = miso_psi_tidy.loc[miso_psi_tidy['n_exons'] >= 3]
# miso_psi_tidy.shape

# junction_reads = pd.read_csv('/projects/ps-yeolab/obotvinnik/singlecell_pnms/csvs_for_paper/junction_reads_use_multimapping.csv')
# print(junction_reads.shape)
# junction_reads.head()

miso_exons_names = pd.read_csv('/projects/ps-yeolab/obotvinnik/singlecell_pnms/csvs_for_paper/miso_exons_names.csv', index_col=0)
print(miso_exons_names.shape)
miso_exons_names.head()

from outrigger.region import Region

get_ipython().run_cell_magic('time', '', 'miso_exons = miso_exons_names.applymap(lambda x: Region(x) if not isinstance(x, float) else x)\nprint(miso_exons.shape)\n# miso_exons.head()')

print(miso_exons.shape)
miso_exons.head()

get_ipython().run_line_magic('pdb', '')

import itertools

FIRST_EXON = ['exon_1']

SE_MIDDLE_EXON = ['exon_2']
SE_LAST_EXON = ['exon_3']
SE_NOT_FIRST_EXONS = SE_MIDDLE_EXON + SE_LAST_EXON

MXE_MIDDLE_EXONS = ['exon_2', 'exon_3']
MXE_LAST_EXON = ['exon_4']

def make_junction_regions(exons):
    chrom = exons.iloc[0].chrom
    strand = exons.iloc[0].strand
    junctions = {}

    iterator = itertools.chain(itertools.combinations(exons.iteritems(), 2))
    
    for ((exon_i, exon_i_region), (exon_j, exon_j_region)) in iterator:
        i = exon_i.split('_')[-1]
        j = exon_j.split('_')[-1]
        name = 'junction_{i}{j}'.format(i=i, j=j)
        try:            
            if strand == '+':
                start = exon_i_region.stop + 1
                stop = exon_j_region.start - 1
            else:
                start = exon_j_region.stop + 1
                stop = exon_i_region.start - 1
        except AttributeError:
            # Got to NA exon
            continue

        location = 'junction:{chrom}:{start}-{stop}:{strand}'.format(chrom=chrom, start=start, stop=stop, strand=strand)

        try:
            junction = Region(location)
        except ValueError:
            junction = 'Invalid'
        junctions[name] = junction

    return pd.Series(junctions)


get_ipython().run_line_magic('time', 'miso_junction_regions = miso_exons.tail(100).apply(make_junction_regions, axis=1)')
print(miso_junction_regions.shape)
miso_junction_regions.head()

get_ipython().run_line_magic('time', 'miso_junction_regions = miso_exons.apply(make_junction_regions, axis=1)')
print(miso_junction_regions.shape)
miso_junction_regions.head()

def mxe_bad_exon2_exon3(exons):
    if pd.isnull(exons[MXE_LAST_EXON]).any():
        return False
    
    strand = exons.iloc[0].strand
    
    if strand == '+':
        start = exons['exon_2'].stop + 1
        stop = exons['exon_3'].start - 1
    else:
        start = exons['exon_3'].stop + 1
        stop = exons['exon_2'].start - 1
        
    if start > stop:
        return True
    return False
    

get_ipython().run_line_magic('time', 'invalid_mxe_events = miso_exons.apply(mxe_bad_exon2_exon3, axis=1)')

invalid_mxe_events.to_csv('/projects/ps-yeolab/obotvinnik/singlecell_pnms/csvs_for_paper/miso_junctions_invalid_mxe.csv', 
                          header=True)

invalid_mxe_events[invalid_mxe_events]

get_ipython().run_line_magic('time', 'miso_junctions = miso_junction_regions.applymap(lambda x: x.name if isinstance(x, Region) else x)')
print(miso_junctions.shape)
miso_junctions.head()

miso_junctions.to_csv('/projects/ps-yeolab/obotvinnik/singlecell_pnms/csvs_for_paper/miso_junctions_names.csv')



