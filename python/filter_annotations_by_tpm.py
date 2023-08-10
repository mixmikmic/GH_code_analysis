import pandas as pd
import numpy as np
import os
import glob
from tqdm import tnrange, tqdm_notebook

wd = '/projects/ps-yeolab3/bay001/maps/current_annotations/'
names = ['chrom','start','end','name','score','strand']
hepg2_tpm = pd.read_table(os.path.join(wd,'HepG2_topENSTbyTPM.wnoncoding.csv'))
k562_tpm = pd.read_table(os.path.join(wd,'K562_topENSTbyTPM.wnoncoding.csv'))
hepg2_tpm_1 = hepg2_tpm[hepg2_tpm['ENST_tpm']>=1]
k562_tpm_1 = k562_tpm[k562_tpm['ENST_tpm']>=1]

def filter_and_save(in_file, tpm_df, out_file):
    
    regions = pd.read_table(in_file, names=names)

    new_regions = regions.set_index('name').loc[tpm_df['#ENSG']].dropna().reset_index()
    new_regions = new_regions[['chrom','start','end','name','score','strand']]

    # NaNs introduced as a result of filtering by index convert positions into floats
    new_regions['start'] = new_regions['start'].astype(int)
    new_regions['end'] = new_regions['end'].astype(int)
    new_regions.to_csv(out_file, sep='\t', index=False, header=False)

regions = [
    # 'cds','exons','introns','five_prime_utrs','three_prime_utrs','distintron500','proxintron500','genes',
    'poly_a_sites','transcription_start_sites','start_codons','stop_codons'
]

progress = tnrange(len(regions)*2)
for region in regions:
    in_file = os.path.join(wd,'hg19_v19_{}.bed'.format(region))
    for cell_type in ['HepG2', 'K562']:
        if cell_type == 'HepG2':
            tpm_df = hepg2_tpm_1
            out_file = os.path.join(wd,'hg19_v19_{}.HepG2_tpm1.bed'.format(region))
        elif cell_type == 'K562':
            tpm_df = k562_tpm_1
            out_file = os.path.join(wd,'hg19_v19_{}.K562_tpm1.bed'.format(region))
        filter_and_save(in_file, tpm_df, out_file)
        progress.update(1)



