import pandas as pd
import numpy as np
import os
import glob
from collections import defaultdict

directory = '/projects/ps-yeolab3/bay001/maps/current/se_nr'
images = glob.glob(os.path.join(directory, "*.svg"))

len(images)

images[:5]

def get_reps(line):
    prefix = os.path.basename(line).split('.')[0]
    elements = prefix.split('_')
    if len(elements) != 3:
        uid = elements[0]
        rep = elements[1]
        rbp = elements[3]
    else:
        uid = elements[0]
        rep = elements[1]
        rbp = elements[2]
    return rep

def get_prefix(line):
    prefix = os.path.basename(line).split('.')[0]
    elements = prefix.split('_')
    if len(elements) != 3:
        uid = elements[0]
        rep = elements[1]
        rbp = elements[3]
    else:
        uid = elements[0]
        rep = elements[1]
        rbp = elements[2]
    return uid

def get_rbp(line):
    prefix = os.path.basename(line).split('.')[0]
    elements = prefix.split('_')
    if len(elements) != 3:
        uid = elements[0]
        rep = elements[1]
        rbp = elements[3]
    else:
        uid = elements[0]
        rep = elements[1]
        rbp = elements[2]
    return rbp

prefix = defaultdict(list)
for i in images:
    prefix[get_prefix(i)].append(get_reps(i))
    

# double check this to see if each replicate for each RBP uID was made. 
prefix

date = '6-5-2017'

hepg2_rnaseq_manifest = '/home/bay001/projects/maps_20160420/permanent_data/RNASeq_final_exp_list_HepG2.csv'
k562_rnaseq_manifest = '/home/bay001/projects/maps_20160420/permanent_data/RNASeq_final_exp_list_K562.csv'

hepg2_rnaseq_df = pd.read_table(hepg2_rnaseq_manifest)
k562_rnaseq_df = pd.read_table(k562_rnaseq_manifest)



missing_files = [
    '/projects/ps-yeolab3/bay001/maps/bash_scripts/{}/whole_read-SE_NR_svg.missing.txt'.format(date),
    '/projects/ps-yeolab3/bay001/maps/bash_scripts/{}/SE_PEAK_PNGS.missing.txt'.format(date),
    '/projects/ps-yeolab3/bay001/maps/bash_scripts/{}/IDR_PNGS.missing.txt'.format(date),
    '/projects/ps-yeolab3/bay001/maps/bash_scripts/6-5-2017/a3ss_NR_png.missing.txt',
    '/projects/ps-yeolab3/bay001/maps/bash_scripts/6-5-2017/a5ss_NR_png.missing.txt',
    '/projects/ps-yeolab3/bay001/maps/bash_scripts/6-5-2017/a3ss_NR_svg.missing.txt',
    '/projects/ps-yeolab3/bay001/maps/bash_scripts/6-5-2017/a5ss_NR_svg.missing.txt',
]

for mf in missing_files:
    df = pd.read_table(
        mf,
        names=['rbp','cell']
    )
    k562 = df[df['cell']=='K562']
    hepg2 = df[df['cell']=='HepG2']
    # do inner join, the missing list should not show up in the final expt lists from xintao.
    if len(pd.merge(k562, k562_rnaseq_df, how='inner', left_on=['rbp'], right_on=['Official_RBP'])) != 0:
        print(mf)
    if len(pd.merge(hepg2, hepg2_rnaseq_df, how='inner', left_on=['rbp'], right_on=['Official_RBP'])) != 0:
        print(mf)















