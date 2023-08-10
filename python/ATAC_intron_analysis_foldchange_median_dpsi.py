get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
from tqdm import tnrange, tqdm_notebook

import matplotlib
matplotlib.rcParams.update({'font.size': 18})
sns.set_style("whitegrid")

# Get all knockdown expts and controls
manifest_directory = '/projects/ps-yeolab3/encode/'
k562 = pd.read_table(os.path.join(
    manifest_directory,
    'k562_brenton-graveley_ambiguous_bams_for_integrated_analysis.txt'
), index_col=0)
hepg2 = pd.read_table(os.path.join(
    manifest_directory,
    'hepg2_brenton-graveley_ambiguous_bams_for_integrated_analysis.txt'
), index_col=0)

manifest = pd.concat([k562, hepg2])
manifest.head()

def get_avg(row):
    n1 = row['dpsi_x']
    n2 = row['dpsi_y']
    if np.isnan(n1):
        return n2
    elif np.isnan(n2):
        return n1
    else:
        return (n1 + n2) / 2.0

def get_psi(fn):
    """
    returns the percent spliced in (inc/(inc+exc)) for a dataset.
    """
    names = ['exc_count', 'inc_count', 'dpsi']
    df = pd.read_table(fn, sep='\t', index_col=0, names=names).drop_duplicates()
    sum_exc = df['exc_count'].sum()
    sum_inc = df['inc_count'].sum()
    psi = sum_inc / float(sum_exc + sum_inc)
    return psi

def get_fold_enrichment(kd, ctrl):
    """
    Given two datasets, calculate the fold enrichment 
    """
    kd_psi = get_psi(kd)
    ctrl_psi = get_psi(ctrl)
    return kd_psi/ctrl_psi

def get_all_required_files(expt_id, jxc_dir, df=manifest):
    """
    For each experiment ID, return: distributions of 
    the average dpsi vals for both reps for each condition
    """
    datapoints = []
    
    sub = df.ix[expt_id]
    ciri_prefix = '.primary.namesort.bam.CIandRI.psi'
    atac_prefix = '.primary.namesort.bam.ATAC.psi'
    
    ciri_expt_files = [
        os.path.join(jxc_dir, (sub['expt_rep1'] + ciri_prefix)), 
        os.path.join(jxc_dir, (sub['expt_rep2'] + ciri_prefix))
    ]
    ciri_ctrl_files = [
        os.path.join(jxc_dir, (sub['control_rep1'] + ciri_prefix)), 
        os.path.join(jxc_dir, (sub['control_rep2'] + ciri_prefix))
    ]
    
    atac_expt_files = [
        os.path.join(jxc_dir, (sub['expt_rep1'] + atac_prefix)), 
        os.path.join(jxc_dir, (sub['expt_rep2'] + atac_prefix))
    ]
    atac_ctrl_files = [
        os.path.join(jxc_dir, (sub['control_rep1'] + atac_prefix)), 
        os.path.join(jxc_dir, (sub['control_rep2'] + atac_prefix))
    ]
    try:
        ciri_r1_foldenr = get_fold_enrichment(ciri_expt_files[0], ciri_ctrl_files[0])
        ciri_r2_foldenr = get_fold_enrichment(ciri_expt_files[1], ciri_ctrl_files[1])
        atac_r1_foldenr = get_fold_enrichment(atac_expt_files[0], atac_ctrl_files[0])
        atac_r2_foldenr = get_fold_enrichment(atac_expt_files[1], atac_ctrl_files[1])
    
        datapoints.append([ciri_r1_foldenr,atac_r1_foldenr])
        datapoints.append([ciri_r2_foldenr,atac_r2_foldenr])
    except KeyError:
        return []
    return datapoints

jxc_dir = '/home/bay001/projects/encode/analysis/atac_intron_analysis/jxc_from_eric'

progress = tnrange(len(k562.index))
for index in k562.index:
    rbp = k562.ix[index]['name']
    datapoints = get_all_required_files(index,jxc_dir)
    print(rbp, datapoints)
    progress.update(1)

get_psi('/home/bay001/projects/encode/analysis/atac_intron_analysis/jxc_from_eric/ENCFF074LFN.bam.primary.namesort.bam.CIandRI.psi')

pd.read_table("/home/bay001/projects/encode/analysis/atac_intron_analysis/jxc_from_eric/ENCFF074LFN.bam.primary.namesort.bam.CIandRI.psi",
             names=['exc_count', 'inc_count', 'dpsi'], index_col=0).drop_duplicates()['exc_count'].sum()

pd.read_table("/home/bay001/projects/encode/analysis/atac_intron_analysis/jxc_from_eric/ENCFF074LFN.bam.primary.namesort.bam.CIandRI.psi",
             names=['exc_count', 'inc_count', 'dpsi'], index_col=0).drop_duplicates()['inc_count'].sum()



