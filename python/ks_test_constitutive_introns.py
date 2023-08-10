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
    
def get_avg_dpsi(files):
    names = ['exc_count', 'inc_count', 'dpsi']
    df1 = pd.read_table(files[0], sep='\t', index_col=0, names=names).drop_duplicates()
    df2 = pd.read_table(files[1], sep='\t', index_col=0, names=names).drop_duplicates()
    merged = pd.merge(df1, df2, how='outer', left_index=True, right_index=True)
    merged['avg'] = merged.apply(get_avg, axis=1)
    return merged['avg']

def get_all_required_files(expt_id, jxc_dir, df=manifest):
    """
    For each experiment ID, return: distributions of 
    the average dpsi vals for both reps for each condition
    """
    sub = df.ix[expt_id]
    prefix = '.primary.namesort.bam.CIandRI.psi'
    expt_files = [
        os.path.join(jxc_dir, (sub['expt_rep1'] + prefix)), 
        os.path.join(jxc_dir, (sub['expt_rep2'] + prefix))
    ]
    ctrl_files = [
        os.path.join(jxc_dir, (sub['control_rep1'] + prefix)), 
        os.path.join(jxc_dir, (sub['control_rep2'] + prefix))
    ]
    kd_avg_dpsi = get_avg_dpsi(expt_files)
    ctrl_avg_dpsi = get_avg_dpsi(ctrl_files)
    return kd_avg_dpsi, ctrl_avg_dpsi
    
#eCDF

colors = sns.color_palette("hls", 8)

def plot_cdf(expt, ctrl, output_file):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5,7.5))

    num_bins = 100
    n, bins, patches = plt.hist(
        ctrl, 
        normed=True, 
        cumulative=True, 
        histtype='step', 
        color=colors[0], 
        bins=num_bins, 
        alpha=0
    )
    ax.plot(bins[1:], n, color=colors[0], label='ctrl')
    #eCDF

    num_bins = 100
    n, bins, patches = plt.hist(
        expt, 
        normed=True, 
        cumulative=True, 
        histtype='step', 
        color=colors[5], 
        bins=num_bins, 
        alpha=0
    )
    ax.plot(bins[1:], n, color=colors[5], label='expt')
    ax.legend()
    fig.savefig(out_file)

out_file = '/home/bay001/projects/encode/analysis/atac_intron_analysis/ks_test_CI.txt'
out_dir = '/home/bay001/projects/encode/analysis/atac_intron_analysis/cdf_plots_CI/'
jxc_dir = '/home/bay001/projects/encode/analysis/atac_intron_analysis/jxc_from_eric'
o = open(out_file, 'w')
progress = tnrange(len(k562.index))
for index in k562.index:
    
    rbp = k562.ix[index]['name']
    kd, ctrl = get_all_required_files(index,jxc_dir)
    stat, p = stats.ks_2samp(kd, ctrl)
    out_file = os.path.join(out_dir, '{}_{}_{}.png'.format(index, rbp, 'k562'))
    o.write('{}\t{}\t{}\t{}\t{}\n'.format(index, rbp, 'k562', stat, p))
    plot_cdf(kd, ctrl, out_file)
    progress.update(1)

progress = tnrange(len(hepg2.index))
for index in hepg2.index:
    try:
        rbp = hepg2.ix[index]['name']
        kd, ctrl = get_all_required_files(index,jxc_dir)
        stat, p = stats.ks_2samp(kd, ctrl)
        out_file = os.path.join(out_dir, '{}_{}_{}.png'.format(index, rbp, 'hepg2'))
        o.write('{}\t{}\t{}\t{}\t{}\n'.format(index, rbp, 'hepg2', stat, p))
        plot_cdf(kd, ctrl, out_file)
    except IOError:
        print(index)
        pass
    progress.update(1)
o.close()

# check individually some of them

kd, ctrl = get_all_required_files('ENCSR137HKS',jxc_dir)
plot_cdf(kd, ctrl, '~/temp.png')

plt.hist(kd)

plt.hist(ctrl)

df = pd.read_table('/home/bay001/projects/encode/analysis/atac_intron_analysis/20170505.ALLENCODEinclnotsubmitted.txt.nopipes.txt')

df.ix['RNU11'].sort_values(by='fold-enrichment', ascending=False).head()

rbp = 'DKC1'
df[df['element'].str.contains(rbp)].ix['RNU11'] # .sort_values(by='fold-enrichment')

df[df['element'].str.contains(rbp)].ix['RNU1'] # .sort_values(by='fold-enrichment')



dx = pd.read_table('/home/bay001/projects/encode/analysis/atac_intron_analysis/ks_test.txt', names=[
    'enc','rbp','cell','statistic','p'
])
dx.sort_values(by='p')





