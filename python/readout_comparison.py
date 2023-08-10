from  __future__ import division
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
# install matplotlib venn via: pip install matplotlib_venn
from matplotlib_venn import venn3, venn3_circles
get_ipython().magic('matplotlib inline')

# the bin_dat pickle file, outputed by the adnet analysis script
bin_dat_fn = '../example/example_output/main_full/bindat'
# exported from the matlab dremi gui
dremi_fn = './additional_data/20160314_dremi_values_all_overexpressions.csv'
# a dictionary with the old and new names
name_dict = '../example/name_dict.csv'
out_folder = '../example/example_output/main_full/'
neg_ctrl_names = ['empty-1','empty-2','GFP-FLAG-1','GFP-FLAG-2']
# neg_ctrl_names = ['empty-1','empty-2']

# markers that are excluded from the analysis 
excluded_names = ['cleaved PARP-cleaved caspase3', 'cyclin B1', 'p-4EBP1', 'p-HH3', 'p-RB', 'beads']
#excluded_names =[]

bin_dat = pd.read_pickle(bin_dat_fn)
bin_dat.index.get_level_values('target').unique()

name_dict = pd.read_csv(name_dict)
name_dict = {row['old']: row['new'] for idx, row in name_dict.iterrows()}

dat_dremi = pd.read_csv(dremi_fn, sep=',',index_col=False)
dat_dremi.head()

name_dict_dremi = dict((filter(str.isalnum, oldname.lower()), nicename) for oldname, nicename in name_dict.iteritems())

dat_dremi['target'] = dat_dremi['target'].map(lambda x: name_dict_dremi[x])

dat_dremi_stacked = dat_dremi.copy()
dat_dremi_stacked['origin'] = dat_dremi_stacked['origin'].map(lambda x: x.upper())
dat_dremi_stacked['origin'].map(lambda x: x.upper())
dat_dremi_stacked = dat_dremi_stacked.set_index(['origin','target'])

dat_dremi_stacked =  pd.DataFrame(dat_dremi_stacked.stack(), columns=['dremi'])
dat_dremi_stacked.index.names = ['origin', 'target', 'filename']

# extract experiment and row-col from the filename

get_experiment = lambda x: x.split('/')[5]
get_rowcol = lambda x: x.split('/')[5]
dat_dremi_stacked['experiment'] = [x.split('/')[5] for x in dat_dremi_stacked.index.get_level_values('filename')]
dat_dremi_stacked['row_col'] = [x.split('_')[1] for x in dat_dremi_stacked.index.get_level_values('filename')]

dat_dremi_stacked = dat_dremi_stacked.reset_index('filename', drop=True)
dat_dremi_stacked = dat_dremi_stacked.reset_index()

t_dat = bin_dat.reset_index()
t_dat = t_dat[bin_dat.index.names]
t_dat.columns = t_dat.columns.get_level_values(0)
dat_dremi_stacked = pd.merge(t_dat, dat_dremi_stacked, how='outer')

# due to the way the merging was performed, both bindat and dat_dremi are now aligned

dat_dremi_stacked = dat_dremi_stacked.dropna(subset=['marker'])
bin_dat[('stats', 'dremi')] = dat_dremi_stacked['dremi'].tolist()


bin_dat[('stats', 'dremi_median')] = bin_dat[('stats', 'dremi')].groupby(level=['marker', 'origin', 'target', 'timepoint','perturbation']).transform(np.median)

bin_dat = bin_dat.loc[bin_dat.index.get_level_values('target').isin(excluded_names) == False, :]
neg_mark_fil = bin_dat.index.get_level_values('marker').isin(neg_ctrl_names)

bin_dat.loc[neg_mark_fil == False].index.get_level_values('marker').unique()

bins = np.arange(0,1,0.025)

bin_dat.loc[neg_mark_fil == False, ('stats', 'dremi_median')].hist(normed=1,bins=bins)
bin_dat.loc[neg_mark_fil, ('stats', 'dremi_median')].hist(normed=1, alpha=0.4,bins=bins)

maxneg_dremi = bin_dat.loc[neg_mark_fil, ('stats', 'dremi_median')].max()
maxneg_dremi_99 = np.percentile(bin_dat.loc[neg_mark_fil, ('stats', 'dremi_median')].dropna(),99,)
maxneg_dremi_90 = np.percentile(bin_dat.loc[neg_mark_fil, ('stats', 'dremi_median')].dropna(),90,)
print(bin_dat.loc[neg_mark_fil == False, ('stats', 'dremi_median')].max())
print(bin_dat.loc[neg_mark_fil, ('stats', 'dremi_median')].max())

print(maxneg_dremi)
print(np.sum(bin_dat.loc[neg_mark_fil == False, ('stats', 'dremi_median')] > maxneg_dremi)/3)
print(maxneg_dremi_99)
print(np.sum(bin_dat.loc[neg_mark_fil == False, ('stats', 'dremi_median')] > maxneg_dremi_99)/3)

bin_dat['stats']
bins = np.arange(0,1,0.025)

bin_dat.loc[neg_mark_fil == False, ('stats', 'median_mean_var_ratio')].hist(normed=1,bins=bins)
bin_dat.loc[neg_mark_fil, ('stats', 'median_mean_var_ratio')].hist(normed=1, alpha=0.4,bins=bins)

maxneg_bp = bin_dat.loc[neg_mark_fil, ('stats', 'median_mean_var_ratio')].max()
maxneg_bp_99 = np.percentile(bin_dat.loc[neg_mark_fil, ('stats', 'median_mean_var_ratio')].dropna(),99)
maxneg_bp_90 = np.percentile(bin_dat.loc[neg_mark_fil, ('stats', 'median_mean_var_ratio')].dropna(),90)

print(bin_dat.loc[neg_mark_fil == False, ('stats', 'median_mean_var_ratio')].max())
print(bin_dat.loc[neg_mark_fil, ('stats', 'median_mean_var_ratio')].max())

print(maxneg_bp)
print(np.sum(bin_dat.loc[neg_mark_fil == False, ('stats', 'median_mean_var_ratio')] > maxneg_bp)/3)
print(maxneg_bp_99)
print(np.sum(bin_dat.loc[neg_mark_fil == False, ('stats', 'median_mean_var_ratio')] > maxneg_bp_99)/3)

maxneg_bp_99

bin_dat['stats']
bins = np.arange(0,1,0.025)

bin_dat.loc[neg_mark_fil == False, ('stats', 'median_abs_corr_spearman_overall')].hist(normed=1,bins=bins)
bin_dat.loc[neg_mark_fil, ('stats', 'median_abs_corr_spearman_overall')].hist(normed=1, alpha=0.4,bins=bins)

maxneg_sp = bin_dat.loc[neg_mark_fil, ('stats', 'median_abs_corr_spearman_overall')].max()
maxneg_sp_99 = np.percentile(bin_dat.loc[neg_mark_fil, ('stats', 'median_abs_corr_spearman_overall')].dropna(),99,)
maxneg_sp_90 = np.percentile(bin_dat.loc[neg_mark_fil, ('stats', 'median_abs_corr_spearman_overall')].dropna(),90,)
print(bin_dat.loc[neg_mark_fil == False, ('stats', 'median_abs_corr_spearman_overall')].max())
print(bin_dat.loc[neg_mark_fil, ('stats', 'median_abs_corr_spearman_overall')].max())

print(maxneg_sp)
print(np.sum(bin_dat.loc[neg_mark_fil == False, ('stats', 'median_abs_corr_spearman_overall')] > maxneg_sp)/3)
print(maxneg_bp_99)
print(np.sum(bin_dat.loc[neg_mark_fil == False, ('stats', 'median_abs_corr_spearman_overall')] > maxneg_sp_99)/3)

bin_dat['stats']
bins = np.arange(0,1,0.025)

bin_dat.loc[neg_mark_fil == False, ('stats', 'median_abs_corr_pearson_overall')].hist(normed=1,bins=bins)
bin_dat.loc[neg_mark_fil, ('stats', 'median_abs_corr_pearson_overall')].hist(normed=1, alpha=0.4,bins=bins)

maxneg_pc = bin_dat.loc[neg_mark_fil, ('stats', 'median_abs_corr_pearson_overall')].max()
maxneg_pc_99 = np.percentile(bin_dat.loc[neg_mark_fil, ('stats', 'median_abs_corr_pearson_overall')].dropna(),99,)
maxneg_pc_90 = np.percentile(bin_dat.loc[neg_mark_fil, ('stats', 'median_abs_corr_pearson_overall')].dropna(),90,)
print(bin_dat.loc[neg_mark_fil == False, ('stats', 'median_abs_corr_pearson_overall')].max())
print(bin_dat.loc[neg_mark_fil, ('stats', 'median_abs_corr_pearson_overall')].max())

print(maxneg_pc)
print(np.sum(bin_dat.loc[neg_mark_fil == False, ('stats', 'median_abs_corr_pearson_overall')] > maxneg_sp)/3)
print(maxneg_pc_99)
print(np.sum(bin_dat.loc[neg_mark_fil == False, ('stats', 'median_abs_corr_pearson_overall')] > maxneg_sp_99)/3)

#fil = bin_dat[('stats', 'dremi_median')] > 0.24
#fildat = bin_dat.loc[(fil) & (neg_mark_fil)]
#hits_dremi_GFP = set('_'.join([m, t, str(tp)]) for m, t, tp in
#             zip(fildat.reset_index()['marker'],fildat.reset_index()['target'],fildat.reset_index()['timepoint']))
#hits_dremi_GFP

fil = bin_dat[('stats', 'dremi_median')] > maxneg_dremi_99
fildat = bin_dat.loc[(fil) & (neg_mark_fil == False)]

hits_dremi = set('_'.join([m, t, str(tp)]) for m, t, tp in
             zip(fildat.reset_index()['marker'],fildat.reset_index()['target'],fildat.reset_index()['timepoint']))

fil = bin_dat[('stats', 'median_mean_var_ratio')] > maxneg_bp_99
fildat = bin_dat.loc[(fil) & (neg_mark_fil == False)]

hits_bp = set('_'.join([m, t, str(tp)]) for m, t, tp in
             zip(fildat.reset_index()['marker'],fildat.reset_index()['target'],fildat.reset_index()['timepoint']))

fil = bin_dat[('stats', 'median_abs_corr_spearman_overall')] > maxneg_sp_99

fildat = bin_dat.loc[(fil) & (neg_mark_fil == False)]

hits_sp = set('_'.join([m, t, str(tp)]) for m, t, tp in
             zip(fildat.reset_index()['marker'],fildat.reset_index()['target'],fildat.reset_index()['timepoint']))

fil = bin_dat[('stats', 'median_abs_corr_pearson_overall')] > maxneg_pc_99

fildat = bin_dat.loc[(fil) & (neg_mark_fil == False)]

hits_pc = set('_'.join([m, t, str(tp)]) for m, t, tp in
             zip(fildat.reset_index()['marker'],fildat.reset_index()['target'],fildat.reset_index()['timepoint']))

venn3([hits_bp,hits_sp,hits_dremi],set_labels=['bp-R2', 'Spearman', 'DREMI'])

venn3([hits_bp,hits_pc,hits_dremi],set_labels=['bp-R2', 'Pearson', 'DREMI'])

hits_bp

hits_bp.difference(hits_dremi)

hits_bp.difference(hits_dremi.union(hits_sp))

hits_bp.difference(hits_sp)

hits_dremi.difference(hits_bp)

hits_dremi.difference(hits_bp.union(hits_sp))

hits_dremi.difference(hits_sp)

hits_sp.difference(hits_bp)

hits_sp.difference(hits_bp.union(hits_dremi))

hits_bp.union(hits_sp).difference(hits_dremi)

hits_bp.intersection(hits_dremi).difference(hits_sp)

hit_dict = dict()

hit_dict['all_hits'] = list(set(list(hits_sp)+ list(hits_dremi)+ list(hits_bp)))
hit_dict['is_bp'] = [h in hits_bp for h in hit_dict['all_hits'] ]
hit_dict['is_sp'] = [h in hits_sp for h in hit_dict['all_hits'] ]
hit_dict['is_dremi'] = [h in hits_dremi for h in hit_dict['all_hits'] ]


hit_tab = pd.DataFrame.from_dict(hit_dict)
#hit_tab = hit_tab.set_index('all_hits')

hit_tab = hit_tab.sort_values(by=['is_bp','is_dremi', 'is_sp'],ascending=False)

hit_tab.to_csv(os.path.join(out_folder,'20160429_readout_comparison_vsemptygfp_woCC.csv'),index=False)

