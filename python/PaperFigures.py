from nipype.algorithms.metrics import FuzzyOverlap
from nilearn.image import resample_to_img
from scipy.stats import pearsonr
from __future__ import division
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import gridspec
from nilearn import plotting
import nibabel as nib
import seaborn as sns
import pandas as pd
import numpy as np
import palettable
import json
import sys
import os

sys.path.append(os.path.join(os.environ.get("HOME"),"CNP_analysis"))
from utils import get_config, atlas
from utils.prog import log_progress as lp

get_ipython().magic('matplotlib inline')

sns.set_style('white')

cols = palettable.tableau.ColorBlind_10.hex_colors
cols += palettable.tableau.PurpleGray_6.hex_colors
cols += palettable.tableau.Tableau_10.hex_colors

sns.set_palette(palettable.tableau.ColorBlind_10.mpl_colors)

basedir = os.environ.get("PREPBASEDIR")
pipelines = ['fslfeat_5.0.9','fmriprep-1.0.3']
fact = {
    'fslfeat_5.0.9': 2,
    'fmriprep-1.0.3': 3.3
}

subjectsfile = os.path.join(basedir,'fmriprep_vs_feat/smoothness_subjects.csv')
subjects = pd.read_csv(subjectsfile,index_col=0)

sub_wide = pd.melt(subjects,value_vars=[x for x in subjects.columns if x.startswith('FWHM_')],id_vars=['subject','pipeline'])
sub_wide = sub_wide[np.logical_or(sub_wide['variable']=='FWHM_resid',sub_wide['variable']=='FWHM_unpr')]

sub_wide['variable']=='FWHM_resid'
sub_wide.replace(to_replace="FWHM_resid",value="residuals after modeling",inplace=True)
sub_wide.replace(to_replace="FWHM_unpr",value="data after preprocessing",inplace=True)
sub_wide.replace(to_replace="fmriprep-1.0.3",value="fmriprep",inplace=True)
sub_wide.replace(to_replace="fslfeat_5.0.9",value="feat",inplace=True)

fig = plt.figure(figsize=(15, 6))
gs1 = gridspec.GridSpec(2, 2,
                       width_ratios=[3, 2],
                       height_ratios=[1, 1]
                       )

pipelines_labels = ['feat','fmriprep']
samplesize = 30

# FIGURE 1: SLICES

cut_coords = [10,30,60]
for idx,pipeline in enumerate(pipelines):
    ax = plt.subplot(gs1[idx, 0])
    image = os.path.join(basedir,'fmriprep_vs_feat',pipeline,"task_acm/stopsignal/zstat11_ACM_diff.nii.gz")
    plotting.plot_stat_map(image,title=pipelines_labels[idx],threshold=0.25,
                           display_mode='z',cut_coords=cut_coords,vmax=0.8,axes=ax)

# FIGURE 2: SMOOTHNESS


ax2 = plt.subplot(gs1[:, 1])
sns.violinplot(x='variable',y='value',hue='pipeline',data=sub_wide)

plt.ylabel("FWHM")
plt.xlabel("")
plt.legend()    

results = pd.read_csv(os.path.join(basedir,"fmriprep_vs_feat/results.csv"),index_col=0)
results['samplesize'] = [int(x) for x in results['samplesize']]
results = results[results['samplesize']%10==0]
results['pipeline_label'] = 'fmriprep'
results['pipeline_label'][results['pipeline']=='fslfeat_5.0.9'] = 'feat'

results.head()

sns.set_style("whitegrid")
fig = plt.figure(figsize=(15,6), dpi= 100, facecolor='w', edgecolor='k')
names = ['Correlation','Fuzzy Dice Index','Binary Dice Index']
axes = {}
for idx,metric in enumerate(['correlation','fdice','dice']):
    plt.subplot(1,3,idx)
    axes[idx] = sns.boxplot(x="samplesize", y=metric, hue="pipeline_label", data=results,linewidth=1.5)
    plt.ylabel(names[idx])
    plt.title(names[idx])
    leg = axes[idx].get_legend()
    new_title = 'pipeline'
    leg.set_title(new_title)

