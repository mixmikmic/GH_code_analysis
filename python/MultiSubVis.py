from nipype.algorithms.metrics import FuzzyOverlap
from nilearn.image import resample_to_img
from scipy.stats import pearsonr
from __future__ import division
import matplotlib.pyplot as plt
from collections import Counter
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

pipelines= ['fmriprep-1.0.3','fslfeat_5.0.9']

basedir = os.path.join(os.environ.get("PREPBASEDIR"),"fmriprep_vs_feat")

samplesize = 50
experiment = np.random.choice(range(0,100))
sample = 0

zstatfile = {}
pstatfile = {}
tstat = {}

for pipeline in pipelines:
    zfile = os.path.join(basedir,pipeline,'task_group/samplesize_%i/experiment_%i/stopsignal'%(samplesize,experiment),
                                         'sample_%i/cope11/OLS/zstat1.nii.gz'%sample)
    zstatfile[pipeline]=zfile
    pfile = os.path.join(basedir,pipeline,'task_group/samplesize_%i/experiment_%i/stopsignal'%(samplesize,experiment),
                                         'sample_%i/cope11/OLS/pstat1.nii.gz'%sample)
    pstatfile[pipeline]=pfile
    tstat[pipeline]=nib.load(tfile).get_data()

for pipeline in pipelines:
    ts = zstat[pipeline]
    sns.distplot(ts[ts!=0],label=pipeline)

plt.title("Distribution of t-statistics (whole brain) for one example study (n=50).")
plt.xlabel("T-values")
plt.legend()

cut_coords = [-15, -8, 6, 30, 46, 62]
for pipeline in pipelines:
    ts = zstatfile[pipeline]
    plotting.plot_stat_map(ts,title=pipeline,threshold=2.98,display_mode='z',cut_coords=cut_coords,vmax=14)
plotting.show()

for pipeline in pipelines:
    ts = tstatfile[pipeline]
    plotting.plot_glass_brain(ts,title=pipeline,cmap='RdYlBu_r',
                              vmax = 13,colorbar=True,symmetric_cbar=True,plot_abs=False)
plotting.show()

atlas,labels = atlas.create_atlas()

cut_coords = [-2,-4,-8,-10,-40,-45,-50,-55]
plotting.plot_roi(atlas,display_mode='x',cut_coords=cut_coords,cmap='rainbow',alpha=1)

cut_coords = [-2,-4,-8,-10,-40,-45,-50,-55]
for pipeline in pipelines:
    ts = zstatfile[pipeline]
    plotting.plot_stat_map(ts,display_mode='x',cut_coords=cut_coords,cmap='RdYlBu_r',
                       alpha=1,threshold=2.98,title=pipeline)

ES = {}
for pipeline in pipelines:
    ts = zstatfile[pipeline]
    atlas_resampled = resample_to_img(atlas,ts,interpolation='nearest')
    dat = atlas_resampled.get_data()
    ES[pipeline] = {}
    for k,v in labels.iteritems():
        indxs = np.where(dat==k)
        T = nib.load(tfile).get_data()[indxs]
        CD = np.mean(T)/np.sqrt(samplesize)
        ES[pipeline][v] = CD

pd.DataFrame(ES)

results = pd.read_csv(os.path.join(basedir,"results.csv"),index_col=0)
results['samplesize'] = [int(x) for x in results['samplesize']]

with open(os.path.join(basedir,"tvals.json")) as json_data:
    allT = json.load(json_data)
    json_data.close()

results.head()

samplesize = 90
reslong = pd.melt(results,id_vars=['pipeline','samplesize'],
                  value_vars=labels.values(),var_name="ROI",value_name="Cohen's D")

def ESfigure(reslong,samplesize,xlim=[-0.6,0.6]):
    sns.set_style("whitegrid")
    sns.violinplot(x="Cohen's D",y='ROI',hue='pipeline',data=reslong[reslong.samplesize==samplesize],
                   split=True,inner='quartile')
    plt.title("Distribution of effect sizes with samplesize %i"%samplesize)
    plt.xlim(xlim)

ESfigure(reslong,100)

ESfigure(reslong,40)

ESfigure(reslong,10,xlim=[-1,1])

sns.set_style("whitegrid")
fig = plt.figure(figsize=(20,8), dpi= 100, facecolor='w', edgecolor='k')
plt.subplot(1,3,1)
sns.boxplot(x="samplesize", y="correlation", hue="pipeline", data=results)
plt.ylabel("Correlation")
plt.title("Correlation")
plt.subplot(1,3,2)
sns.boxplot(x="samplesize", y="fdice", hue="pipeline", data=results)
plt.ylabel("Fuzzy Dice")
plt.title("Fuzzy Dice")
plt.subplot(1,3,3)
sns.boxplot(x="samplesize", y="dice", hue="pipeline", data=results)
plt.title("Dice Index")
plt.ylabel("Binary Dice")

len(allT[pipeline]['10'])

fig = plt.figure(figsize=(10,6), dpi= 100, facecolor='w', edgecolor='k')

for idx,samplesize in enumerate(np.arange(10,101,40).tolist()):
    for idy,pipeline in enumerate(pipelines):
        if idy==0:
            lsty = '-'
        else:
            lsty = ':'
        sns.distplot(allT[pipeline][str(samplesize)],color=cols[idx],hist=False,
                     kde_kws={"linestyle": lsty},label="%i subjects - %s"%(samplesize,pipeline))

plt.title("Distribution of group analysis T-values (whole brain)")
plt.xlabel("T-values")
plt.legend()

groupfile = os.path.join(basedir,'smoothness_group.csv')
group = pd.read_csv(groupfile)
group.head()

fig = plt.figure(figsize=(10,6), dpi= 100, facecolor='w', edgecolor='k')
for idx,samplesize in enumerate(np.arange(10,101,40).tolist()):
    for idy,pipeline in enumerate(pipelines):
        if idy==0:
            lsty = '-'
        else:
            lsty = ':'
        subset = group[np.logical_and(group.pipeline==pipeline, group.samplesize==samplesize)]
        sns.distplot(subset.FWHM_zstat,color=cols[idx+1],hist=False,
                     kde_kws={"linestyle": lsty,"linewidth":4},label="%i subjects - %s - zstat"%(samplesize,pipeline))
        sns.distplot(subset.FWHM_resid,color=cols[idx+1],hist=False,
                     kde_kws={"linestyle": lsty,"linewidth":1},label="%i subjects - %s - residuals"%(samplesize,pipeline))

plt.title("Distribution of group analysis T-values (whole brain)")
plt.xlabel("FWHM")
plt.legend()

