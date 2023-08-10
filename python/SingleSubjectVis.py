from nilearn import plotting
import matplotlib.pyplot as plt
import nibabel as nib
import seaborn as sns
import pandas as pd
import numpy as np
import palettable
import sys
import os

sys.path.append(os.path.join(os.environ.get("HOME"),"CNP_analysis"))
from utils import get_config

get_ipython().magic('matplotlib inline')

cols = palettable.tableau.ColorBlind_10.hex_colors
cols += palettable.tableau.PurpleGray_6.hex_colors
cols += palettable.tableau.Tableau_10.hex_colors

sns.set_palette(palettable.tableau.ColorBlind_10.mpl_colors)

sns.set_style("whitegrid")

basedir = os.environ.get("PREPBASEDIR")
pipelines = ['fslfeat_5.0.9','fmriprep-1.0.3']
subject = 'sub-50049'

images = {}
for pipeline in pipelines:
    z11 = os.path.join(basedir,'fmriprep_vs_feat/%s/task/%s/stopsignal.feat/stats/zstat11.nii.gz'%(pipeline,subject))
    images[pipeline] = z11

for idx,pipeline in enumerate(pipelines):
    plotting.plot_glass_brain(images[pipeline],title=pipeline,cmap='RdYlBu_r',
                              colorbar=True,symmetric_cbar=True,plot_abs=False)
plotting.show()

cut_coords = [-15, -8, 6, 30, 46, 62]
for idx,pipeline in enumerate(pipelines):
   plotting.plot_stat_map(images[pipeline],title=pipeline,vmax=5,display_mode='z',threshold=1.65,cut_coords=cut_coords)
plotting.show()

meanmasks = {}
totalmasks = {}
subjects = {}
pipelines = ['fslfeat_5.0.9','fmriprep-1.0.3']

for pipeline in pipelines:
    taskdir = os.path.join(basedir,'fmriprep_vs_feat',pipeline,'task')
    subjects[pipeline] = os.listdir(taskdir)
    dims = [65,77,49] if pipeline.startswith('fmriprep') else [97,115,97]
    mask = np.zeros(dims+[len(subjects[pipeline])])
    for idx,subject in enumerate(subjects[pipeline]):
        cf = get_config.get_files(pipeline,subject,'stopsignal')
        maskfile = cf['standard_mask']
        #maskfile = os.path.join(taskdir,subject,'stopsignal.feat/mask.nii.gz')
        if idx==0:
            aff = nib.load(maskfile).affine
            hd = nib.load(maskfile).header
        mask[:,:,:,idx] = nib.load(maskfile).get_data()  
    mnmask = np.mean(mask,3)
    totmask = (mnmask>0)*1.
    meanmasks[pipeline] = nib.Nifti1Image(mnmask, affine = aff, header = hd)
    totalmasks[pipeline] = nib.Nifti1Image(totmask, affine = aff, header = hd)

for idx,pipeline in enumerate(pipelines):
    plotting.plot_glass_brain(meanmasks[pipeline],title=pipeline,cmap='RdYlBu_r',
                              colorbar=True,threshold=0.01,symmetric_cbar=False)

plotting.show()

# collect all subjects
prepdir = os.environ.get("PREPBASEDIR")
fmriprepdir = os.path.join(prepdir,'fmriprep-1.0.3','fmriprep')
subjects = [x for x in os.listdir(fmriprepdir) if x[-4:]!='html' and x[:4]=='sub-']

subs = []
for subject in subjects:
    image = os.path.join(prepdir,'fmriprep-1.0.3','fmriprep',subject,'func',
                         '%s_task-%s_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'%(subject,'stopsignal'))
    if os.path.exists(image):
        subs.append(subject)

subs = set(subs)-set(['sub-50010','sub-10527'])
subjects = list(subs)

tvals = {}
for pipeline in pipelines:
    tvals[pipeline]=[]
    for subject in subjects:
        t11 = os.path.join(prepdir,'fmriprep_vs_feat/%s/task/%s/stopsignal.feat/stats/zstat11.nii.gz'%(pipeline,subject))
        im = nib.load(t11).get_data()
        imnonnul = im[im!=0]
        tvals[pipeline] += imnonnul.tolist()

plt.figure(figsize=(10,7))
for pipeline in pipelines:
    sns.distplot(tvals[pipeline],label=pipeline)
plt.title("Distribution of Z-values of 1st level analysis - GO-StopSuccess contrast")
plt.legend()

subjectsfile = os.path.join(basedir,'fmriprep_vs_feat/smoothness_subjects.csv')
subjects = pd.read_csv(subjectsfile)
subjects.head()

fig = plt.figure(figsize=(10,6), dpi= 100, facecolor='w', edgecolor='k')
for idy,pipeline in enumerate(pipelines):
    if idy==0:
        lsty = '-'
    else:
        lsty = ':'
    subset = subjects[subjects.pipeline==pipeline]
    sns.distplot(subset.FWHM_unpr,color=cols[0],hist=False,label="%s - after preprocessing"%pipeline,kde_kws={"linestyle": lsty})
    sns.distplot(subset.FWHM_data,color=cols[2],hist=False,label="%s - after smoothing"%pipeline,kde_kws={"linestyle": lsty})
    sns.distplot(subset.FWHM_zstat,color=cols[1],hist=False,label="%s - zstat"%pipeline,kde_kws={"linestyle": lsty})
    sns.distplot(subset.FWHM_resid,color=cols[3],hist=False,label="%s - residuals"%pipeline,kde_kws={"linestyle": lsty})

plt.title("Smoothness of single subject data")
plt.xlabel("FWHM")
plt.legend()    

for idx,pipeline in enumerate(pipelines):
    image = os.path.join(basedir,'fmriprep_vs_feat',pipeline,"task_acm/stopsignal/zstat11_ACM_diff.nii.gz")
    plotting.plot_glass_brain(image,title=pipeline,vmin=-1,vmax=1,colorbar=True,cmap='RdYlBu_r',symmetric_cbar=True)
plotting.show()

cut_coords = [-15, -8, 6, 30, 46, 62]
for idx,pipeline in enumerate(pipelines):
    image = os.path.join(basedir,'fmriprep_vs_feat',pipeline,"task_acm/stopsignal/zstat11_ACM_diff.nii.gz")
    plotting.plot_stat_map(image,title=pipeline,threshold=0.25,display_mode='z',cut_coords=cut_coords,vmax=0.8)
plotting.show()

plt.figure(figsize=(10,7))
for idx,pipeline in enumerate(pipelines):
    image = os.path.join(basedir,'fmriprep_vs_feat',pipeline,"task_acm/stopsignal/zstat11_ACM_diff.nii.gz")
    imdat = nib.load(image).get_data()
    sns.distplot(imdat[imdat != 0],label=pipeline,norm_hist=True,bins=np.arange(-1,1,0.02),kde_kws={'bw':0.015})
plt.xlim([-0.4,0.4])
plt.xlabel("Activation percentage")
plt.legend()

