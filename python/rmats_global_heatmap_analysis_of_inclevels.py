get_ipython().system(' cd /home/bay001/projects/codebase/bfx/;python setup.py build && python setup.py install')

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
from rnaseq import rmats_inclevel_analysis as rmats
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tnrange, tqdm_notebook

pd.set_option('display.max_columns', 500)

wd = '/projects/ps-yeolab3/bay001/maps/current_normed_annotations/se/'


min_events = 0

for cell in ['K562','HepG2']:
    for condition in ['positive','negative']:
        all_files = glob.glob(os.path.join(wd,'*{}-SE.MATS.JunctionCountOnly.{}.nr.txt'.format(cell, condition)))
        merged = pd.DataFrame()
        progress = tnrange(len(all_files))
        for f in all_files:
            df = pd.read_table(f)
            if df.shape[0] >= min_events:
                name = os.path.basename(f).replace('SE.MATS.JunctionCountOnly.','')
                df['junction'] = rmats.get_junction_region(df)
                dx = rmats.get_avg_dpsi_for_all_junctions(df)
                dx.columns = [name]
                merged = pd.merge(merged, dx, how='outer', left_index=True, right_index=True)
            progress.update(1)
        for method in ['pearson','spearman','kendall']:
            g = sns.clustermap(merged.fillna(0).corr(method=method), xticklabels=False, figsize=(10,30))
            x = plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # For y axis
            plt.title('All events any sample (average dpsi) {}, {}, {}'.format(min_events, cell, condition, method))
            plt.savefig(
                '/home/bay001/projects/encode/analysis/dpsi_distribution_figures/nonredundant/{}.{}.correlations.{}.png'.format(
                    cell, condition, method,
                )
            )
            plt.savefig(
                '/home/bay001/projects/encode/analysis/dpsi_distribution_figures/nonredundant/{}.{}.correlations.{}.svg'.format(
                    cell, condition, method,
                )
            )
        merged.to_csv(
            '/home/bay001/projects/encode/analysis/dpsi_distribution_figures/nonredundant/{}.{}.avg_dpsi.txt'.format(
                cell, condition
            ),
            sep='\t'
        )



g = sns.clustermap(merged.fillna(0).corr(method='pearson'), xticklabels=False, figsize=(10,30))
x = plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # For y axis

g = sns.clustermap(merged.fillna(0).corr(method='pearson'), xticklabels=False, figsize=(10,30))
x = plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # For y axis

merged.isnull().sum(axis=1)

merged.shape

for cell in ['K562']:
    for condition in ['positive']:
        all_files = glob.glob(os.path.join(wd,'*{}-SE.MATS.JunctionCountOnly.{}.nr.txt'.format(cell, condition)))
        merged = pd.DataFrame()
        progress = tnrange(len(all_files))
        for f in all_files:
            df = pd.read_table(f)
            if df.shape[0] >= min_events:
                name = os.path.basename(f).replace('-SE.MATS.JunctionCountOnly.txt','')
                df['junction'] = rmats.get_junction_region(df)
                dx = rmats.get_avg_dpsi_for_all_junctions(df)
                dx.columns = [name]
                merged = pd.merge(merged, dx, how='outer', left_index=True, right_index=True)
            progress.update(1)

merged



