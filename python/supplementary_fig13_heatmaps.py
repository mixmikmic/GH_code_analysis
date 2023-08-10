import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns

import adnet.library as lib
get_ipython().magic('matplotlib notebook')

fn_bindat = "../example/example_output/main_full/bindat"
fn_completedat = "../example/example_output/main_full/complete_dat"
plot_folder = '../example/testoutput/main_full'
vr_tresh = 0.11

complete_data = pd.read_pickle(fn_completedat)
bin_dat = pd.read_pickle(fn_bindat)

plot_mat = bin_dat['stats']['mean_var_ratio'].copy()

lvl =  ['marker','experiment', 'timepoint','target']
plot_mat = plot_mat.reset_index(lvl, drop=False)
plot_mat['max'] = plot_mat.groupby(['marker'])['mean_var_ratio'].transform(lambda x: 1-np.sum(x >vr_tresh))
plot_mat = plot_mat.sort_values(by=['max']+lvl, ascending=False)
plot_mat =plot_mat.set_index(['max']+lvl, append=False)

plot_mat.loc[plot_mat['mean_var_ratio'] <0,'mean_var_ratio'] = 0

plot_mat = plot_mat.unstack('target')
plot_mat = plot_mat.reset_index('max')

order_index = plot_mat.index


plot_mat = plot_mat['mean_var_ratio']
col = [c for c in complete_data.columns if c != 'GFP']
plot_mat  =  plot_mat[col]

p = plt.figure(figsize=(10,50))
sns.heatmap(plot_mat,square=False, annot=False)
plt.yticks(rotation=0)
plt.xticks(rotation=90) 

plot_mat = complete_data.groupby(level=['marker','experiment', 'timepoint']).median()

plot_mat = lib.transform_arcsinh(plot_mat, reverse=True)
plot_mat = plot_mat.applymap(lambda x: np.max([abs(x),0.0001]))
fil = plot_mat.index.get_level_values('marker').isin(['GFP-FLAG-1','GFP-FLAG-2'])
ref = plot_mat.loc[fil].xs(0, level='timepoint').mean(axis=0)


plot_mat = plot_mat.apply(lambda x: x/ref, axis=1)

plot_mat = plot_mat.reorder_levels(['marker','experiment', 'timepoint'])
plot_mat = plot_mat.loc[order_index]

plot_mat = plot_mat.apply(np.log2)

p = plt.figure(figsize=(10,50))
sns.heatmap(plot_mat, square=True, vmin=-6, vmax=6)
plt.yticks(rotation=0)
plt.xticks(rotation=90) 





