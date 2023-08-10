get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')
import pandas as pd
import numpy as np
import pylab as pl
from matplotlib import pyplot as plt
import palettable as pal

evaluation_frame = pd.read_csv("../assembly_evaluation_data/MMETSP_plotting_data.csv",index_col="SampleName")

evaluation_frame

reads_frame2 = evaluation_frame[['Phylum','Input.Reads','Unique_kmers_assembly','perc_kept_diginorm','mean_orf_percent.x','score.x','Complete_eukaryotic_BUSCO_perc']].dropna()

groups_of_interest=evaluation_frame.groupby('Phylum').count().sort_values(by='Input.Reads', 
                                                                      ascending=False).iloc[0:8].drop(['Unknown']).index
groups_of_interest

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()



import palettable as pal

reads_mean=evaluation_frame.groupby('Phylum').mean().loc[groups_of_interest]
reads_std=evaluation_frame.groupby('Phylum').std().loc[groups_of_interest]

X='n_seqs.x'
Y='mean_orf_percent.x'

t=range(len(reads_mean.index))
fig,ax=plt.subplots(1)
fig.set_size_inches(6,6)
cmap=pal.colorbrewer.qualitative.Paired_9.hex_colors

for n,i in enumerate(groups_of_interest):
    ax.scatter(reads_mean.loc[i,X], reads_mean.loc[i,Y], color=cmap[n], label=i, s=100)
    ax.errorbar(reads_mean.loc[i,X], reads_mean.loc[i,Y], xerr=reads_std.loc[i,X], yerr=reads_std.loc[i,Y], color=cmap[n])

ax.set_xlabel('Number of Transcripts', fontsize='x-large')
ax.set_ylabel('Mean percentage of ORFs', fontsize='x-large')
ax.legend(ncol=1, loc=[1.01,.65], scatterpoints=1)

simpleaxis(ax)
#reads_mean.loc[i].plot(style='o', x=X, y=Y, 
#                 yerr=reads_std.loc[i], xerr=reads_std.loc[i], 
#                 color=cmap)
# reads_mean.plot(kind='scatter', x=X, y=Y, 
#             yerr=reads_std, xerr=reads_std, 
#             c=t, lw=0, s=200, 
#                 cmap=pal.colorbrewer.diverging.PiYG_11.get_mpl_colormap(), ax=ax, 
#                 colorbar=False, label=reads_mean.index)


import palettable as pal
reads_mean=evaluation_frame.groupby('Phylum').mean().loc[groups_of_interest]
reads_std=evaluation_frame.groupby('Phylum').std().loc[groups_of_interest]

Y='score.x'
X='perc_kept_diginorm'

t=range(len(reads_mean.index))
fig,ax=plt.subplots(1)
fig.set_size_inches(6,6)
cmap=pal.colorbrewer.qualitative.Paired_9.hex_colors

for n,i in enumerate(groups_of_interest):
    ax.scatter(evaluation_frame.loc[i,X], reads_mean.loc[i,Y], color=cmap[n], label=i, s=100)
    ax.errorbar(evaluation_frame.loc[i,X], reads_mean.loc[i,Y], xerr=reads_std.loc[i,X], yerr=reads_std.loc[i,Y], color=cmap[n])

ax.set_xlabel('Number of unique kmers', fontsize='x-large')
ax.set_ylabel('Transrate score', fontsize='x-large')
ax.legend(ncol=1, loc=[1.01,.65], scatterpoints=1)

simpleaxis(ax)
#reads_mean.loc[i].plot(style='o', x=X, y=Y, 
#                 yerr=reads_std.loc[i], xerr=reads_std.loc[i], 
#                 color=cmap)
# reads_mean.plot(kind='scatter', x=X, y=Y, 
#             yerr=reads_std, xerr=reads_std, 
#             c=t, lw=0, s=200, 
#                 cmap=pal.colorbrewer.diverging.PiYG_11.get_mpl_colormap(), ax=ax, 
#                 colorbar=False, label=reads_mean.index)

import palettable as pal
reads_mean=evaluation_frame.groupby('Phylum').mean().loc[groups_of_interest]
reads_std=evaluation_frame.groupby('Phylum').std().loc[groups_of_interest]

X='gc_skew.x'
Y='n_seqs.x'

t=range(len(reads_mean.index))
fig,ax=plt.subplots(1)
cmap=pal.colorbrewer.qualitative.Accent_8.hex_colors
# for n,i in enumerate(groups_of_interest):
#     reads_mean.loc[i].plot(kind='scatter', x=X, y=Y, 
#                 yerr=reads_std.loc[i], xerr=reads_std.loc[i], 
#                 color=cmap[n])
reads_mean.plot(kind='scatter', x=X, y=Y, 
            yerr=reads_std, xerr=reads_std, 
            c=t, lw=0, s=200, 
                cmap=pal.colorbrewer.diverging.PiYG_11.get_mpl_colormap(), ax=ax, colorbar=False)


reads_mean.loc[i]

reads_frame1 = evaluation_frame[['Phylum','Input.Reads','Unique_kmers_assembly','num_kmers_reads_diginorm','proportion_kept_diginorm']].dropna()

g = sns.PairGrid(reads_frame1, hue="Phylum", dropna=True)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();
g.savefig("kmer_scatter.pdf")

reads_frame3 = evaluation_frame[['Phylum','n_seqs.x','p_refs_with_CRBB','largest.x','n50.x']].dropna()

g = sns.PairGrid(reads_frame3, hue="Phylum", dropna=True)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();
g.savefig("n_seqs_scatter.pdf")

evaluation_frame.columns

