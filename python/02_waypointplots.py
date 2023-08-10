import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import six

from anchor.visualize import MODALITY_TO_COLOR, MODALITY_ORDER, MODALITY_PALETTE
modality_order = MODALITY_ORDER

sns.set(style='ticks', context='paper', rc={'font.sans-serif':'Arial', 'pdf.fonttype': 42})

get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

folder = 'figures' #'/home/obotvinnik/Dropbox/figures2/singlecell_pnm/figure4_voyages'

import flotilla

flotilla_dir = '/projects/ps-yeolab/obotvinnik/flotilla_projects'
study = flotilla.embark('singlecell_pnm_figure5_voyages', flotilla_dir=flotilla_dir)

figure_folder = '{}/002_waypointplots'.format(folder)
get_ipython().system(' mkdir $figure_folder')

waypoints = study.supplemental.waypoints.rename(columns={'Unnamed: 1': 'event_id'})

waypoints = waypoints.set_index('event_id', append=True)
waypoints.head()

import bonvoyage

study.supplemental.modalities_tidy.groupby(['phenotype', 'modality']).size()

study.supplemental.modalities_tidy = study.supplemental.modalities_tidy.replace(
    {'modality': {'ambivalent': 'uncategorized', 'concurrent': 'middle'}})

study.supplemental.modalities_tidy.groupby(['phenotype', 'modality']).size()

modalities_grouped = study.supplemental.modalities_tidy.groupby('phenotype')

import matplotlib as mpl

sns.set(style='ticks', context='paper', rc={'font.sans-serif':'Arial', 'pdf.fonttype': 42})

from anchor.visualize import MODALITY_TO_COLOR, MODALITY_ORDER, MODALITY_PALETTE
modality_order = MODALITY_ORDER

pkm_event = u'isoform1=junction:chr15:72494962-72499068:-@exon:chr15:72494795-72494961:-@junction:chr15:72492997-72494794:-|isoform2=junction:chr15:72495530-72499068:-@exon:chr15:72495363-72495529:-@junction:chr15:72492997-72495362:-'

import bonvoyage

get_ipython().run_line_magic('pinfo2', 'bonvoyage.waypointplot')

kinds = 'scatter', 'hexbin'

colorbar_ticklabels = [r'$10^{' + str(i) + '}$' for i in range(1, 5)]

for phenotype, df in waypoints.groupby(level=0, axis=0):
    df.index = df.index.droplevel(0)
    marker = study.phenotype_to_marker[phenotype]
    
#     six.print_(df.head())
    
    for kind in kinds:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        kwargs = dict(rasterized=True, alpha=0.2, marker=marker) if kind == 'scatter' else dict(gridsize=10, vmax=4)
        bonvoyage.waypointplot(df, ax=ax, kind=kind, **kwargs)

#         # Double-plot PKM
#         x, y = df.loc[pkm_event]
#         color = study.phenotype_to_color[phenotype]
#         ax.plot(x, y, marker, markerfacecolor=None, markeredgecolor='k', markeredgewidth=.5, color=color)
#         ax.set(title='')
        
        ax.set(title='')
        fig.tight_layout()
#         sns.despine(offset=2)
        fig.savefig('{}/{}_{}.pdf'.format(figure_folder, phenotype, kind), dpi=300)
        
        if kind == 'hexbin':
            fig_colorbar, ax_colorbar = plt.subplots(figsize=(1, 1.5))
            plt.colorbar(ax.collections[0], cax=ax_colorbar, 
                         orientation='vertical',  label='Count', 
                         ticks=[1, 2, 3, 4])#mpl.ticker.MaxNLocator(4))
            ax_colorbar.yaxis.set_ticklabels(colorbar_ticklabels)
            fig_colorbar.tight_layout()
            fig_colorbar.savefig('{}/{}_{}_colorbar.pdf'.format(figure_folder, phenotype, kind), dpi=300)
        
        
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    modality_df = modalities_grouped.get_group(phenotype)
    groupby = modality_df.set_index('event_id')['modality']
    bonvoyage.waypointplot(df, ax=ax, kind='scatter', rasterized=True, features_groupby=groupby, marker=marker)
    ax.set(title='')
#     sns.despine(offset=3)
    fig.tight_layout()
    fig.savefig('{}/{}_scatter_modality.pdf'.format(figure_folder, phenotype), dpi=300)

kind = 'hexbin'

fig, ax = plt.subplots(figsize=(1.5, 1.5))
kwargs = dict(rasterized=True, alpha=0.2, marker=marker) if kind == 'scatter' else dict(gridsize=10, vmax=4)
bonvoyage.waypointplot(df, ax=ax, kind=kind, **kwargs)


fig_colorbar, ax_colorbar = plt.subplots(figsize=(1, 1.5))
plt.colorbar(ax.collections[0], cax=ax_colorbar, 
                         orientation='vertical',  label='Count', 
                         ticks=[-1, 0, 1, 2, 3, 4])


# # Double-plot PKM
# x, y = df.loc[pkm_event]
# color = study.phenotype_to_color[phenotype]
# ax.plot(x, y, marker, markerfacecolor=None, markeredgecolor='k', markeredgewidth=1, color=color)
# ax.set(title='')

kind = 'scatter'

colorbar_ticklabels = [r'$10^{' + str(i) + '}$' for i in range(4)]

for phenotype, df in waypoints.groupby(level=0, axis=0):
    df.index = df.index.droplevel(0)
#     six.print_(df.head())
    
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    modality_df = modalities_grouped.get_group(phenotype)
    groupby = modality_df.set_index('event_id')['modality']
    marker = study.phenotype_to_marker[phenotype]
    bonvoyage.waypointplot(df, ax=ax, kind='scatter', rasterized=True, features_groupby=groupby, marker=marker)
    
    # Double-plot PKM
    x, y = df.loc[pkm_event]
    color = MODALITY_TO_COLOR[groupby[pkm_event]]
    ax.plot(x, y, marker, markerfacecolor=None, markeredgecolor='k', markeredgewidth=1, color=color)
    ax.set(title='')
    
    fig.tight_layout()
    fig.savefig('{}/{}_scatter_modality_pkm_annotated.pdf'.format(figure_folder, phenotype), dpi=300)





