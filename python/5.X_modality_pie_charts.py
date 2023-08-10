# mkdir /home/obotvinnik/Dropbox/figures2/singlecell_pnm/figure4_voyages/

import six

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from anchor.visualize import MODALITY_TO_COLOR, MODALITY_ORDER, MODALITY_PALETTE
modality_order = MODALITY_ORDER

sns.set(style='ticks', context='talk', rc={'font.sans-serif':'Arial', 'pdf.fonttype': 42})

get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

folder = 'figures' #'/home/obotvinnik/Dropbox/figures2/singlecell_pnm/figure4_voyages'

# import flotilla
# study = flotilla.embark('singlecell_pnm_figure2_modalities_bayesian', 
#                         flotilla_dir='/projects/ps-yeolab/obotvinnik/flotilla_projects/')
# study = flotilla.embark('singlecell_pnm_figure1_supplementary_post_splicing_filtering')

fig, ax = plt.subplots(figsize=(2, 2))

ax.pie([.75, .25], startangle=90, colors=['.20', MODALITY_TO_COLOR['bimodal']], 
       wedgeprops = { 'linewidth' : 2, 'edgecolor':'white' })
fig.savefig('{}/bimodal_unimodal_pie_chart.pdf'.format(folder))

fig, ax = plt.subplots(figsize=(2, 2))

ax.pie([.8, .20], startangle=45, colors=['.20', MODALITY_TO_COLOR['bimodal']],
       wedgeprops = { 'linewidth' : 2, 'edgecolor':'white' })
fig.savefig('{}/changing_events_pie_chart.pdf'.format(folder))

study.supplemental.modalities_tidy.groupby('phenotype').size()



