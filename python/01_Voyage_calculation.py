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

folder = 'figures'

import flotilla
study = flotilla.embark('singlecell_pnm_figure2_modalities_bayesian',
# study = flotilla.embark('singlecell_pnm_figure2_modalities_bayesian_kmers_cisbp', 
# study = flotilla.embark('singlecell_pnm_figure2_modalities_rmdup', 
                        flotilla_dir='/projects/ps-yeolab/obotvinnik/flotilla_projects/')
# study = flotilla.embark('singlecell_pnm_figure1_supplementary_post_splicing_filtering')

study.splicing.minimum_samples

not_outliers = study.splicing.singles.index.difference(study.splicing.outliers.index)

psi = study.splicing.singles.ix[not_outliers]
grouped = psi.groupby(study.sample_id_to_phenotype)
psi_filtered = grouped.apply(lambda x: x.dropna(axis=1, thresh=study.splicing.minimum_samples))

psi_filtered.head()

get_ipython().run_cell_magic('time', '', "\nfrom bonvoyage import Waypoints\n\nws = Waypoints()\n\nwaypoints = psi_filtered.groupby(study.sample_id_to_phenotype).apply(\n    lambda x: ws.fit_transform(x.dropna(how='all', axis=1)))")

waypoints.max()

pd.DataFrame(ws.seed_data_transformed).max()

transitions = study.phenotype_transitions + [('iPSC', 'MN')]
transitions

from bonvoyage.voyages import Voyages

v = Voyages()

voyages = v.voyages(waypoints, transitions)
voyages.head()

voyages['transition'] = voyages.group1 + '-' + voyages.group2
voyages.head()

voyages.shape

study.supplemental.modalities_tidy.head()

groups = 'group1', 'group2'
voyages_modalities = voyages.copy()

for group in groups:
    voyages_modalities = voyages_modalities.merge(study.supplemental.modalities_tidy, 
                                       left_on=[group, 'event_id'], 
                                       right_on=['phenotype', 'event_id'], copy=False)
    print(voyages_modalities.shape)
    voyages_modalities = voyages_modalities.drop(['phenotype'], axis=1)
    voyages_modalities = voyages_modalities.rename(columns={'modality': '{}_modality'.format(group)})
print(voyages_modalities.shape)
voyages_modalities.head()

study.supplemental.waypoints = waypoints
study.supplemental.voyages = voyages_modalities

study.save('singlecell_pnm_figure5_voyages', flotilla_dir='/projects/ps-yeolab/obotvinnik/flotilla_projects/')

# ll /projects/ps-yeolab/obotvinnik/flotilla_projects//singlecell_pnm_figure4_voyages

# ! md5sum /projects/ps-yeolab/obotvinnik/flotilla_projects/singlecell_pnm_figure4_voyages/*csv



