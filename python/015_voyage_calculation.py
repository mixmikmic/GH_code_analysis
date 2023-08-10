from __future__ import print_function

from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


import flotilla
sns.set(style='ticks', context='talk')

figure_folder = 'figures'
get_ipython().system(' mkdir $figure_folder')

flotilla_dir = '/projects/ps-yeolab/obotvinnik/flotilla_projects/'

study = flotilla.embark('singlecell_pnm_miso_modalities', flotilla_dir=flotilla_dir)

study.splicing.minimum_samples

not_outliers = study.splicing.singles.index.difference(study.splicing.outliers.index)

psi = study.splicing.singles.ix[not_outliers]
grouped = psi.groupby(study.sample_id_to_phenotype)
psi_filtered = grouped.apply(lambda x: x.dropna(axis=1, thresh=study.splicing.minimum_samples))

from astrolabe import Waypoints

ws = Waypoints()

waypoints = psi_filtered.groupby(study.sample_id_to_phenotype).apply(
    lambda x: ws.fit_transform(x.dropna(how='all', axis=1)))

transitions = study.phenotype_transitions + [('iPSC', 'MN')]
transitions

from astrolabe.voyages import Voyages

v = Voyages()

voyages = v.voyages(waypoints, transitions)
voyages.head()


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

study.save('singlecell_pnm_miso_modalities_voyages', flotilla_dir=flotilla_dir)



