import pandas as pd
import gffutils
import pybedtools
import re
import numpy as np
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

v19db_filename = '/projects/ps-yeolab/genomes/hg19/gencode/v19/gencode.v19.annotation.gtf.db'
v19db = gffutils.FeatureDB(v19db_filename)

folder = '/projects/ps-yeolab/obotvinnik/singlecell_pnms/'
csv_folder = '{}/csvs_for_paper'.format(folder)
# folder2 = '/projects/ps-yeolab2/obotvinnik/singlecell_pnms'

import matplotlib.pyplot as plt

from __future__ import print_function

psi = pd.read_csv('{}/outrigger_v2_bam_unstranded/psi/outrigger_psi.csv'.format(folder), index_col=0).T
print(psi.shape)
psi.head()

bins = np.linspace(0, 1, 50)
bins

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

bins = np.linspace(0, 1, 50)

sns.distplot(psi.values.flat, bins=bins, hist_kws=dict(range=(0, 1)))

notnull = psi.notnull()

constitutively0 = (psi == 0)[notnull].all()

constitutively1 = (psi == 1)[notnull].all()
alternative = psi.columns[~constitutively0 & ~constitutively1]
print('len(alternative)', len(alternative))


constitutively0 = constitutively0[constitutively0].index
constitutively1 = constitutively1[constitutively1].index

print('len(constitutively0)', len(constitutively0))
print('len(constitutively1)', len(constitutively1))

constitutive = constitutively0 | constitutively1
print('len(constitutive)', len(constitutive))

csv_folder

psi_constitutive1 = psi[constitutively1]
psi_constitutive1.to_csv('{}/psi_constitutively1.csv'.format(csv_folder))
psi_constitutive1.head()

psi_constitutive1.shape

psi_constitutive0 = psi[constitutively0]
psi_constitutive0.to_csv('{}/psi_constitutively0.csv'.format(csv_folder))
psi_constitutive0.head()

psi_alternative = psi[alternative]
psi_alternative.to_csv('{}/psi_alternative.csv'.format(csv_folder))
# psi_alternative.to_csv('{}/splicing.csv'.format(csv_folder))
psi_alternative.head()

get_ipython().system(' rm -rf $csv_folder/splicing.csv')
get_ipython().system(' rm -rf $csv_folder/psi.csv')

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

bins = np.linspace(0, 1, 50)

sns.distplot(psi_alternative.values.flat, bins=bins, hist_kws=dict(range=(0, 1)))



