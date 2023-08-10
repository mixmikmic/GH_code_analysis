get_ipython().magic('pylab inline')

import numpy as np
import pandas as pd
import dask.dataframe as dd

print(np.__version__, pd.__version__)

df = dd.read_hdf('dithered.hdf', key='dithered')

df.info()

# The columns we actually want to use
requested_columns = ['patch', 'footprint','base_PsfFlux_flux','base_PsfFlux_fluxSigma']

selected_columns = df[requested_columns].compute()
selected_columns.shape

selected_patch = df.query("patch == \"'10,10'\"")[requested_columns].compute()
selected_patch.shape

first_100 = df.head(100)
first_100.shape

first_100.head(5)

selected_patch.info()

selected_columns.info()

first_100.info()

