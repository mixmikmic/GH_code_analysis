# %load ../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)
 
#%config InlineBackend.figure_formats = {'pdf',}
get_ipython().magic('matplotlib inline')

import seaborn as sns
sns.set_context('notebook')
sns.set_style('darkgrid')

# Creating DataFrame with MultiIndexes on both axis
idx = pd.MultiIndex.from_product([['A','B','C','D'],['XX', 'YY', 'ZZ'], [1,2]], names=['Operator', 'Facility', 'Shift'])

arrays = np.array(['OB11', 'OB11', 'HH90']), np.array(['M1', 'M2', 'M3'])
idx2 = pd.MultiIndex.from_arrays(arrays, names=['Machine type', 'MachineID'])

df = pd.DataFrame(np.random.randint(0, 50, (24,3)), index=idx, columns=idx2)
df

# Index must be sorted
df.sort_index(axis=1,inplace=True)
df.sort_index(axis=0,inplace=True)

idx = pd.IndexSlice
df.loc[idx[['B','D'], :, 2],idx[:,['M1','M3']]]

