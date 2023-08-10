import pandas as pd
import numpy as np

vessels = pd.read_csv('../data/AIS/vessel_information.csv', index_col=0)

vessels.shape

vessels.columns

# Sample Series object
flag = vessels.flag
flag

# Numpy-style indexing
flag[:10]

# Indexing by label
flag[[298716,725011300]]

vessels[['num_names','num_types']].head()

vessels[vessels.max_loa > 700]

vessels.loc[720768000, ['names','flag', 'type']]

vessels.loc[:4731, 'names']

vessels.columns

vessels.loc[:310, 'flag':'loa']

vessels.iloc[:5, 5:8]

np.random.seed(42)
normal_vals = pd.DataFrame({'x{}'.format(i):np.random.randn(100) for i in range(5)})

normal_vals.head()

normal_vals.where(normal_vals > 0).head()

normal_vals.where(normal_vals > 0, other=-normal_vals).head()

normal_vals.where(normal_vals>0, other=lambda y: -y*100).head()

normal_vals.mask(normal_vals>0).head()

normal_vals[(normal_vals.x1 > normal_vals.x0) & (normal_vals.x3 > normal_vals.x2)].head()

normal_vals.query('(x1 > x0) & (x3 > x2)').head()

min_loa = 700

vessels.query('max_loa > @min_loa')

