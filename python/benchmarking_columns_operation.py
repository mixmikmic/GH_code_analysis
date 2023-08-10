import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.random(size=(10000,2)), columns=['foo', 'bar'])

df.head()

get_ipython().run_cell_magic('timeit', '', "pd.Series([\n    x[1]['foo'] * x[1]['bar'] for x in df.iterrows()\n])")

get_ipython().run_cell_magic('timeit', '', "df.apply(lambda x: x['foo'] * x['bar'], axis=1)")

get_ipython().run_cell_magic('timeit', '', 'df.foo.mul(df.bar)')

