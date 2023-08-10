import numpy as np
import pandas as pd

x = np.array([1, 2, 3])
x

x.dtype

y = np.array([[True, False], [False, True]])
y

y.shape

x = np.random.randint(0, 10, 10)
y = np.random.randint(0, 10, 10)
x, y

[i + j for i, j in zip(x, y)]

x + y

# add a scalar to a 1-d array
x = np.arange(5)
print('x:  ', x)
print('x+1:', x + 1, end='\n\n')

y = np.random.uniform(size=(2, 5))
print('y:  ', y,  sep='\n')
print('y+1:', y + 1, sep='\n')

x * y

x.reshape(1, -1).repeat(2, axis=0) * y

import numpy as np
import pandas as pd

# Many ways to construct a DataFrame
# We pass a dict of {column name: column values}
np.random.seed(42)
df = pd.DataFrame({'A': [1, 2, 3], 'B': [True, True, False],
                   'C': np.random.randn(3)},
                  index=['a', 'b', 'c'])  # also this weird index thing
df

from IPython.display import Image

Image('figures/dataframe.png')

# Single column, reduces to a Series
df['A']

cols = ['A', 'C']
df[cols]

df.loc[['a', 'b']]

df.loc['a':'b']

df.iloc[[0, 1]]

df.iloc[:2]

df.loc['a', 'B']

df.loc['a':'b', ['A', 'C']]

# __getitem__ like before
df['A']

# .loc, like before
df.loc[:, 'A']

# using `.` attribute lookup
df.A

df['mean'] = ['a', 'b', 'c']

df.mean

df['mean']

df.index

df.columns

