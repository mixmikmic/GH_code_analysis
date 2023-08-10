from pandas import Series, DataFrame
import pandas as pd
import numpy as np

s = Series([4, 7, -5, 3])
s1 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
s2 = Series([2, -9, 6, 7], index=['b', 'a', 'c', 'd'])
data = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
s3 = Series(data)

s.values

s.index

s1['a']

s[0] = 9

s1[['a', 'b', 'c']]

s > 0

s[s > 0]

s * 2

np.exp(s)

'a' in s1

s = s.reindex([0, 1, 2, 3, 4])

s

s.isnull()

s1 + s2

s1 * s2

s1 / s2

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
'year': [2000, 2001, 2002, 2001, 2002],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
index=['one', 'two', 'three', 'four', 'five']
df = DataFrame(data, index=index)

df

df.columns

df['pop']

df[['state', 'pop']]

df.ix['one']

df['debt'] = np.arange(5.)
df

df.drop('five')

df.drop('debt', axis=1)

df['pop'] > 2

df[df['pop'] > 2]

