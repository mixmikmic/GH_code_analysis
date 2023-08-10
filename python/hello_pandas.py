get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

import pandas as pd

s = pd.Series([1,3,5,np.nan,6,8])
s

dates = pd.date_range('20160101', periods=6)
dates

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
df

df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                     'E' : pd.Categorical(["a","b","c", "d"]),
                     'F' : 'foo' })
df2

df2.dtypes

df.index, df.columns, df.values

df2.index, df2.columns, df2.values

df.describe()

df2.describe()

df.T

df.sort_index(axis=1, ascending=False)

df.sort_values(by='B')

df['A']

df[0:3]

df['20160102':'20160104']

df.loc[dates[2]]

df.loc[:,['A','D']]

df.iloc[3:5,0:2]

df[df.A > 0.5]

df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']
df2

df2[df2['E'].isin(['two','four'])]

df['F'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20160101', periods=6))
df

df.at[dates[0],'A'] = 0.456
df.at[dates[0],'A']

df.iat[0,1] = 0.123
df.iat[0,1]

df.loc[:,'D'] = np.array([5] * len(df))
df.loc[:,'D']

df.loc[:,'B':'D'] = np.random.randn(len(df), 3)
df.loc[:,'B':'D']

df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[:, 'E'] = np.random.randn(len(df1))
df1

df1.iloc[1,5] = np.nan
df1.dropna(how='any')

df1.iloc[1,5] = np.nan
df1.fillna(value=5)

pd.isnull(df1)

df.median()

s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
s

df.sub(s, axis='index')

df.apply(np.cumsum)

a = np.array([[1,2,3], [4,5,6]])
np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns

np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows

df.apply(lambda x: (x.max(),  x.min()))

s = pd.Series(np.random.randint(0, 7, size=10))
s.value_counts()

s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()

s.str.capitalize()

s.str.cat()

df = pd.DataFrame(np.random.randn(10, 4))
pieces = [df[:3], df[3:7], df[7:]]
pieces[0]

pd.concat(pieces)

left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
pd.merge(left, right, on='key')

left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
pd.merge(left, right, on='key')

left = pd.DataFrame({'key': ['foo', 'bar', 'foo'], 'lval': [1, 2, 3]})
right = pd.DataFrame({'key': ['foo', 'bar', 'foo'], 'rval': [4, 5, 6]})
joined = pd.merge(left, right, on='key')
joined

joined[joined.key == 'foo'].lval.sum()

joined.groupby(by='key').sum()

df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
s = df.iloc[3]
df.append(s, ignore_index=True)

df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                           'foo', 'bar', 'foo', 'foo'],
                    'B' : ['one', 'one', 'two', 'three',
                           'two', 'two', 'one', 'three'],
                    'C' : np.random.randn(8),
                    'D' : np.random.randn(8)})

df.groupby(['A','B']).sum()

tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                       'foo', 'foo', 'qux', 'qux'],
                      ['one', 'two'] * 4]))
tuples

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
index

df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df

df.loc['foo', 'one']

df.loc['foo', :].A

df2 = df[:4]
stacked = df2.stack()

stacked.unstack()

stacked.unstack(0)

stacked.unstack(1)

stacked.unstack(2)

df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                  'B' : ['A', 'B', 'C'] * 4,
                  'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                  'D' : np.random.randn(12),
                  'E' : np.random.randn(12)})
df

pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])

df = pd.DataFrame(data={'Province' : ['ON','QC','BC','AL','AL','MN','ON'],
                         'City' : ['Toronto','Montreal','Vancouver','Calgary','Edmonton','Winnipeg','Windsor'],
                         'Sales' : [13,6,16,8,4,3,1]})
df

table = pd.pivot_table(df,values=['Sales'],index=['Province'],columns=['City'],aggfunc=np.sum,margins=True)
table

table.stack('City')

rng = pd.date_range('1/1/2016', periods=100, freq='S')
rng[50]

len(rng)

ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.head()

ts5 = ts.resample('5Min')

ts5.count()

ts5.median()

ts.asfreq('10T')

df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
df

df["grade"] = df["raw_grade"].astype("category")
df["grade"]

df["grade"].cat.categories = ["good", "normal", "bad"]
df

df.groupby("grade").size()

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2016', periods=1000))
ts = ts.cumsum()
ts.plot()

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                   columns=['A', 'B', 'C', 'D'])

df = df.cumsum()
plt.figure(); df.plot(); plt.legend(loc='best')

df.to_csv('/tmp/foo.csv')

pd.read_csv('/tmp/foo.csv')

df.to_hdf('/tmp/foo.h5','df')



