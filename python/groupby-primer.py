import numpy as np
import pandas as pd

df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                          'foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three',
                          'two', 'two', 'one', 'three'],
                   'C' : np.random.randn(8),
                   'D' : np.random.randn(8)})
df

g = df.groupby(['B', 'A'])
g

g.sum()
g.aggregate(np.sum) #equivalent

df.groupby(['A', 'B'], as_index=False).sum() # group, but don't make A and B indices
df.groupby(['A', 'B']).sum().reset_index() #equivalent

def get_letter_type(letter):
    if letter.lower() in 'aeiou':
        return 'vowel'
    else:
        return 'consonant'

g = df.groupby(get_letter_type, axis=1)
g.get_group('consonant')
g.sum()

g.groups

arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
          ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]

index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])

s = pd.Series(np.random.randn(8), index=index)

s

s.groupby(level=0).sum()
s.groupby(level='second').sum() # same as level=1
s.sum(level='second') # same as line above

arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
          ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]

index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])

df = pd.DataFrame({'A': [1, 1, 1, 1, 2, 2, 3, 3],
                   'B': np.arange(8),
                   'C': np.arange(3, 11)},
                  index=index)
df

# group by the 2nd index level and the 'A' column:
df.groupby([pd.Grouper(level=1), 'A']).sum()
# or, equivalently, feed in directly as keys to groupby()
df.groupby(['second', 'A']).sum()

df

grouped = df.groupby('A')
for name, group in grouped:
    print('name = {}'.format(name))
    print(group)
grouped.get_group(1)

df_re = pd.DataFrame({'A': [1] * 5 + [5] * 5,
                      'B': np.arange(10)})
print(df_re)
#whereas aggregate lowers the dims, transform functions dont

#aggregate
df_re.groupby('A').sum()

#transform
df_re.groupby('A').expanding().sum()

# filter functions return a subset of the original object
dff = pd.DataFrame({'A': np.arange(8), 'B': list('aabbbbcc')})
print(dff)

dff.groupby('B').filter(lambda x: len(x) > 2) #only pass groups with more than 2 rows.

dff.groupby('B').filter(lambda x: len(x) > 2, dropna=False) # fill w NaN instead of dropping..

#apply can apply any function to each group
dff.groupby('B').apply(lambda x: x.describe())

# variables of 'Categorical' class can be used as groupby keys
data = pd.Series(np.random.randn(100))

factor = pd.qcut(data, [0, .25, .5, .75, 1.])

data.groupby(factor).mean()



