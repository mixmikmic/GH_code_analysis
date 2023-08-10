import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

a = pd.Series(data=[1, 2, 3, 4])#, index=[0, 1, 2, 3])
print(type(a))
a

b = pd.Series(data=[1, 2, 3, 4], index=['Nick', 'Bill', 'John', 'Bob'])
b

c = pd.Series(data=[2, 2, 3, 'hi'], index=['Nick', 'Bill', 'John', 'Bob'])
c

b.name = 'SubjNums'
b

b['Nick']

b.Nick

np.mean(b)  
print(np.mean(b))

a = pd.Series(data=[1, 2, 3], index=[1, 2, 3])
b = pd.Series(data=[1, 2, 3], index=[2, 3, 4])  
pd.DataFrame({'a': a, 'b': b})

df = pd.DataFrame()
df['Subject'] = ['bob', 'amy', 'bill']
df['Acc'] = [0.8, 0.7, 0.9]
df['isRat'] = [True, False, False]
print(type(df))
df

data = np.arange(20).reshape(10, 2)
pd.DataFrame(data, columns=['Speed', 'Accuracy'])

print([fun for fun in dir(pd) if 'in' in fun])

df['Subject']

df.Subject

df[['Subject', 'Acc']]

dfs = df.set_index('Subject')
dfs

dfs.index

dfs.loc['nate'] = 1
dfs['new'] = 'hi'
dfs

dfs.iloc[1]
dfs.iloc[0:2]

print(df['Subject'])
df['Subject'][2] = 'Harry'

df.ix[2, 'Subject'] = 'Harry'
df

df.T

df[df['isRat'] == False]

get_ipython().run_cell_magic('HTML', '', '<iFrame src="http://pandas.pydata.org/pandas-docs/stable/visualization.html" width=900 height=400></iFrame>')

df.plot.bar(x='Subject', y='Acc', title='Subject Accuracy')

dfs.plot.bar(y='Acc', title='Subject Accuracy')

fig, axes = plt.subplots(nrows=1, ncols=2)
df.plot.bar(x='Subject', y='Acc', ax=axes[0])
df.plot.hist(y='Acc', ax=axes[1])
axes[1].set_title('Accuracy Histogram')

print([fun for fun in dir(df) if 'to_' in fun[:3]])

users = pd.read_csv('users.csv', delimiter='|')
users

users.head(3)

st = users['Job'].describe()
st

users.Sex.describe()

users.groupby('Sex').Age.describe(percentiles=[.5])

for sex, data in users.groupby('Sex'):
    print(sex)
    print(data.head(2))  
    print('')

users.groupby('Sex').Age.hist(alpha=.4, bins=15)

ages = users.groupby('Sex').Age
ages.mean().plot.bar(yerr=ages.std(), rot=0, title='Mean Subject Age by Sex')

users.groupby('Job').count().Num.plot.bar(rot=70, title='Total Users by Job')



