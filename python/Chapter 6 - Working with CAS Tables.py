import swat

conn = swat.CAS('server-name.mycompany.com', 5570, 'username', 'password')

import pandas as pd

df = pd.read_csv('/u/username/data/iris.csv')

df.columns

tbl = conn.loadtable('data/iris.csv', caslib='casuser').casTable

tbl.columns

df.dtypes

tbl.dtypes

df.get_dtype_counts()

tbl.get_dtype_counts()

tbl.size

tbl.shape

tbl.info()

tbl.head(n=3)

tbl.tail(n=4)

tbl.head(columns=['sepal_length', 'petal_length'])

desc = df.describe()
desc

type(desc)

casdesc = tbl.describe()
casdesc

type(casdesc)

# conn.tableinfo('datasources.megacorp5m')

# %time mega.describe()

tbl.describe(percentiles=[0.3, 0.8])

tbl.describe(include='character')

tbl.describe(include='all')

tbl.describe(include=['numeric', 'character'])

tbl.describe(stats=['count', 'nmiss', 'sum', 'probt', 'freq'])

tbl.describe(stats='all')

tbl.count()

tbl.mean()

tbl.probt()

get_ipython().magic('matplotlib inline')

# from matplotlib.pyplot import show

tbl.plot()

tbl[['sepal_length', 'sepal_width']].plot()

tbl[['sepal_length', 'sepal_width']].plot(kind='area')

tbl[['sepal_length', 'sepal_width']].plot.area()

tbl.save(name='data/irisout.csv', caslib='casuser')

tbl.save(name='data/irisout.sashdat', caslib='casuser')

tbl.sort_values(['sepal_length', 'sepal_width'])

sorttbl = tbl.sort_values(['sepal_length', 'sepal_width'])
sorttbl

sorttbl.head(10)

sorttbl.tail(5)

sorttbl = tbl.sort_values(['sepal_length', 'sepal_width'], ascending=[False, True])
sorttbl

sorttbl.head(10)

for col in sorttbl:
    print(col)

for i, col in enumerate(sorttbl):
    print(i, col)

for col, dtype in zip(sorttbl, sorttbl.dtypes):
    print(col, dtype)

for col, obj in sorttbl.iteritems():
    print(col, obj)
    print('')

for row in sorttbl.iterrows():
    print(row)

for row in sorttbl.itertuples():
    print(row)

col = sorttbl['sepal_width']
col

col.head()

sorttbl.species

widths = sorttbl[['sepal_width', 'petal_width', 'species']]
widths

widths.head()

widths.describe()

widths.columninfo()

sorttbl

sorttbl.loc[:, 'petal_width'].head()

sorttbl.loc[:, 'sepal_length':'petal_length'].head()

sorttbl.loc[:, ['petal_width', 'sepal_width']].head()

sorttbl.loc[:, 3].head()

sorttbl.iloc[:, 0:3].head()

sorttbl.iloc[:, [3, 1]].head()

sorttbl.ix[:, [3, 'sepal_width']].head()

sorttbl.ix[:, 'sepal_width'].head()

sorttbl.ix[:, 'sepal_width':-2].head()

sorttbl.ix[:, ['sepal_width', 3, 4]].head()

expr = sorttbl.petal_length > 6.5

expr.head()

newtbl = sorttbl[expr]
newtbl.head()

newtbl = sorttbl[sorttbl.petal_length > 6.5]
newtbl.head()

newtbl2 = newtbl[newtbl.petal_width < 2.2]
newtbl2.head()

sorttbl[(sorttbl.petal_length > 6.5) & 
        (sorttbl.petal_width < 2.2)]

sorttbl[(sorttbl.petal_length + sorttbl.petal_width) * 2 > 17.5].head()

sorttbl[sorttbl.species.str.upper().str.startswith('SET')].head()

sorttbl['sepal_factor'] = ((sorttbl.sepal_length + sorttbl.sepal_width) * 2)
sorttbl.head()

sorttbl['total_factor'] = sorttbl.sepal_factor + sorttbl.petal_width + sorttbl.petal_length
sorttbl.head()

sorttbl['names'] = 'sepal / petal'
sorttbl.head()

sorttbl['cap_names'] = sorttbl.names.str.title()
sorttbl.head()

tbl.set_param('groupby', ['species'])
tbl

tbl.summary(subset=['min', 'max'])

tbl.del_param('groupby')
tbl

grptbl = tbl.groupby(['species'])
grptbl

grptbl.summary(subset=['min', 'max'])

grpsumm = grptbl.summary(subset=['min', 'max'])
grpsumm.concat_bygroups()

grpcorr = grptbl.correlation()
grpcorr

grpcorr.get_tables('Correlation')

swat.concat(grpcorr.get_tables('Correlation'))

grpsumm.get_group(['versicolor'])

grpsumm.get_group(species='versicolor')

grpmdsumm = tbl.mdsummary(sets=[dict(groupby=['sepal_length']),
                                dict(groupby=['petal_length'])])

list(grpmdsumm.keys())

grpmdsumm.get_set(1)

grpmdsumm.get_set(1).concat_bygroups()

conn.close()



