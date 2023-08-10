import pandas as pd
import numpy as np

df = pd.DataFrame({'key1' : ['a','a','b','b','a'],
                   'key2' : ['one', 'two', 'one', 'two', 'one'],
                   'data1' : np.random.randn(5),
                   'data2' : np.random.randn(5)
                  })

df

df.groupby('key1').mean()

df.groupby(['key1', 'key2']).mean()

df.groupby(['key1', 'key2']).size()

df.mean()

df['data1'].groupby(df['key1']).mean()

df.groupby(['key1', 'key2']).sum()

get_ipython().magic('pinfo df.groupby')

df['data1'].groupby([df['key1'], df['key2']]).mean()

df.groupby([df['key1'], df['key2']]).size()

df.groupby('key1')[['data1']].mean()

df.groupby([df['key1'], df['key2']]).size().unstack()

df

states = np.array(['Ohio', 'Iowa', 'Iowa', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])

df['data1'].groupby([states, years]).mean()

df['data1'].groupby([states, years]).size()

for name, group in df.groupby('key1'):
    print name, group
    

for name, group in df.groupby(['key1', 'key2']):
    print name, group

get_ipython().magic('pinfo df.groupby')

for name, group in df['data1'].groupby(df['key2']):
    print name, group

df.groupby('key1').last()

df.groupby('key1').first()

df.groupby('key1').max()

df.groupby('key2').min()

df['key1'].min()

# Groupby external dictionary

people = pd.DataFrame(np.random.randn(5, 5),
                     columns = ['a', 'b', 'c', 'd', 'e'],
                     index = ['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])

people

mapping = {'a': 'red',
          'b': 'red',
          'c': 'blue',
          'd': 'blue',
          'e': 'red',
          'f': 'orange'}

mapping

people.groupby(mapping, axis = 1).sum()

people.groupby(['X', 'Y', 'X', 'X', 'Y']).sum()

df_new = pd.read_csv('df.csv')

df_new.groupby('category').apply(lambda g: np.average(g['data'], weights=g['weights']))

aapl = pd.read_csv('aapl.csv')

aapl['size'] = aapl['Strike'] * aapl['Vol']

aapl.loc[aapl['Type']=='call', 'size'].sum()/aapl.loc[aapl['Type']=='call', 'Vol'].sum()

aapl.loc[aapl['Type']=='put', 'size'].sum() / aapl.loc[aapl['Type']=='put', 'Vol'].sum()

get_ipython().magic('pinfo np.average')

aapl.groupby('Type').apply(lambda g: np.average(g['Strike'], weights=g['Vol']))

aapl.head()

aapl.groupby('Type').apply(lambda g: np.average(g['Strike'], weights=g['Open_Int']))

tips = pd.read_csv('tips.csv')

tips[-4:]

tips.pivot_table(index = ['gender', 'smoker'])

tips.pivot_table('tip', index = ['gender', 'smoker'])

tips.pivot_table('total_bill', index = ['gender', 'smoker'])

tips.pivot_table(index=['day'])

tips.pivot_table(index=['day', 'time'])

tips.pivot_table(index=['time'])

tips.groupby(['day', 'time']).size()

tips.groupby(['day', 'time']).mean()

get_ipython().magic('pinfo tips.pivot_table')

tips.pivot_table(index = ['gender', 'smoker'], aggfunc=np.sum)

tips['tip_pct'] = tips['tip'] / tips['total_bill']

tips

tips.pivot_table(['tip_pct', 'size'], 
                index = ['gender', 'day'],
                columns = ['smoker'])

tips.pivot_table(['tip_pct', 'size'], 
                index = ['gender', 'day'],
                columns = ['time', 'smoker'])

tips.pivot_table(['tip', 'size'], 
                index = ['gender', 'day'],
                columns = ['time', 'smoker'])

tips.pivot_table(index = ['day', 'time'])

tips.pivot_table('tip', index = ['day', 'time'],
                columns =['gender', 'smoker'])



