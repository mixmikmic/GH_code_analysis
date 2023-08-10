import pandas as pd
import numpy as np

drug = pd.read_csv('https://github.com/JamesByers/Datasets/raw/master/drug-use-by-age.csv')

drug.head()

drug.tail()

drug.shape

drug.columns

drug['crack-use'].head(3)

drug[['crack-use']].head()

type(drug['crack-use'])

type(drug[['crack-use']])

drug[['crack-use']].columns

drug[['age','crack-use']].head()

drug.age.head()  #### Remember this will be a Series, not a DataFrame.

drug.info()

drug.describe()



drug.mean()

dia = pd.read_csv('https://github.com/JamesByers/Datasets/raw/master/diamonds.csv')

dia.head()

dia.iloc[0:,1:]

dia.head()

dia.shape

dia.info()

dia.describe()

dia.index = dia['cut']

dia.loc['Ideal','carat'].sum()

dia.reset_index(drop=True, inplace=True)

dia.head()

new_index_values = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q']

drug.index = new_index_values
drug.head()



subset = drug.loc['B':'F',['marijuana-use','marijuana-frequency']]
subset

drug.iloc[[1,2,3,4,5],[4,5]]

drug.ix[['B','C'],['marijuana-use','marijuana-frequency']]

drug.index=drug['age']
drug.head()

drug.ix['26-29',[4,5]]

drug.reset_index(drop=True, inplace=True)

drug.head()

mydata = pd.DataFrame({'Letters':['A','B','C'], 'Integers':[1,2,3], 'Floats':[2.2, 3.3, 4.4]}, index=mydata['Letters'])
mydata

mydata.rename(columns={'Integers':'Ints'},inplace=True)
mydata

mydata.dtypes, mydata.columns

mydata.columns=['A','B','C']
mydata

mydata.ix[1,'B'] = 100
mydata

mydata.ix[:,'A'] = 'foo'
mydata

mydata.ix[0,['B','C']]  = [1000,1]
mydata

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

drug.plot(x='age',y='crack-use')

drug.hist('marijuana-use')

drug[drug['marijuana-use']>25][['age','marijuana-frequency']]

drug[(drug['marijuana-use']>20) & (drug['n'] > 4000)]



