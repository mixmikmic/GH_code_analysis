import os
import glob

os.getcwd()

path = 'C:\\Users\\Dell\\Downloads'

extension = 'csv'
os.chdir(path)

result = [i for i in glob.glob('*.{}'.format(extension))]
print(result)

import pandas as pd

os.listdir()

diamonds=pd.read_csv("C:\\Users\\Dell\\Downloads\\BigDiamonds.csv\\BigDiamonds.csv")

type(diamonds)

len(diamonds)

diamonds.columns

diamonds.shape

diamonds.info()

diamonds.head()

diamonds2=diamonds.copy()

pd.value_counts(diamonds3.cut)

diamonds.describe()

diamonds = diamonds.notnull() * 1

diamonds.head()

diamonds=diamonds.drop('Unnamed: 0',1)

diamonds.head()

diamonds2.head()

diamonds3=diamonds2.copy()

diamonds2.fillna("AJAY").head()

diamonds2=diamonds2.dropna(how="any")

diamonds2.info()

data=diamonds3.values
data

diamonds3.columns

g=pd.DataFrame(data=data[0:,0:],    # values
              index=range(0,len(data)),    # 1st column as index
              columns=diamonds3.columns[0:])  # 1st row as the column names

g.head()

diamonds3.iloc[2:5,:]

diamonds3.iloc[:,2:5]

diamonds3[['cut','color','clarity']].head()

diamonds3.ix[20:40]

diamonds3.corr()

diamonds3.head()

diamonds3.drop(diamonds3.index[[1,3]]).head()

s=pd.Series(range(0,100))

type(s)

diamonds3.drop(diamonds3.index[[s]]).head()

del diamonds

diamonds3.query('carat >.50 and price >3000')

del diamonds3["Unnamed: 0"]

diamonds3.query('price >5000')

diamonds2.query('color=="J" or price >4000')

diamonds3['newvar']=1

diamonds3.head()

diamonds3.loc[diamonds3.price>=5000,'newvar']="Expensive"

diamonds3.query('price >5000').head()

diamonds3['ppc']=diamonds3.price/diamonds3.carat

diamonds3.head()

diamonds4=diamonds3.copy()

diamonds3=diamonds3.dropna(how='any')

diamonds3.head()

os.listdir()

result = [i for i in glob.glob('*.{}'.format(extension))]
print(result)

f=pd.read_csv('ccFraud.csv')

f.dtypes

f.index

f.columns

f.values

f.describe()

f.T

f.sort(columns='balance')

f.sort_index(axis=0, ascending=False)

f.sort_index(axis=1)

f.head()

f.tail(2)

f['balance']

f[1:3]

f.loc[:,['balance' , 'gender' ]]

f[['balance' , 'gender' ]]

f[f['balance'] > 3000]



