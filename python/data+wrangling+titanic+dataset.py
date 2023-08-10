import IPython
print(IPython.sys_info())

import pandas as pd

titanic=pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Titanic.csv")

titanic.info()

titanic.head()

titanic=titanic.drop('Unnamed: 0',1)

titanic.head()

titanic2=titanic.copy()

pd.value_counts(titanic.PClass)

pd.value_counts(titanic.Sex)

pd.value_counts(titanic.Survived)

titanic.iloc[1:3,:]

titanic.head(7)

titanic[['PClass','Age','SexCode']].head()

titanic.Age.head()

tpy=titanic.values

tpy

import os as os

os.getcwd()

os.chdir('C:\\Users\\Dell\\Desktop')

os.listdir()

titanic.to_csv('C:\\Users\\Dell\\Desktop\\titanic2.csv', index=False)

os.listdir()

titanic.head()

titanic.query("PClass=='1st' and Survived ==1")

193/322

titanic.query("PClass=='3rd' and Survived==1").count()

138/711

pd.crosstab(titanic.PClass,titanic.Survived)

pd.crosstab(titanic.PClass,titanic.Survived,margins=True)

pd.crosstab(titanic.PClass,titanic.Survived,normalize='index')

titanic.query("PClass=='1st' and Sex=='female'").count()

titanic.query("PClass=='1st' and Sex=='female' and Survived==1").count()

134/143

titanic.query("PClass=='3rd' and Sex=='male' and Survived==1").count()

titanic.query("PClass=='3rd' and Sex=='male' ").count()

58/499

pd.crosstab([titanic.PClass, titanic.Sex], titanic.Survived,  margins=True)

titanic2.head()

titanic.loc[titanic.Survived==1,'Survived2']='Alive'

titanic.loc[titanic.Survived!=1,'Survived2']='Dead'

titanic.head()

import numpy as np

titanic = titanic.assign(e=pd.Series(np.random.randn(len(titanic))).values)

titanic.head()

type(titanic.Name)



