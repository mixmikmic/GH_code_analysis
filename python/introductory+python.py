#comments in Python
'''multiple lines of comments are being shown here'''

2+3+5

66-3-(-4)

32*3

2**3

2^3

43/3

43//3

43%3

import math as mt

mt.exp(2)

mt.log(10)

mt.exp(1)

mt.log(8,2)

mt.sqrt(1000)

import numpy as np

np.std([23,45,67,78])

dir(mt)

type(1)

type("Ajay")

type([23,45,67])

a=[23,45,67]

len(a)

np.std(a)

np.var(a)

123456789123456789*9999999999999999

get_ipython().magic('pinfo2 np.random')

from random import randrange,randint

print(randint(0,90))

randrange(1000)

for x in range(0,10):
    print(randrange(10000000000000000))

def mynewfunction(x,y):
    taxes=((x-1000000)*0.35+100000-min(y,100000))
    print(taxes)

mynewfunction(2200000,300000)

import os as os

get_ipython().magic('pinfo2 os')

for x in range(0,30,6):
    print(x)

def mynewfunction(x,y):
    z=x**3+3*x*y+20*y
    print(z)

for x in range(0,30,6):
    mynewfunction(x,10)

import os as os

os.getcwd()

os.listdir()

os.chdir('C:\\Users\\Dell')

mystring='Hello World'

mystring

mystring[1]

mystring[0]

print(mystring)

type(mystring)

len(mystring)

newstring2='Aye aye me heartie\'s'

newstring3="Aye aye me heartie's"

10*newstring3

ne1= "'Ajay','Vijay','Anita','Ankit'"

type(ne1)

str(ne1)

ne1[1]

ne2= ['Ajay','Vijay','Anita','Ankit']

str(ne2)

ne2[1]

myname1='Ajay'
myname2='John'

message= "Hi I am %s howdy"

message %myname1

message %myname2

ne2

ne2.append('Anna')

ne2

del ne2[0]

ne2

ne3=('Sachin','Dhoni','Gavaskar','Kapil')

dir(ne3)

favourite_movie=['micky mouse,steamboat willie', 'vijay,slumdog millionaire', 'john,passion of christ', 'donald,arthur']


type(favourite_movie)

favourite_movie2={'micky mouse:steamboat willie', 'vijay:slumdog millionaire', 'john:passion of christ', 'donald:arthur'}

type(favourite_movie2)

favourite_movie3={'micky mouse':'steamboat willie', 'vijay':'slumdog millionaire', 'john':'passion of christ', 'donald':'arthur'}

type(favourite_movie3)

favourite_movie3['micky mouse']


import re

names =["Anna", "Anne", "Annaporna","Shubham","Aruna"]

for name in names:
    print(re.search(r'(An)',name))

for name in names:
    print(re.search(r'(A)',name))

for name in names:
    print(re.search(r'(a)',name))

for name in names:
    print(bool(re.search(r'(a)',name)))

import numpy as np

numlist=["$10000","$20,000","30,000",40000,"50000   "] 

for i,value in enumerate(numlist):
    print(i)       
    print(value)

for i,value in enumerate(numlist):
 
    numlist[i]=re.sub(r"([$,])","",str(value))
    numlist[i]=int(numlist[i])

numlist

np.mean(numlist)

from datetime import datetime

datetime.now()

date_obj=datetime.strptime("15/August/2007","%d/%B/%Y")

date_obj

a=date_obj-datetime.now()

a.days

a.seconds

os.getcwd()

import IPython 
print (IPython.sys_info())

get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information')

os.chdir('C:\\Users\\Dell\\Downloads')

os.listdir()

import glob as glob

path = os.getcwd()
extension = 'csv'
os.chdir(path)
result = [i for i in glob.glob('*.{}'.format(extension))]
print(result)

import pandas as pd

fraud=pd.read_csv('ccFraud.csv')

mtcars=pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")

smalldiamonds=pd.read_csv("C:\\Users\\Dell\\Desktop\\Diamond (8).csv")

fraud.columns

fraud.shape

len(fraud)

len(fraud.columns)

fraud.dtypes

fraud.info()

mtcars.info()

smalldiamonds.info()

fraud.head()

fraud.tail()

fraud2=fraud.copy()

fraud.describe()

fraud.gender.describe()

mtcars.head()

mtcars=mtcars.drop("Unnamed: 0",1)

mtcars.head()

import IPython
print (IPython.sys_info())



get_ipython().system('pip install version_information')
get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information')


get_ipython().system('pip freeze')

get_ipython().system('pip install guppy')

fraud.head()

fraud.head().gender

fraud.gender.head()

fraud['gender'].head()

fraud[['gender','state','balance']].head()

fraud.ix[10:20]

fraud.iloc[:,:]

fraud.iloc[10:20,1:4]

fraud.describe()

fraud.gender.value_counts()

fraud.state.value_counts()

fraud.fraudRisk.value_counts()

pd.crosstab(fraud.fraudRisk,fraud.gender)

pd.crosstab(fraud.fraudRisk,fraud.gender,margins=True)

np.random.choice(100,10)

a=len(fraud)

b=0.0001

a*b

fraud.ix[np.random.choice(len(fraud),a*b)]

get_ipython().system(' pip install pandasql')

from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

mtcars.head()

pysqldf("SELECT * FROM mtcars  LIMIT 10;")

pysqldf("SELECT * FROM mtcars  WHERE gear > 4;")

pysqldf("SELECT AVG(mpg),gear FROM mtcars group by gear  ;")

mtcars.mpg.mean()

g1=pd.groupby(mtcars,mtcars.gear)

g1.mean()

mtcars.gear.value_counts()

mtcars.cyl.unique()

pd.crosstab(mtcars.gear,mtcars.cyl)

mtcars.pivot_table(index='gear', columns='cyl', values='mpg', fill_value=0)

fraud.head()

del fraud['custID']

fraud.head()

fraud3=fraud

del fraud['state']

fraud3.head()

wine=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header=None)

wine.info()

wine.columns=['WineClass','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline'] 

wine.info()

wine.head()

wine.WineClass.value_counts()

classby=pd.groupby(wine,wine.WineClass)

classby.mean()

wine.describe()

wine.Ash.describe()



