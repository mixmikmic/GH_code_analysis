import os
import glob

os.getcwd()

path = 'C:\\Users\\Dell\\Downloads'

extension = 'csv'
os.chdir(path)

result = [i for i in glob.glob('*.{}'.format(extension))]
print(result)

import pandas as pd

iris=pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv")

iris.info()

import seaborn as sns
get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt 
get_ipython().magic('matplotlib inline')

plt.bar(iris['Sepal.Length'],iris['Sepal.Width'],label="bar1",color='r')

plt.bar(iris['Petal.Length'],iris['Petal.Width'],label="bar1",color='g')

fig=plt.figure()

ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)

ax1.boxplot(iris['Sepal.Length'])
ax1.set_xlabel('Sepal.Length')
plt.show()


ax2.boxplot(iris['Petal.Length'])
ax2.set_xlabel('Petal.Length')
plt.show()

plt.boxplot(iris['Petal.Length'])

plt.hist(iris['Sepal.Length'])

plt.scatter(iris['Petal.Length'],iris['Sepal.Length'])



slices=pd.value_counts(iris.Species)
print(slices)

labels=pd.Series(iris.Species.unique())
print(labels)

colors=['r','y','g']

plt.pie(pd.value_counts(iris.Species),labels=['virginica','versicolor','setosa'],colors=['r','y','g'],autopct='%1.1f%%')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
np.random.seed(sum(map(ord, "aesthetics")))

os.listdir()

def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

sinplot()

sns.set_style("white")

sinplot()

sns.set_style("ticks")
sinplot()

sns.palplot(sns.color_palette())

sns.palplot(sns.color_palette("hls",8))

sns.palplot(sns.color_palette("BuGn", 10))

sinplot()

diamonds=pd.read_csv("C:\\Users\\Dell\\Downloads\\BigDiamonds.csv\\BigDiamonds.csv")

type(diamonds)

len(diamonds)

diamonds.columns

diamonds.shape

diamonds.info()

diamonds.head()

diamonds2=diamonds.copy()

pd.value_counts(diamonds2.cut)

diamonds.describe()

diamonds=diamonds.drop("Unnamed: 0",1)

diamonds=diamonds.dropna(how="any")

sns.distplot(diamonds.price, bins=20, kde=True, rug=False)

sns.distplot(diamonds.price, bins=20, kde=False, rug=False)

sns.boxplot(x="color", y="price", data=diamonds)

sns.jointplot('price','carat',data=diamonds2)

sns.factorplot(x="color", y="price",
col="cut", data=diamonds, kind="box", size=4, aspect=.5);

from ggplot import *


p + geom_point()

p + geom_point() +facet_grid('cut')

p = ggplot(aes(x='price', y='carat',color="cut"), data=diamonds)
p + geom_point()

p = ggplot(aes(x='price', y='carat'), data=diamonds)
p

p = ggplot(aes(x='price', y='carat',color="clarity"), data=diamonds)
p + geom_point()

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



