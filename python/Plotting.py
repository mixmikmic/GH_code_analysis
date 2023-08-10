import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import random
get_ipython().run_line_magic('matplotlib', 'inline')

x = [1,2,3]
y = [1,4,9]
plt.plot(x,y)
plt.show()

x = np.asarray(range(-2000,2000))
y = x**2
plt.plot(x,y)
plt.show()

x = np.linspace(-9,9,100)
y = np.sin(x)
z = np.cos(x)
plt.plot(x, y , label=' y = sin(x) ')
plt.plot(x, z , label=' y = cos(x) ')
plt.legend()
plt.show()

x = np.random.normal(size=100)
sns.distplot(x)

#x = np.linspace(1,100,1000)
#sns.distplot(x)

x = []
for _ in range(100):
    x.append(random.randint(1,100))
sns.distplot(x)

iris = sns.load_dataset('iris')

sns.distplot(iris['sepal_length'])

sns.distplot(iris['sepal_width'])

plt.scatter(iris['sepal_length'], iris['sepal_width'])
plt.show()

sns.jointplot(iris['sepal_length'], iris['sepal_width'])

sns.jointplot(iris['sepal_length'], iris['sepal_width'], kind='kde')

titanic = sns.load_dataset('titanic')

titanic

sns.kdeplot(titanic['fare'], shade=True)

sns.kdeplot(titanic.pclass[titanic['survived']==1], color='blue', label="Survived")
sns.kdeplot(titanic.pclass[titanic['survived']==0], color='green', label="Not-Survived")

sns.barplot(y=titanic['survived'], x =titanic['who'])

sns.barplot(y=titanic['survived'], x =titanic['who'], hue=titanic['class'])

sns.countplot(x = titanic['who'])

sns.countplot(x = titanic['who'], hue=titanic['class'])

sns.heatmap(iris.corr(), )

sns.heatmap(titanic.corr(), annot=True)



