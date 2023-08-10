get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('classic')
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12.5, 6.0)
import numpy as np
import pandas as pd

x = np.linspace(0, 10, 500)
y = np.array([np.sin(x), np.cos(x), np.sin(x+1)]).T
plt.plot(x, y)
plt.legend(['sin(x)', 'cos(x)', 'sinc(x+1)']);

import seaborn as sns
sns.set()

plt.plot(x, y)
plt.legend(['sin(x)', 'cos(x)', 'sinc(x+1)']);

iris = sns.load_dataset('iris')
iris.head()

ax = plt.axes()
ax.hist(iris.sepal_length, alpha=0.5)
ax.hist(iris.petal_length, alpha=0.5)
ax.hist(iris.petal_width, alpha=0.5)
ax.legend();

ax = plt.axes()
sns.kdeplot(iris.sepal_length, shade=True, ax=ax)
sns.kdeplot(iris.petal_length, shade=True, ax=ax)
sns.kdeplot(iris.petal_width, shade=True, ax=ax)
ax.set_ylim((0, 0.5));  # KDE is always normalised

ax = plt.axes()
sns.distplot(iris.sepal_length, ax=ax)
sns.distplot(iris.petal_length, ax=ax)
sns.distplot(iris.petal_width, ax=ax);
ax.set_ylim((0, 0.7));

sns.kdeplot(iris.sepal_width, iris.sepal_length);

sns.jointplot(iris.sepal_width, iris.sepal_length);

sns.jointplot(iris.sepal_width, iris.sepal_length, kind='kde');

mean = [0, 0]
cov = [[3, 1],
       [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10240).T
sns.jointplot(x, y, kind='hex');

planets = sns.load_dataset('planets')
planets.head()

sns.jointplot(planets.orbital_period, planets.mass, kind='reg');

sns.pairplot(iris, hue='species');

titanic = sns.load_dataset('titanic')
titanic.head()

grid = sns.FacetGrid(titanic, row='sex', col='class', margin_titles=True)
grid.map(plt.hist, 'age');

sns.factorplot('class', 'fare', 'sex', data=titanic, kind='box', size=12);

sns.factorplot('class', 'fare', 'sex', data=titanic, kind='violin', size=12);

sns.factorplot('year', data=planets, kind='count', aspect=3);

sns.factorplot('year', data=planets, hue='method', kind='count', aspect=2, size=6);

