import matplotlib.pyplot as plt
plt.style.use('classic')
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd

print("tedd")

# Create some data
rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)

# Plot the data with matplotlib defaults
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');

import seaborn as sns
sns.set()

# same plotting code as above!
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');

data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

for col in 'xy':
    plt.hist(data[col], normed=True, alpha=0.5)

for col in 'xy':
    sns.kdeplot(data[col], shade=True)

sns.distplot(data['x'])
sns.distplot(data['y']);

sns.kdeplot(data);

with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='kde');

with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='hex')

iris = sns.load_dataset("iris")
iris.head()

sns.pairplot(iris, hue='species', size=2.5);

tips = sns.load_dataset('tips')
tips.head()

tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15));

with sns.axes_style(style='ticks'):
    g = sns.factorplot("day", "total_bill", "sex", data=tips, kind="box")
    g.set_axis_labels("Day", "Total Bill");

with sns.axes_style('white'):
    sns.jointplot("total_bill", "tip", data=tips, kind='hex')

sns.jointplot("total_bill", "tip", data=tips, kind='reg');

planets = sns.load_dataset('planets')
planets.head()

with sns.axes_style('white'):
    g = sns.factorplot("year", data=planets, aspect=2,
                       kind="count", color='steelblue')
    g.set_xticklabels(step=5)

with sns.axes_style('white'):
    g = sns.factorplot("year", data=planets, aspect=4.0, kind='count',
                       hue='method', order=range(2001, 2015))
    g.set_ylabels('Number of Planets Discovered')

# !curl -O https://raw.githubusercontent.com/jakevdp/marathon-data/master/marathon-data.csv

data = pd.read_csv('marathon-data.csv')
data.head()

data.dtypes

def convert_time(s):
    h, m, s = map(int, s.split(':'))
    return pd.datetools.timedelta(hours=h, minutes=m, seconds=s)

data = pd.read_csv('marathon-data.csv',
                   converters={'split':convert_time, 'final':convert_time})
data.head()

data.dtypes

data['split_sec'] = data['split'].astype(int) / 1E9
data['final_sec'] = data['final'].astype(int) / 1E9
data.head()

with sns.axes_style('white'):
    g = sns.jointplot("split_sec", "final_sec", data, kind='hex')
    g.ax_joint.plot(np.linspace(4000, 16000),
                    np.linspace(8000, 32000), ':k')

data['split_frac'] = 1 - 2 * data['split_sec'] / data['final_sec']
data.head()

sns.distplot(data['split_frac'], kde=False);
plt.axvline(0, color="k", linestyle="--");

sum(data.split_frac < 0)

g = sns.PairGrid(data, vars=['age', 'split_sec', 'final_sec', 'split_frac'],
                 hue='gender', palette='RdBu_r')
g.map(plt.scatter, alpha=0.8)
g.add_legend();

sns.kdeplot(data.split_frac[data.gender=='M'], label='men', shade=True)
sns.kdeplot(data.split_frac[data.gender=='W'], label='women', shade=True)
plt.xlabel('split_frac');

sns.violinplot("gender", "split_frac", data=data, palette=["lightblue", "lightpink"]);

data['age_dec'] = data.age.map(lambda age: 10 * (age // 10))
data.head()

men = (data.gender == 'M')
women = (data.gender == 'W')

with sns.axes_style(style=None):
    sns.violinplot("age_dec", "split_frac", hue="gender", data=data,
                  split=True, inner="quartile", palette=["lightblue", "lightpink"]);

(data.age > 80).sum()

g = sns.lmplot('final_sec', 'split_frac', col='gender', data=data,
               markers=".", scatter_kws=dict(color='c'))
g.map(plt.axhline, y=0.1, color="k", ls=":");

